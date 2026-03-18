// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ttl-convert-ttl-to-compute"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLCONVERTTTLTOCOMPUTE
#include "ttlang/Dialect/TTL/Passes.h.inc"

static RankedTensorType getTensorType(Value v) {
  return dyn_cast<RankedTensorType>(v.getType());
}

static Value buildInitTensor(OpBuilder &b, Location loc, RankedTensorType type,
                             Value exemplar) {
  SmallVector<Value> dynDims;
  for (auto dim : llvm::enumerate(type.getShape())) {
    if (dim.value() == ShapedType::kDynamic) {
      dynDims.push_back(tensor::DimOp::create(b, loc, exemplar, dim.index()));
    }
  }
  return tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType(),
                                 dynDims);
}

/// Collect all unique output CBs from store users of an op's result.
/// Preserves first-seen order and deduplicates (same CB stored to twice
/// produces one output). Returns empty if no stores exist or if any
/// store's view is not from cb_reserve.
static SmallVector<Value> collectOutputCBs(Operation *op) {
  assert(op->getNumResults() > 0 &&
         "collectOutputCBs requires op with results");
  SmallVector<Value> result;
  DenseSet<Value> seen;
  for (OpOperand &use : op->getResult(0).getUses()) {
    if (auto storeOp = dyn_cast<StoreOp>(use.getOwner())) {
      auto reserve = storeOp.getView().getDefiningOp<CBReserveOp>();
      if (!reserve) {
        return {};
      }
      Value cb = reserve.getCb();
      if (seen.insert(cb).second) {
        result.push_back(cb);
      }
    }
  }
  return result;
}

/// Find the last block-level store that uses this op's result.
/// Used to position the compute op after all reserves (which precede their
/// stores) so that reserve views dominate the compute body.
static StoreOp findLastStore(Operation *op) {
  if (op->getNumResults() == 0) {
    return {};
  }
  StoreOp last;
  for (OpOperand &use : op->getResult(0).getUses()) {
    if (auto s = dyn_cast<StoreOp>(use.getOwner())) {
      if (!last || last->isBeforeInBlock(s)) {
        last = s;
      }
    }
  }
  return last;
}

/// Position the rewriter before the last store so that the new compute op
/// is placed after all reserves (which precede their stores).
static void insertAtLastStore(PatternRewriter &rewriter, Operation *op) {
  StoreOp lastStore = findLastStore(op);
  assert(lastStore && "insertAtLastStore called but op has no store users; "
                      "callers must verify via collectOutputCBs first");
  rewriter.setInsertionPoint(lastStore);
}

/// Create tile_store(s) in the compute body for the given tile result and
/// erase the corresponding block-level stores. Populates tile_store indices
/// from iter_index ops and the output indexing map.
static void emitTileStores(PatternRewriter &rewriter, Location loc,
                           Value tileResult, Operation *sourceOp) {
  assert(sourceOp->getNumResults() > 0 &&
         "emitTileStores requires op with results");

  // Find the parent ComputeOp from the current insertion point.
  auto *insertBlock = rewriter.getInsertionBlock();
  auto computeOp = dyn_cast<ComputeOp>(insertBlock->getParentOp());
  assert(computeOp && "emitTileStores must be called inside a compute body");

  SmallVector<Value> iterIndices = getOrCreateIterIndices(rewriter, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  size_t numInputs = computeOp.getNumInputs();

  // Build CB -> output index mapping for multi-output disambiguation.
  size_t numOutputs = computeOp.getNumOutputs();
  DenseMap<Value, size_t> cbToOutputIdx;
  if (numOutputs > 1) {
    for (auto [idx, output] : llvm::enumerate(computeOp.getOutputs())) {
      Value cb = getAttachedCB(output);
      if (cb) {
        cbToOutputIdx[cb] = idx;
      }
    }
  }

  // Collect-then-erase: cannot erase stores while iterating getUses().
  SmallVector<StoreOp> storesToErase;
  for (OpOperand &use : sourceOp->getResult(0).getUses()) {
    auto storeOp = dyn_cast<StoreOp>(use.getOwner());
    if (!storeOp) {
      continue;
    }

    // Determine output index for this store's view CB.
    size_t outputIdx = 0;
    if (numOutputs > 1) {
      Value viewCB = getAttachedCB(storeOp.getView());
      if (viewCB) {
        auto it = cbToOutputIdx.find(viewCB);
        if (it != cbToOutputIdx.end()) {
          outputIdx = it->second;
        }
      }
    }
    AffineMap outputMap = indexingMaps[numInputs + outputIdx];
    SmallVector<Value> indices =
        applyIndexingMapToIterIndices(rewriter, loc, outputMap, iterIndices);

    TileStoreOp::create(rewriter, loc, tileResult, storeOp.getView(), indices);
    storesToErase.push_back(storeOp);
  }
  for (StoreOp s : storesToErase) {
    rewriter.eraseOp(s);
  }
}

//===----------------------------------------------------------------------===//
// Tile op emission for fusion
//===----------------------------------------------------------------------===//

/// Emit the tile-level op corresponding to a block-level elementwise op.
/// Returns the result Value, or null on failure.
static Value emitTileOpFor(OpBuilder &b, Location loc, Operation *sourceOp,
                           ValueRange tileOperands, Type tileType) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op>(sourceOp))                                               \
    return TILE_OP::create(b, loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##Op>(sourceOp))                                               \
    return TILE_OP::create(b, loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Fused compute building
//===----------------------------------------------------------------------===//

/// Check if an op is a user signpost or a tile-level dprint that should
/// be pulled into the compute body alongside fused tile ops. Only DST and
/// tile mode dprints need tile-level context; scalar and CB prints stay
/// outside the loop.
static bool isSideEffectOpForCompute(Operation *op) {
  if (auto sp = dyn_cast<SignpostOp>(op)) {
    return sp.getName().starts_with("ttl_");
  }
  if (auto dp = dyn_cast<DPrintOp>(op)) {
    StringRef mode = dp.getMode();
    return mode == "dst" || mode == "tile";
  }
  return false;
}

/// Collect signpost and dprint ops interleaved with fused ops so they can
/// be moved into the compute body. Walks backwards from the first fused op
/// for leading ops, between fused ops for interleaved ones, and forward
/// from the last fused op for trailing ones (stopping at cb_push/cb_pop).
static SmallVector<std::pair<Operation *, Operation *>>
collectInterleavedSideEffectOps(const ElementwiseTraceResult &trace,
                                Operation *sinkOp) {
  DenseSet<Operation *> fusedSet(trace.opsInOrder.begin(),
                                 trace.opsInOrder.end());

  // Find first and last fused ops in block order.
  Operation *firstFused = nullptr;
  Operation *lastFused = nullptr;
  for (auto &op : *sinkOp->getBlock()) {
    if (fusedSet.contains(&op)) {
      if (!firstFused) {
        firstFused = &op;
      }
      lastFused = &op;
    }
  }
  if (!firstFused) {
    return {};
  }

  // Result: pairs of (op, insertAfterThisFusedOp). nullptr means
  // the op is leading (before all fused ops).
  SmallVector<std::pair<Operation *, Operation *>> result;

  // Leading ops: walk backwards from first fused op.
  SmallVector<Operation *> leading;
  for (auto *op = firstFused->getPrevNode(); op; op = op->getPrevNode()) {
    if (isSideEffectOpForCompute(op)) {
      leading.push_back(op);
    } else {
      break;
    }
  }
  for (auto it = leading.rbegin(); it != leading.rend(); ++it) {
    result.push_back({*it, nullptr});
  }

  // Interleaved ops: walk from first to last fused op.
  Operation *prevFused = nullptr;
  for (auto *op = firstFused; op && op != lastFused->getNextNode();
       op = op->getNextNode()) {
    if (fusedSet.contains(op)) {
      prevFused = op;
    } else if (isSideEffectOpForCompute(op)) {
      result.push_back({op, prevFused});
    }
  }

  // Trailing ops: walk forward from last fused op, skipping
  // non-side-effect ops (store, attach_cb) until cb_push/cb_pop.
  for (auto *op = lastFused->getNextNode(); op; op = op->getNextNode()) {
    if (isSideEffectOpForCompute(op)) {
      result.push_back({op, lastFused});
    } else if (isa<CBPushOp>(op) || isa<CBPopOp>(op)) {
      break;
    }
  }

  return result;
}

/// Build a fused ttl.compute from traced elementwise chain.
/// The trace result contains CB-attached root inputs and ops to fuse.
static LogicalResult buildFusedCompute(Operation *sinkOp,
                                       PatternRewriter &rewriter,
                                       const ElementwiseTraceResult &trace) {
  auto type = getTensorType(sinkOp->getResult(0));
  if (!type) {
    return failure();
  }

  // Find all output CBs via stores on the sink op's result.
  SmallVector<Value> outCbs = collectOutputCBs(sinkOp);
  if (outCbs.empty()) {
    return rewriter.notifyMatchFailure(
        sinkOp, "no output CB found (missing ttl.store or view not from "
                "ttl.cb_reserve)");
  }

  // Collect signpost and dprint ops before they get orphaned by fusion.
  auto sideEffectPairs = collectInterleavedSideEffectOps(trace, sinkOp);

  Location loc = sinkOp->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build indexing maps: broadcast-aware for inputs, identity for output.
  // When an input has size 1 in a dimension but the output doesn't, that
  // dimension is broadcast and the map should project to constant 0.
  // This is required for TilingInterface: without correct maps, subblocking
  // would create out-of-bounds slices on broadcast dimensions.
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    auto inputType = getTensorType(trace.rootInputs[i]);
    if (inputType && inputType.getRank() == type.getRank()) {
      SmallVector<AffineExpr> exprs;
      bool hasBroadcast = false;
      for (int64_t d = 0; d < type.getRank(); ++d) {
        if (inputType.getDimSize(d) == 1 && type.getDimSize(d) != 1) {
          exprs.push_back(getAffineConstantExpr(0, ctx));
          hasBroadcast = true;
        } else {
          exprs.push_back(getAffineDimExpr(d, ctx));
        }
      }
      if (hasBroadcast) {
        maps.push_back(
            AffineMapAttr::get(AffineMap::get(type.getRank(), 0, exprs, ctx)));
      } else {
        maps.push_back(AffineMapAttr::get(identityMap));
      }
    } else {
      maps.push_back(AffineMapAttr::get(identityMap));
    }
  }
  for (size_t i = 0; i < outCbs.size(); ++i) {
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes(type.getRank(),
                                   rewriter.getStringAttr("parallel"));

  // Position compute after all reserves by inserting before the last store.
  insertAtLastStore(rewriter, sinkOp);

  // Create init tensors and attach to output CBs.
  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outCb : outCbs) {
    Value init = buildInitTensor(rewriter, loc, type, trace.rootInputs[0]);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outCb);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(type);
  }

  // Create ttl.compute op
  auto computeOp = ComputeOp::create(
      rewriter, loc, TypeRange(resultTypes), trace.rootInputs.getArrayRef(),
      ValueRange(allInitAttached), rewriter.getArrayAttr(maps),
      rewriter.getArrayAttr(iterTypes));

  // Build the body region
  Block *body = rewriter.createBlock(&computeOp.getBody());
  // TODO(#264): Assumes all inputs/outputs have the same element type (from
  // output). This forces all block arguments to have the output's dtype, which
  // may cause issues when fusing mixed dtype operations (e.g., f32 + bf16).
  Type scalarType = type.getElementType();
  Type tileType = ttcore::TileType::get(scalarType);

  // Add block arguments for each root input + output
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    body->addArgument(tileType, loc);
  }
  for (size_t i = 0; i < outCbs.size(); ++i) {
    body->addArgument(tileType, loc);
  }

  rewriter.setInsertionPointToStart(body);

  // Map tensor values to tile values (for wiring up operands)
  DenseMap<Value, Value> tensorToTile;
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    tensorToTile[trace.rootInputs[i]] = body->getArgument(i);
  }

  // Categorize collected side-effect ops by position relative to fused ops.
  assert(!trace.opsInOrder.empty() &&
         "buildFusedCompute requires non-empty opsInOrder");
  DenseMap<Operation *, SmallVector<Operation *>> opsBefore;
  SmallVector<Operation *> leadingOps;
  SmallVector<Operation *> trailingOps;

  Operation *lastFusedOp = trace.opsInOrder.back();
  for (auto &[sideEffectOp, afterFused] : sideEffectPairs) {
    if (!afterFused) {
      leadingOps.push_back(sideEffectOp);
    } else if (afterFused == lastFusedOp) {
      trailingOps.push_back(sideEffectOp);
    } else {
      // Attach to the next fused op after afterFused.
      bool found = false;
      for (size_t i = 0; i < trace.opsInOrder.size(); ++i) {
        if (trace.opsInOrder[i] == afterFused &&
            i + 1 < trace.opsInOrder.size()) {
          opsBefore[trace.opsInOrder[i + 1]].push_back(sideEffectOp);
          found = true;
          break;
        }
      }
      if (!found) {
        trailingOps.push_back(sideEffectOp);
      }
    }
  }

  // Helper: clone a signpost or dprint op into the compute body.
  auto emitSideEffectOp = [&](Operation *op) {
    if (auto sp = dyn_cast<SignpostOp>(op)) {
      SignpostOp::create(rewriter, sp.getLoc(), sp.getNameAttr(),
                         sp.getIsEndAttr());
    } else {
      rewriter.clone(*op);
    }
  };

  // Emit leading side-effect ops
  for (auto *op : leadingOps) {
    emitSideEffectOp(op);
  }

  // Emit tile ops in topological order with interleaved side-effect ops
  Value finalResult;
  for (Operation *op : trace.opsInOrder) {
    auto it = opsBefore.find(op);
    if (it != opsBefore.end()) {
      for (auto *seOp : it->second) {
        emitSideEffectOp(seOp);
      }
    }

    Value tileResult;

    // Special case: BcastOp reads from CB, needs TileBcastOp
    if (auto bcastOp = dyn_cast<BcastOp>(op)) {
      Value inputTile = tensorToTile[bcastOp.getInput()];
      Value outputTile = body->getArguments().back(); // output block arg
      tileResult = TileBcastOp::create(rewriter, loc, tileType, inputTile,
                                       outputTile, bcastOp.getBcastTypeAttr());
    } else {
      // Elementwise ops
      SmallVector<Value, 2> tileOperands;
      for (Value operand : getElementwiseOperands(op)) {
        auto it2 = tensorToTile.find(operand);
        if (it2 == tensorToTile.end()) {
          return op->emitError(
              "fusion failed: operand not mapped to tile value");
        }
        tileOperands.push_back(it2->second);
      }

      tileResult = emitTileOpFor(rewriter, loc, op, tileOperands, tileType);
      if (!tileResult) {
        return op->emitError("fusion failed: unsupported op type");
      }
    }

    tensorToTile[op->getResult(0)] = tileResult;
    finalResult = tileResult;
  }

  // Emit trailing begin signposts and dprints, then tile stores, then end
  // signposts. This places tile_store inside the innermost signpost scope.
  auto isEndSignpost = [](Operation *op) {
    auto sp = dyn_cast<SignpostOp>(op);
    return sp && sp.getIsEnd();
  };
  auto firstEndIt = llvm::find_if(trailingOps, isEndSignpost);
  for (auto it = trailingOps.begin(); it != firstEndIt; ++it) {
    emitSideEffectOp(*it);
  }

  emitTileStores(rewriter, loc, finalResult, sinkOp);

  for (auto it = firstEndIt; it != trailingOps.end(); ++it) {
    emitSideEffectOp(*it);
  }

  YieldOp::create(rewriter, loc);
  rewriter.replaceOp(sinkOp, computeOp.getResult(0));

  // Erase the fused ops in reverse topological order (sink to roots).
  // This ensures each op's users are erased before the op itself.
  for (auto it = trace.opsInOrder.rbegin(); it != trace.opsInOrder.rend();
       ++it) {
    Operation *op = *it;
    if (op != sinkOp && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  // Erase the original side-effect ops (now cloned into compute body).
  for (auto &[op, _] : sideEffectPairs) {
    rewriter.eraseOp(op);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering to ttl.compute with tile ops
//===----------------------------------------------------------------------===//

/// Build a ttl.compute op with a single binary tile operation in the body.
/// Inputs must already be attached to CBs via ttl.attach_cb.
/// Output CBs are the reserved CBs to which the op's result is stored.
template <typename TileOp>
static LogicalResult buildBinaryCompute(Operation *op,
                                        PatternRewriter &rewriter, Value lhs,
                                        Value rhs) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  // Try direct CB attachment first
  Value lhsCb = getAttachedCB(lhs);
  Value rhsCb = getAttachedCB(rhs);

  // If inputs aren't CB-attached, try fusion
  if (!lhsCb || !rhsCb) {
    auto traceResult = traceElementwiseToRoots(op->getResult(0));
    if (traceResult.failureReason == TraceFailureReason::Success &&
        !traceResult.opsInOrder.empty()) {
      return buildFusedCompute(op, rewriter, traceResult);
    }
    emitFusionFailureDiagnostics(op, traceResult);
    return failure();
  }

  // Find all output CBs via stores on this op's result.
  SmallVector<Value> outCbs = collectOutputCBs(op);
  if (outCbs.empty()) {
    return rewriter.notifyMatchFailure(
        op, "no output CB found (missing ttl.store, view not from "
            "ttl.cb_reserve, or intermediate value handled by fusion)");
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: 2 inputs + N outputs.
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  SmallVector<Attribute> maps(2, AffineMapAttr::get(identityMap));
  for (size_t i = 0; i < outCbs.size(); ++i) {
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes(type.getRank(),
                                   rewriter.getStringAttr("parallel"));

  // Position compute after all reserves by inserting before the last store.
  insertAtLastStore(rewriter, op);

  // Create init tensors and attach to output CBs.
  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outCb : outCbs) {
    Value init = buildInitTensor(rewriter, loc, type, lhs);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outCb);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(type);
  }

  // Inputs are already attached, use them directly.
  // Create ttl.compute op
  auto computeOp = ComputeOp::create(
      rewriter, loc, TypeRange(resultTypes), ValueRange{lhs, rhs},
      ValueRange(allInitAttached), rewriter.getArrayAttr(maps),
      rewriter.getArrayAttr(iterTypes));

  // Build the body region with tile type block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type scalarType = type.getElementType();
  // Create tile type: !ttcore.tile<32x32, dtype>
  Type tileType = ttcore::TileType::get(scalarType);
  body->addArgument(tileType, loc); // lhs tile
  body->addArgument(tileType, loc); // rhs tile
  for (size_t i = 0; i < outCbs.size(); ++i) {
    body->addArgument(tileType, loc); // output tile
  }

  rewriter.setInsertionPointToStart(body);
  Value result = TileOp::create(rewriter, loc, tileType, body->getArgument(0),
                                body->getArgument(1));
  emitTileStores(rewriter, loc, result, op);
  YieldOp::create(rewriter, loc);
  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

/// Build a ttl.compute op with a single unary tile operation in the body.
/// Input must already be attached to a CB via ttl.attach_cb.
/// Output CBs are the reserved CBs to which the op's result is stored.
template <typename TileOp>
static LogicalResult buildUnaryCompute(Operation *op, PatternRewriter &rewriter,
                                       Value input) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  // Try direct CB attachment first
  Value inputCb = getAttachedCB(input);

  // If input isn't CB-attached, try fusion
  if (!inputCb) {
    auto traceResult = traceElementwiseToRoots(op->getResult(0));
    if (traceResult.failureReason == TraceFailureReason::Success &&
        !traceResult.opsInOrder.empty()) {
      return buildFusedCompute(op, rewriter, traceResult);
    }
    emitFusionFailureDiagnostics(op, traceResult);
    return failure();
  }

  // Find all output CBs via stores on this op's result.
  SmallVector<Value> outCbs = collectOutputCBs(op);
  if (outCbs.empty()) {
    return rewriter.notifyMatchFailure(
        op, "no output CB found (missing ttl.store, view not from "
            "ttl.cb_reserve, or intermediate value handled by fusion)");
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: 1 input + N outputs.
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  SmallVector<Attribute> maps(1, AffineMapAttr::get(identityMap));
  for (size_t i = 0; i < outCbs.size(); ++i) {
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel for now
  SmallVector<Attribute> iterTypes(type.getRank(),
                                   rewriter.getStringAttr("parallel"));

  // Position compute after all reserves by inserting before the last store.
  insertAtLastStore(rewriter, op);

  // Create init tensors and attach to output CBs.
  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outCb : outCbs) {
    Value init = buildInitTensor(rewriter, loc, type, input);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outCb);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(type);
  }

  // Input is already attached, use it directly.
  // Create ttl.compute op
  auto computeOp = ComputeOp::create(
      rewriter, loc, TypeRange(resultTypes), ValueRange{input},
      ValueRange(allInitAttached), rewriter.getArrayAttr(maps),
      rewriter.getArrayAttr(iterTypes));

  // Build the body region with tile type block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type scalarType = type.getElementType();
  // Create tile type: !ttcore.tile<32x32, dtype>
  Type tileType = ttcore::TileType::get(scalarType);
  body->addArgument(tileType, loc); // input tile
  for (size_t i = 0; i < outCbs.size(); ++i) {
    body->addArgument(tileType, loc); // output tile
  }

  rewriter.setInsertionPointToStart(body);
  Value result = TileOp::create(rewriter, loc, tileType, body->getArgument(0));
  emitTileStores(rewriter, loc, result, op);
  YieldOp::create(rewriter, loc);
  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

namespace {
//===----------------------------------------------------------------------===//
// Templated Elementwise Lowering Patterns
//===----------------------------------------------------------------------===//

/// Pattern for binary elementwise ops: TTL tensor op -> ttl.compute with tile
/// op.
template <typename TTLOp, typename TileOp>
struct LowerBinaryToCompute : OpRewritePattern<TTLOp> {
  using OpRewritePattern<TTLOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildBinaryCompute<TileOp>(op.getOperation(), rewriter, op.getLhs(),
                                      op.getRhs());
  }
};

/// Pattern for unary elementwise ops: TTL tensor op -> ttl.compute with tile
/// op.
template <typename TTLOp, typename TileOp>
struct LowerUnaryToCompute : OpRewritePattern<TTLOp> {
  using OpRewritePattern<TTLOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryCompute<TileOp>(op.getOperation(), rewriter,
                                     op.getInput());
  }
};

//===----------------------------------------------------------------------===//
// Bcast Lowering Pattern
//===----------------------------------------------------------------------===//

/// Build affine map for bcast shape expansion.
/// For col bcast (N,1) -> (N,M): returns (i,j) -> (i,0)
/// For row bcast (1,M) -> (N,M): returns (i,j) -> (0,j)
/// For scalar bcast (1,1) -> (N,M): returns (i,j) -> (0,0)
/// For no expansion: returns identity map.
static AffineMap buildBcastInputMap(MLIRContext *ctx, bool expandRows,
                                    bool expandCols) {
  if (expandRows && expandCols) {
    return AffineMap::get(
        2, 0, {getAffineConstantExpr(0, ctx), getAffineConstantExpr(0, ctx)},
        ctx);
  }
  if (expandCols) {
    return AffineMap::get(
        2, 0, {getAffineDimExpr(0, ctx), getAffineConstantExpr(0, ctx)}, ctx);
  }
  if (expandRows) {
    return AffineMap::get(
        2, 0, {getAffineConstantExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
  }
  return AffineMap::getMultiDimIdentityMap(2, ctx);
}

/// Validate that shape expansion is compatible with bcast type.
static LogicalResult validateBcastExpansion(BcastOp op, bool expandRows,
                                            bool expandCols) {
  auto bcastType = op.getBcastType();
  if (expandRows && expandCols) {
    if (bcastType != BcastType::Scalar) {
      return op.emitError("row+col expansion requires scalar bcast type");
    }
  } else if (expandCols) {
    if (bcastType != BcastType::Col) {
      return op.emitError("col expansion requires col bcast type");
    }
  } else if (expandRows) {
    if (bcastType != BcastType::Row) {
      return op.emitError("row expansion requires row bcast type");
    }
  }
  return success();
}

/// Pattern for bcast op: TTL tensor op -> ttl.compute with tile_bcast.
/// Supports shape expansion where input CB can be smaller than output CB.
struct LowerBcastToCompute : OpRewritePattern<BcastOp> {
  using OpRewritePattern<BcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BcastOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = getTensorType(op.getResult());
    auto inputType = getTensorType(op.getInput());
    if (!outputType || !inputType) {
      return failure();
    }

    Value inputCb = getAttachedCB(op.getInput());
    Value outCb = getAttachedCB(op.getOutput());
    if (!inputCb) {
      return op.emitError(
          "broadcast input must come directly from a circular buffer, not from "
          "an elementwise result; move the broadcast to its own compute block "
          "or make it the first operation in a fused sequence");
    }
    if (!outCb) {
      return op.emitError("bcast output must be attached to a circular buffer");
    }

    if (inputType.getRank() != 2 || outputType.getRank() != 2) {
      return op.emitError("bcast requires rank-2 tensors");
    }

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    bool expandRows = inputShape[0] != outputShape[0];
    bool expandCols = inputShape[1] != outputShape[1];

    if (expandRows && inputShape[0] != 1) {
      return op.emitError("row expansion requires input dim 0 to be 1");
    }
    if (expandCols && inputShape[1] != 1) {
      return op.emitError("col expansion requires input dim 1 to be 1");
    }

    if (failed(validateBcastExpansion(op, expandRows, expandCols))) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(2, ctx);
    AffineMap inputMap = buildBcastInputMap(ctx, expandRows, expandCols);

    SmallVector<Attribute> maps = {AffineMapAttr::get(inputMap),
                                   AffineMapAttr::get(outputMap),
                                   AffineMapAttr::get(outputMap)};

    SmallVector<Attribute> iterTypes(outputType.getRank(),
                                     rewriter.getStringAttr("parallel"));

    // Position compute after all reserves by inserting before the last store.
    if (findLastStore(op)) {
      insertAtLastStore(rewriter, op);
    }

    Value init = buildInitTensor(rewriter, loc, outputType, op.getOutput());
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outCb);

    auto computeOp = ComputeOp::create(
        rewriter, loc, TypeRange{outputType},
        ValueRange{op.getInput(), op.getOutput()}, ValueRange{initAttached},
        rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = outputType.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc);
    body->addArgument(tileType, loc);
    body->addArgument(tileType, loc);

    rewriter.setInsertionPointToStart(body);
    Value result =
        TileBcastOp::create(rewriter, loc, tileType, body->getArgument(0),
                            body->getArgument(1), op.getBcastType());
    emitTileStores(rewriter, loc, result, op.getOperation());
    YieldOp::create(rewriter, loc);
    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Lowering
//===----------------------------------------------------------------------===//

/// Lowers passthrough ttl.store (CB-attached input) by creating a compute
/// with tile_store. Stores whose input comes from an elementwise op are
/// already erased by the elementwise builders (emitTileStores).
struct LowerStoreToCompute : OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getTensor();
    Value reserveView = op.getView();
    auto reserve = reserveView.getDefiningOp<CBReserveOp>();
    if (!reserve) {
      return rewriter.notifyMatchFailure(op, "view not from ttl.cb_reserve");
    }
    Value outputCb = reserve.getCb();

    // Passthrough: input is CB-attached, create a new compute with tile_store.
    if (!getAttachedCB(input)) {
      return rewriter.notifyMatchFailure(
          op, "store input must be CB-attached (elementwise stores are "
              "handled by their respective builders)");
    }

    auto inputType = getTensorType(input);
    if (!inputType) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), ctx);
    SmallVector<Attribute> maps = {AffineMapAttr::get(identityMap),
                                   AffineMapAttr::get(identityMap)};
    SmallVector<Attribute> iterTypes(inputType.getRank(),
                                     rewriter.getStringAttr("parallel"));

    Value init = buildInitTensor(rewriter, loc, inputType, input);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outputCb);

    auto computeOp = ComputeOp::create(
        rewriter, loc, TypeRange{inputType}, ValueRange{input},
        ValueRange{initAttached}, rewriter.getArrayAttr(maps),
        rewriter.getArrayAttr(iterTypes));

    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = inputType.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc);
    body->addArgument(tileType, loc);

    rewriter.setInsertionPointToEnd(body);
    SmallVector<Value> iterIndices =
        getOrCreateIterIndices(rewriter, computeOp);
    SmallVector<Value> storeIndices =
        applyIndexingMapToIterIndices(rewriter, loc, identityMap, iterIndices);
    TileStoreOp::create(rewriter, loc, body->getArgument(0), reserveView,
                        storeIndices);
    YieldOp::create(rewriter, loc);

    // make_early_inc_range: replaceOp erases attachOp, invalidating the
    // use-list iterator.
    for (OpOperand &use : llvm::make_early_inc_range(input.getUses())) {
      if (auto attachOp = dyn_cast<AttachCBOp>(use.getOwner())) {
        if (attachOp.getCb() == outputCb) {
          rewriter.replaceOp(attachOp, computeOp.getResult(0));
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Type Aliases - Generated from TTLElementwiseOps.def (tile-based)
//===----------------------------------------------------------------------===//

// Generate type aliases for binary operations using tile ops
// (TTK_INIT and TTK_COMPUTE are unused here, only needed for TTKernel lowering)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
// Generate type aliases for unary operations using tile ops
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  using Lower##TTL_OP = LowerUnaryToCompute<TTL_OP##Op, TILE_OP>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// Pass Implementations
//===----------------------------------------------------------------------===//

struct TTLConvertTTLToComputePass
    : public tt::ttl::impl::TTLConvertTTLToComputeBase<
          TTLConvertTTLToComputePass> {
  using tt::ttl::impl::TTLConvertTTLToComputeBase<
      TTLConvertTTLToComputePass>::TTLConvertTTLToComputeBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    populateTTLToComputePatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void populateTTLToComputePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Register patterns for lowering to ttl.compute with tile ops.
  // These are generated from TTLElementwiseOps.def using tile-based mappings.
  // (TTK_INIT and TTK_COMPUTE are unused here, only needed for TTKernel
  // lowering)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  patterns.add<Lower##TTL_OP>(ctx);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  patterns.add<LowerBcastToCompute>(ctx);
  patterns.add<LowerStoreToCompute>(ctx);
}

} // namespace mlir::tt::ttl
