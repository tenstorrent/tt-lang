// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Materialize Block Expressions Pass
//===----------------------------------------------------------------------===//
//
// For each ttl.store whose tensor operand is a block_expr DAG, traces backward
// to DFB-attached root inputs, then creates a ttl.compute at the store
// position with tile ops matching the expression DAG.
//
// Stores whose block_expr DAGs overlap (share any block_expr op) are grouped
// into a single compute op with multiple tile_stores.  Grouping uses
// EquivalenceClasses (union-find) keyed on shared block_expr ops.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-materialize-block-exprs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLMATERIALIZEBLOCKEXPRS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Check if an operation is a block_expr op (lazy expression node).
static bool isBlockExprOp(Operation *op) {
  return isa<AddBlockExprOp, SubBlockExprOp, MulBlockExprOp, DivBlockExprOp,
             MaxBlockExprOp, MinBlockExprOp, ExpBlockExprOp, LogBlockExprOp,
             SqrtBlockExprOp, RsqrtBlockExprOp, TanhBlockExprOp,
             SigmoidBlockExprOp, NegBlockExprOp, AbsBlockExprOp,
             ReluBlockExprOp, FloorBlockExprOp, RecipBlockExprOp,
             SinBlockExprOp, CosBlockExprOp, TanBlockExprOp, BlockExprMatmulOp,
             BlockExprBcastOp>(op);
}

/// Check if a block_expr op is unary (one operand).
static bool isBlockExprUnary(Operation *op) {
  return isa<ExpBlockExprOp, LogBlockExprOp, SqrtBlockExprOp, RsqrtBlockExprOp,
             TanhBlockExprOp, SigmoidBlockExprOp, NegBlockExprOp,
             AbsBlockExprOp, ReluBlockExprOp, FloorBlockExprOp,
             RecipBlockExprOp, SinBlockExprOp, CosBlockExprOp, TanBlockExprOp>(
      op);
}

/// Get the operands of a block_expr op that should be traced backward.
static SmallVector<Value, 2> getBlockExprOperands(Operation *op) {
  if (isBlockExprUnary(op)) {
    return {op->getOperand(0)};
  }
  if (isa<BlockExprBcastOp>(op)) {
    // Bcast: only trace the input, not the output (which provides shape).
    return {op->getOperand(0)};
  }
  // Binary ops (add, sub, mul, div, max, min, matmul): trace both operands.
  return {op->getOperand(0), op->getOperand(1)};
}

/// Trace backward from a value through block_expr ops to DFB-attached roots.
/// Collects all block_expr ops in topological order (roots first).
struct BlockExprTrace {
  llvm::SmallSetVector<Value, 4> rootInputs;
  llvm::SmallSetVector<Operation *, 8> opsInOrder;
};

static BlockExprTrace traceBlockExprToRoots(Value value) {
  BlockExprTrace result;

  std::function<bool(Value)> trace = [&](Value v) -> bool {
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isBlockExprOp(defOp)) {
      result.rootInputs.insert(v);
      return true;
    }

    for (Value operand : getBlockExprOperands(defOp)) {
      if (!trace(operand)) {
        return false;
      }
    }

    result.opsInOrder.insert(defOp);
    return true;
  };

  trace(value);
  return result;
}

/// Merge multiple per-store traces into a single trace.  Root inputs are
/// unioned.  opsInOrder is rebuilt by walking the block in order, which
/// produces a valid topological ordering because the Python frontend emits
/// ops in dependency order.
static BlockExprTrace
mergeTraces(const DenseMap<Operation *, BlockExprTrace> &storeTraces,
            SmallVectorImpl<StoreOp> &stores) {
  BlockExprTrace merged;
  DenseSet<Operation *> allOps;

  for (StoreOp store : stores) {
    auto it = storeTraces.find(store.getOperation());
    assert(it != storeTraces.end());
    const BlockExprTrace &trace = it->second;
    for (Value root : trace.rootInputs)
      merged.rootInputs.insert(root);
    for (Operation *op : trace.opsInOrder)
      allOps.insert(op);
  }

  // Walk block to collect ops in block order (valid topological order).
  Block *block = stores.front()->getBlock();
  for (auto &op : *block) {
    if (allOps.contains(&op))
      merged.opsInOrder.insert(&op);
  }

  return merged;
}

/// Check if an op is a user signpost or tile-level dprint that should be
/// pulled into the compute body alongside fused tile ops.
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

/// Collected signpost/dprint ops categorized by position relative to fused
/// block_expr ops, supporting multiple stores in the trailing region.
struct InterleavedSideEffects {
  SmallVector<Operation *> leadingOps;
  DenseMap<Operation *, SmallVector<Operation *>> opsBefore;
  /// Per-store trailing ops.  trailingPerStore[i] holds the signpost ops
  /// that surround stores[i] (begin signposts before it, end signposts
  /// after it).
  SmallVector<SmallVector<Operation *>> trailingPerStore;
  /// All collected side-effect ops, for erasure after materialization.
  SmallVector<Operation *> allCollected;

  /// Collect signpost/dprint ops interleaved with block_expr ops and the
  /// stores.
  static InterleavedSideEffects
  collect(const llvm::SmallSetVector<Operation *, 8> &opsInOrder,
          SmallVectorImpl<StoreOp> &stores) {
    InterleavedSideEffects result;
    if (opsInOrder.empty())
      return result;

    DenseSet<Operation *> fusedSet(opsInOrder.begin(), opsInOrder.end());

    // Find first and last block_expr ops in block order.
    Operation *firstFused = nullptr;
    Operation *lastFused = nullptr;
    for (auto &op : *stores.front()->getBlock()) {
      if (fusedSet.contains(&op)) {
        if (!firstFused)
          firstFused = &op;
        lastFused = &op;
      }
    }
    if (!firstFused)
      return result;

    // Leading: walk backwards from first fused op.
    SmallVector<Operation *> leading;
    for (auto *op = firstFused->getPrevNode(); op; op = op->getPrevNode()) {
      if (isSideEffectOpForCompute(op)) {
        leading.push_back(op);
      } else {
        break;
      }
    }
    for (auto it = leading.rbegin(); it != leading.rend(); ++it) {
      result.leadingOps.push_back(*it);
      result.allCollected.push_back(*it);
    }

    // Interleaved: walk from first to last fused op.
    Operation *prevFused = nullptr;
    for (auto *op = firstFused; op && op != lastFused->getNextNode();
         op = op->getNextNode()) {
      if (fusedSet.contains(op)) {
        prevFused = op;
      } else if (isSideEffectOpForCompute(op)) {
        result.allCollected.push_back(op);
        // Categorize: attach to the next fused op in topological order.
        Operation *lastOp = opsInOrder.back();
        if (prevFused == lastOp) {
          // Would be trailing, but we handle trailing per-store below.
          // This shouldn't happen since lastFused is the last in the
          // interleaved region. Fall through to opsBefore logic.
        }
        bool found = false;
        for (size_t idx = 0; idx < opsInOrder.size(); ++idx) {
          if (opsInOrder[idx] == prevFused && idx + 1 < opsInOrder.size()) {
            result.opsBefore[opsInOrder[idx + 1]].push_back(op);
            found = true;
            break;
          }
        }
        // Side-effect after last fused op in topological order: should not
        // appear in the interleaved region (between first..last fused in
        // block order).  Treat as trailing for safety.
        if (!found) {
          // Will be collected in the per-store trailing below.
        }
      }
    }

    // Trailing: partition side-effect ops around each store.
    DenseSet<Operation *> storeSet;
    for (StoreOp store : stores)
      storeSet.insert(store.getOperation());

    for (size_t si = 0; si < stores.size(); ++si) {
      SmallVector<Operation *> perStore;

      // Backward from store: collect begin signposts / dprints.
      // Stop at end signpost (belongs to previous store), another store,
      // lastFused, or a non-side-effect op.
      SmallVector<Operation *> before;
      for (auto *op = stores[si]->getPrevNode(); op; op = op->getPrevNode()) {
        if (storeSet.contains(op) || op == lastFused)
          break;
        if (isSideEffectOpForCompute(op)) {
          if (auto sp = dyn_cast<SignpostOp>(op); sp && sp.getIsEnd())
            break;
          before.push_back(op);
        } else {
          break;
        }
      }
      for (auto it = before.rbegin(); it != before.rend(); ++it)
        perStore.push_back(*it);

      // Forward from store: collect end signposts / dprints.
      // Stop at begin signpost (belongs to next store), another store,
      // cb_push/cb_pop, or a non-side-effect op.
      for (auto *op = stores[si]->getNextNode(); op; op = op->getNextNode()) {
        if (storeSet.contains(op))
          break;
        if (isa<CBPushOp>(op) || isa<CBPopOp>(op))
          break;
        if (isSideEffectOpForCompute(op)) {
          if (auto sp = dyn_cast<SignpostOp>(op); sp && !sp.getIsEnd())
            break;
          perStore.push_back(op);
        } else {
          break;
        }
      }

      for (auto *op : perStore)
        result.allCollected.push_back(op);
      result.trailingPerStore.push_back(std::move(perStore));
    }

    return result;
  }

  /// Clone a signpost or dprint op into the compute body.
  static void emitOne(Operation *op, OpBuilder &builder) {
    if (auto sp = dyn_cast<SignpostOp>(op)) {
      SignpostOp::create(builder, sp.getLoc(), sp.getNameAttr(),
                         sp.getIsEndAttr());
    } else {
      builder.clone(*op);
    }
  }

  void emitLeading(OpBuilder &builder) const {
    for (auto *op : leadingOps)
      emitOne(op, builder);
  }

  void emitBefore(Operation *traceOp, OpBuilder &builder) const {
    auto it = opsBefore.find(traceOp);
    if (it != opsBefore.end()) {
      for (auto *seOp : it->second)
        emitOne(seOp, builder);
    }
  }

  void emitTrailingBeforeStore(size_t storeIdx, OpBuilder &builder) const {
    if (storeIdx >= trailingPerStore.size())
      return;
    auto isEndSignpost = [](Operation *op) {
      auto sp = dyn_cast<SignpostOp>(op);
      return sp && sp.getIsEnd();
    };
    const auto &trailing = trailingPerStore[storeIdx];
    auto firstEndIt = llvm::find_if(trailing, isEndSignpost);
    for (auto it = trailing.begin(); it != firstEndIt; ++it)
      emitOne(*it, builder);
  }

  void emitTrailingAfterStore(size_t storeIdx, OpBuilder &builder) const {
    if (storeIdx >= trailingPerStore.size())
      return;
    auto isEndSignpost = [](Operation *op) {
      auto sp = dyn_cast<SignpostOp>(op);
      return sp && sp.getIsEnd();
    };
    const auto &trailing = trailingPerStore[storeIdx];
    auto firstEndIt = llvm::find_if(trailing, isEndSignpost);
    for (auto it = firstEndIt; it != trailing.end(); ++it)
      emitOne(*it, builder);
  }

  void eraseOriginals() {
    for (auto *op : allCollected)
      op->erase();
  }
};

/// Emit the tile-level op corresponding to a block_expr op.
/// Returns the result Value, or null on failure.
static Value emitTileOpForBlockExpr(OpBuilder &b, Location loc,
                                    Operation *blockExprOp,
                                    ValueRange tileOperands, Type tileType) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##BlockExprOp>(blockExprOp))                                   \
    return TILE_OP::create(b, loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##BlockExprOp>(blockExprOp))                                   \
    return TILE_OP::create(b, loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  return nullptr;
}

static RankedTensorType getTensorType(Value v) {
  return dyn_cast<RankedTensorType>(v.getType());
}

/// Materialize a group of stores that share block_expr ops into a single
/// ttl.compute with one set of tile ops and N tile_stores.
static LogicalResult materializeStoreGroup(SmallVectorImpl<StoreOp> &stores,
                                           const BlockExprTrace &trace,
                                           OpBuilder &builder) {
  assert(!stores.empty());
  Location loc = stores.back().getLoc();

  // All stores must have the same output type (same tile domain).
  auto outputType = getTensorType(stores.front().getView());
  if (!outputType)
    return stores.front().emitError("store view must have ranked tensor type");
  for (size_t i = 1; i < stores.size(); ++i) {
    StoreOp store = stores[i];
    auto storeType = getTensorType(store.getView());
    if (storeType != outputType) {
      return store.emitError(
          "multi-store group: output type mismatch across stores");
    }
  }

  if (trace.opsInOrder.empty())
    return success();

  // Validate bcast inputs.
  for (Operation *op : trace.opsInOrder) {
    if (auto bcastOp = dyn_cast<BlockExprBcastOp>(op)) {
      Value input = bcastOp.getInput();
      if (input.getDefiningOp() && isBlockExprOp(input.getDefiningOp())) {
        return bcastOp.emitError(
            "broadcast input must come directly from a circular buffer "
            "(DFB-attached value), not from an intermediate expression");
      }
    }
  }

  // Collect signpost/dprint ops before they get orphaned.
  auto sideEffects =
      InterleavedSideEffects::collect(trace.opsInOrder, stores);

  MLIRContext *ctx = builder.getContext();

  // Collect output CBs (one per store).
  SmallVector<Value> outCbs;
  for (StoreOp store : stores) {
    auto reserve = store.getView().getDefiningOp<CBReserveOp>();
    if (!reserve)
      return store.emitError("store view not from cb_reserve");
    outCbs.push_back(reserve.getCb());
  }

  // Build indexing maps: broadcast-aware for inputs, identity per output.
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(outputType.getRank(), ctx);

  for (Value root : trace.rootInputs) {
    auto inputType = getTensorType(root);
    if (inputType && inputType.getRank() == outputType.getRank()) {
      SmallVector<AffineExpr> exprs;
      bool hasBroadcast = false;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        if (inputType.getDimSize(d) == 1 && outputType.getDimSize(d) != 1) {
          exprs.push_back(getAffineConstantExpr(0, ctx));
          hasBroadcast = true;
        } else {
          exprs.push_back(getAffineDimExpr(d, ctx));
        }
      }
      if (hasBroadcast) {
        maps.push_back(AffineMapAttr::get(
            AffineMap::get(outputType.getRank(), 0, exprs, ctx)));
      } else {
        maps.push_back(AffineMapAttr::get(identityMap));
      }
    } else {
      maps.push_back(AffineMapAttr::get(identityMap));
    }
  }
  for (size_t i = 0; i < outCbs.size(); ++i)
    maps.push_back(AffineMapAttr::get(identityMap));

  SmallVector<Attribute> iterTypes(outputType.getRank(),
                                   builder.getStringAttr("parallel"));

  // Position compute before the last store so all reserves dominate.
  builder.setInsertionPoint(stores.back());

  // Create init tensors and attach to output CBs.
  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outCb : outCbs) {
    Value init = tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                         outputType.getElementType());
    Value initAttached =
        AttachCBOp::create(builder, loc, init.getType(), init, outCb);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(outputType);
  }

  auto computeOp = ComputeOp::create(
      builder, loc, TypeRange(resultTypes), trace.rootInputs.getArrayRef(),
      ValueRange(allInitAttached), builder.getArrayAttr(maps),
      builder.getArrayAttr(iterTypes));

  // Build the body region.
  Block *body = builder.createBlock(&computeOp.getBody());
  Type tileType = ttcore::TileType::get(outputType.getElementType());

  for (size_t i = 0; i < trace.rootInputs.size(); ++i)
    body->addArgument(tileType, loc);
  for (size_t i = 0; i < outCbs.size(); ++i)
    body->addArgument(tileType, loc);

  builder.setInsertionPointToStart(body);

  // Map tensor values to tile values.
  DenseMap<Value, Value> tensorToTile;
  for (size_t i = 0; i < trace.rootInputs.size(); ++i)
    tensorToTile[trace.rootInputs[i]] = body->getArgument(i);

  sideEffects.emitLeading(builder);

  // Matmul+add deferred emission map.
  DenseMap<Value, std::pair<Value, Value>> deferredMatmul;

  // Emit tile ops in topological order with interleaved side-effect ops.
  Value finalResult;
  for (Operation *op : trace.opsInOrder) {
    sideEffects.emitBefore(op, builder);
    Value tileResult;

    if (auto bcastOp = dyn_cast<BlockExprBcastOp>(op)) {
      Value inputTile = tensorToTile[bcastOp.getInput()];
      Value outputTile = body->getArguments().back();
      tileResult = TileBcastOp::create(builder, loc, tileType, inputTile,
                                       outputTile, bcastOp.getBcastTypeAttr());
    } else if (auto matmulOp = dyn_cast<BlockExprMatmulOp>(op)) {
      Value lhsTile = tensorToTile[matmulOp.getLhs()];
      Value rhsTile = tensorToTile[matmulOp.getRhs()];

      bool deferred = false;
      if (matmulOp.getResult().hasOneUse()) {
        Operation *user = *matmulOp.getResult().getUsers().begin();
        if (isa<AddBlockExprOp>(user) && trace.opsInOrder.contains(user)) {
          deferredMatmul[matmulOp.getResult()] = {lhsTile, rhsTile};
          deferred = true;
        }
      }
      if (!deferred) {
        tileResult = TileMatmulBlockOp::create(builder, loc, tileType, lhsTile,
                                               rhsTile, Value());
      }
    } else {
      if (isa<AddBlockExprOp>(op)) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        auto tryFold = [&](Value tensorA, Value tensorB) -> Value {
          auto dfIt = deferredMatmul.find(tensorA);
          if (dfIt == deferredMatmul.end())
            return nullptr;
          auto [mmLhs, mmRhs] = dfIt->second;
          Value accTile = tensorToTile.lookup(tensorB);
          if (!accTile)
            return nullptr;
          deferredMatmul.erase(dfIt);
          return TileMatmulBlockOp::create(builder, loc, tileType, mmLhs,
                                           mmRhs, accTile);
        };
        Value folded = tryFold(lhs, rhs);
        if (!folded)
          folded = tryFold(rhs, lhs);
        if (folded)
          tileResult = folded;
      }

      if (!tileResult) {
        for (Value operand : getBlockExprOperands(op)) {
          auto dfIt = deferredMatmul.find(operand);
          if (dfIt != deferredMatmul.end()) {
            auto [mmLhs, mmRhs] = dfIt->second;
            Value mmTile = TileMatmulBlockOp::create(builder, loc, tileType,
                                                     mmLhs, mmRhs, Value());
            tensorToTile[operand] = mmTile;
            deferredMatmul.erase(dfIt);
          }
        }
      }

      if (!tileResult) {
        SmallVector<Value, 2> tileOperands;
        for (Value operand : getBlockExprOperands(op)) {
          auto it = tensorToTile.find(operand);
          if (it == tensorToTile.end()) {
            return op->emitError("block_expr materialization: operand not "
                                 "mapped to tile value");
          }
          tileOperands.push_back(it->second);
        }

        tileResult =
            emitTileOpForBlockExpr(builder, loc, op, tileOperands, tileType);
        if (!tileResult)
          return op->emitError("block_expr materialization: unsupported op");
      }
    }

    if (tileResult) {
      tensorToTile[op->getResult(0)] = tileResult;
      finalResult = tileResult;
    }
  }

  // Emit tile_stores: one per store, with per-store signpost wrapping.
  SmallVector<Value> iterIndices = getOrCreateIterIndices(builder, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  size_t numInputs = trace.rootInputs.size();

  // Build CB -> output index mapping for multi-output disambiguation.
  DenseMap<Value, size_t> cbToOutputIdx;
  if (outCbs.size() > 1) {
    for (auto [idx, outCb] : llvm::enumerate(outCbs))
      cbToOutputIdx[outCb] = idx;
  }

  // Emit tile_stores in reverse order to match ConvertTTLToCompute convention
  // (last store first in pack order).
  for (int si = stores.size() - 1; si >= 0; --si) {
    sideEffects.emitTrailingBeforeStore(si, builder);

    // Resolve the tile value for this store.
    StoreOp store = stores[si];
    Value storeTensor = store.getTensor();
    Value tileVal = tensorToTile.lookup(storeTensor);
    if (!tileVal)
      tileVal = finalResult;

    // Determine output index for this store's CB.
    size_t outputIdx = 0;
    if (outCbs.size() > 1) {
      Value viewCB = getAttachedCB(store.getView());
      if (viewCB) {
        auto it = cbToOutputIdx.find(viewCB);
        if (it != cbToOutputIdx.end())
          outputIdx = it->second;
      }
    }
    AffineMap outputMap = indexingMaps[numInputs + outputIdx];
    SmallVector<Value> indices =
        applyIndexingMapToIterIndices(builder, loc, outputMap, iterIndices);

    TileStoreOp::create(builder, loc, tileVal, store.getView(), indices);

    sideEffects.emitTrailingAfterStore(si, builder);
  }

  YieldOp::create(builder, loc);

  // Erase stores.
  for (StoreOp store : stores)
    store.erase();

  // Erase block_expr ops in reverse topological order.
  for (auto it = trace.opsInOrder.rbegin(); it != trace.opsInOrder.rend();
       ++it) {
    if ((*it)->use_empty())
      (*it)->erase();
  }

  sideEffects.eraseOriginals();
  return success();
}

struct TTLMaterializeBlockExprsPass
    : public impl::TTLMaterializeBlockExprsBase<TTLMaterializeBlockExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Step 1: Trace each store's block_expr DAG.
    SmallVector<StoreOp> blockExprStores;
    DenseMap<Operation *, BlockExprTrace> storeTraces;

    funcOp.walk([&](StoreOp storeOp) {
      Value tensor = storeOp.getTensor();
      if (tensor.getDefiningOp() && isBlockExprOp(tensor.getDefiningOp())) {
        blockExprStores.push_back(storeOp);
        storeTraces[storeOp.getOperation()] = traceBlockExprToRoots(tensor);
      }
    });

    if (blockExprStores.empty())
      return;

    // Step 2: Union-find grouping.  Stores whose block_expr DAGs share
    // any block_expr op are placed in the same equivalence class.
    llvm::EquivalenceClasses<Operation *> storeGroups;
    DenseMap<Operation *, Operation *> blockExprToStore;

    for (StoreOp store : blockExprStores) {
      Operation *storePtr = store.getOperation();
      storeGroups.insert(storePtr);
      const BlockExprTrace &trace = storeTraces[storePtr];
      for (Operation *op : trace.opsInOrder) {
        auto it = blockExprToStore.find(op);
        if (it != blockExprToStore.end()) {
          storeGroups.unionSets(storePtr, it->second);
        } else {
          blockExprToStore[op] = storePtr;
        }
      }
    }

    // Step 3: Collect groups and materialize.
    // Build groups by leader: map leader -> list of stores.
    DenseMap<Operation *, SmallVector<StoreOp>> groupMap;
    for (StoreOp store : blockExprStores) {
      Operation *leader =
          storeGroups.getLeaderValue(store.getOperation());
      groupMap[leader].push_back(store);
    }

    llvm::errs() << "MaterializeBlockExprs: " << blockExprStores.size()
                  << " stores in " << groupMap.size() << " groups\n";
    for (auto &[leader, group] : groupMap)
      llvm::errs() << "  group size: " << group.size() << "\n";

    OpBuilder builder(&getContext());
    for (auto &[leader, groupStores] : groupMap) {
      // Sort by block position for deterministic emission order.
      DenseMap<Operation *, unsigned> blockPos;
      unsigned pos = 0;
      for (auto &op : *groupStores.front()->getBlock())
        blockPos[&op] = pos++;
      llvm::sort(groupStores, [&](StoreOp lhs, StoreOp rhs) {
        return blockPos[lhs.getOperation()] < blockPos[rhs.getOperation()];
      });

      // Merge traces across the group.
      BlockExprTrace merged = mergeTraces(storeTraces, groupStores);

      if (failed(materializeStoreGroup(groupStores, merged, builder)))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
