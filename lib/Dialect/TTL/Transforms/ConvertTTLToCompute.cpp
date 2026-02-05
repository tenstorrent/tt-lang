// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

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
      dynDims.push_back(b.create<tensor::DimOp>(loc, exemplar, dim.index()));
    }
  }
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                   dynDims);
}

/// Find the CB that this operation's result will be stored to.
/// Looks for attach_cb or tensor_store ops that use this operation's result.
static Value findOutputCB(Operation *op) {
  if (op->getNumResults() == 0) {
    return nullptr;
  }
  Value opResult = op->getResult(0);
  for (OpOperand &use : opResult.getUses()) {
    if (auto attachOp = dyn_cast<AttachCBOp>(use.getOwner())) {
      return attachOp.getCb();
    }
    if (auto storeOp = dyn_cast<TensorStoreOp>(use.getOwner())) {
      return storeOp.getCb();
    }
  }
  return nullptr;
}

/// Find unused bind_cb ops in the function that can be used for output CBs.
/// Returns bind_cb ops that are not used by any attach_cb op.
// TODO: Use AnalysisManager to cache CB usage analysis and avoid re-walking
// the function for each operation.
static SmallVector<BindCBOp> findUnusedBindCBs(Operation *op) {
  SmallVector<BindCBOp> unused;
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    return unused;
  }

  // Collect all bind_cb ops and used CBs in a single walk.
  SmallVector<BindCBOp> allBindCBs;
  DenseSet<Value> usedCBs;
  parentFunc->walk([&](Operation *walkOp) {
    if (auto bindOp = dyn_cast<BindCBOp>(walkOp)) {
      allBindCBs.push_back(bindOp);
    } else if (auto attachOp = dyn_cast<AttachCBOp>(walkOp)) {
      usedCBs.insert(attachOp.getCb());
    }
  });

  // Return unused ones
  for (auto bindOp : allBindCBs) {
    if (!usedCBs.contains(bindOp.getResult())) {
      unused.push_back(bindOp);
    }
  }

  return unused;
}

//===----------------------------------------------------------------------===//
// Tile op emission for fusion
//===----------------------------------------------------------------------===//

/// Emit the tile-level op corresponding to a tensor-level elementwise op.
/// Returns the result Value, or null on failure.
static Value emitTileOpFor(OpBuilder &b, Location loc, Operation *tensorOp,
                           ValueRange tileOperands, Type tileType) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Fused compute building
//===----------------------------------------------------------------------===//

/// Build a fused ttl.compute from traced elementwise chain.
/// The trace result contains CB-attached root inputs and ops to fuse.
static LogicalResult buildFusedCompute(Operation *sinkOp,
                                       PatternRewriter &rewriter,
                                       const ElementwiseTraceResult &trace) {
  auto type = getTensorType(sinkOp->getResult(0));
  if (!type) {
    return failure();
  }

  // Find output CB
  Value outCb = findOutputCB(sinkOp);
  if (!outCb) {
    auto unusedCBs = findUnusedBindCBs(sinkOp);
    if (unusedCBs.empty()) {
      return sinkOp->emitError("no unused bind_cb found for output");
    }
    outCb = unusedCBs.front().getResult();
  }

  Location loc = sinkOp->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build indexing maps: identity for each input and output
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    maps.push_back(AffineMapAttr::get(identityMap));
  }
  maps.push_back(AffineMapAttr::get(identityMap)); // output

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  // Create init tensor and attach to output CB
  Value init = buildInitTensor(rewriter, loc, type, trace.rootInputs[0]);
  Value initAttached =
      rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, trace.rootInputs.getArrayRef(),
      ValueRange{initAttached}, rewriter.getArrayAttr(maps),
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
  body->addArgument(tileType, loc); // output tile

  rewriter.setInsertionPointToStart(body);

  // Map tensor values to tile values (for wiring up operands)
  DenseMap<Value, Value> tensorToTile;
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    tensorToTile[trace.rootInputs[i]] = body->getArgument(i);
  }

  // Emit tile ops in topological order
  Value finalResult;
  for (Operation *op : trace.opsInOrder) {
    Value tileResult;

    // Special case: BcastOp reads from CB, needs TileBcastOp
    if (auto bcastOp = dyn_cast<BcastOp>(op)) {
      Value inputTile = tensorToTile[bcastOp.getInput()];
      Value outputTile = body->getArguments().back(); // output block arg
      tileResult = rewriter.create<TileBcastOp>(
          loc, tileType, inputTile, outputTile, bcastOp.getBcastTypeAttr());
    } else {
      // Elementwise ops
      SmallVector<Value, 2> tileOperands;
      for (Value operand : getElementwiseOperands(op)) {
        auto it = tensorToTile.find(operand);
        if (it == tensorToTile.end()) {
          return op->emitError(
              "fusion failed: operand not mapped to tile value");
        }
        tileOperands.push_back(it->second);
      }

      tileResult = emitTileOpFor(rewriter, loc, op, tileOperands, tileType);
      if (!tileResult) {
        return op->emitError("fusion failed: unsupported op type");
      }
    }

    tensorToTile[op->getResult(0)] = tileResult;
    finalResult = tileResult;
  }

  rewriter.create<YieldOp>(loc, ValueRange{finalResult});
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

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering to ttl.compute with tile ops
//===----------------------------------------------------------------------===//

/// Build a ttl.compute op with a single binary tile operation in the body.
/// Inputs must already be attached to CBs via ttl.attach_cb.
/// An unused bind_cb must exist for the output.
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

  // Find the output CB. First check if there's an attach_cb or tensor_store
  // that uses this result. Otherwise, find an unused bind_cb.
  Value outCb = findOutputCB(op);
  if (!outCb) {
    auto unusedCBs = findUnusedBindCBs(op);
    if (unusedCBs.empty()) {
      return op->emitError("no unused bind_cb found for output; ensure a "
                           "ttl.bind_cb exists for the output tensor");
    }
    outCb = unusedCBs.front().getResult();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: (d0, d1, ...) -> (d0, d1, ...)
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  // inputs
  maps.push_back(AffineMapAttr::get(identityMap));
  maps.push_back(AffineMapAttr::get(identityMap));
  // outputs
  maps.push_back(AffineMapAttr::get(identityMap));

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  // Create init tensor and attach to output CB.
  Value init = buildInitTensor(rewriter, loc, type, lhs);
  Value initAttached =
      rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

  // Inputs are already attached, use them directly.
  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{lhs, rhs}, ValueRange{initAttached},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region with tile type block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type scalarType = type.getElementType();
  // Create tile type: !ttcore.tile<32x32, dtype>
  Type tileType = ttcore::TileType::get(scalarType);
  body->addArgument(tileType, loc); // lhs tile
  body->addArgument(tileType, loc); // rhs tile
  body->addArgument(tileType, loc); // output tile

  rewriter.setInsertionPointToStart(body);
  Value result = rewriter.create<TileOp>(loc, tileType, body->getArgument(0),
                                         body->getArgument(1));
  rewriter.create<YieldOp>(loc, ValueRange{result});

  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

/// Build a ttl.compute op with a single unary tile operation in the body.
/// Input must already be attached to a CB via ttl.attach_cb.
/// An unused bind_cb must exist for the output.
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

  // Find the output CB. First check if there's an attach_cb or tensor_store
  // that uses this result. Otherwise, find an unused bind_cb.
  Value outCb = findOutputCB(op);
  if (!outCb) {
    auto unusedCBs = findUnusedBindCBs(op);
    if (unusedCBs.empty()) {
      return op->emitError("no unused bind_cb found for output; ensure a "
                           "ttl.bind_cb exists for the output tensor");
    }
    outCb = unusedCBs.front().getResult();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: (d0, d1, ...) -> (d0, d1, ...)
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  // input
  maps.push_back(AffineMapAttr::get(identityMap));
  // output
  maps.push_back(AffineMapAttr::get(identityMap));

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  // Create init tensor and attach to output CB.
  Value init = buildInitTensor(rewriter, loc, type, input);
  Value initAttached =
      rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

  // Input is already attached, use it directly.
  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{input}, ValueRange{initAttached},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region with tile type block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type scalarType = type.getElementType();
  // Create tile type: !ttcore.tile<32x32, dtype>
  Type tileType = ttcore::TileType::get(scalarType);
  body->addArgument(tileType, loc); // input tile
  body->addArgument(tileType, loc); // output tile

  rewriter.setInsertionPointToStart(body);
  Value result = rewriter.create<TileOp>(loc, tileType, body->getArgument(0));
  rewriter.create<YieldOp>(loc, ValueRange{result});

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

    Value init = buildInitTensor(rewriter, loc, outputType, op.getOutput());
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{outputType}, ValueRange{op.getInput(), op.getOutput()},
        ValueRange{initAttached}, rewriter.getArrayAttr(maps),
        rewriter.getArrayAttr(iterTypes));

    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = outputType.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc);
    body->addArgument(tileType, loc);
    body->addArgument(tileType, loc);

    rewriter.setInsertionPointToStart(body);
    Value result =
        rewriter.create<TileBcastOp>(loc, tileType, body->getArgument(0),
                                     body->getArgument(1), op.getBcastType());
    rewriter.create<YieldOp>(loc, ValueRange{result});

    rewriter.replaceOp(op, computeOp.getResult(0));
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
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
// Generate type aliases for unary operations using tile ops
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  using Lower##TTL_OP = LowerUnaryToCompute<TTL_OP##Op, TILE_OP>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// TensorStore Lowering (passthrough case only)
//===----------------------------------------------------------------------===//

/// Pattern for tensor_store with CB-attached input (passthrough case).
/// Creates a ComputeOp that copies input tiles to output CB.
/// For elementwise ops, tensor_store's input is from ComputeOp (not
/// CB-attached), so this pattern won't match. Those tensor_stores are erased in
/// TTKernel lowering.
struct LowerTensorStoreToCompute : OpRewritePattern<TensorStoreOp> {
  using OpRewritePattern<TensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getTensor();
    Value outputCb = op.getCb();

    // Only handle passthrough case where input is CB-attached.
    // Elementwise case: input is from ComputeOp, not CB-attached - skip.
    if (!getAttachedCB(input)) {
      return failure();
    }

    auto inputType = getTensorType(input);
    if (!inputType) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Build identity maps for input and output
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), ctx);
    SmallVector<Attribute> maps = {AffineMapAttr::get(identityMap),
                                   AffineMapAttr::get(identityMap)};
    SmallVector<Attribute> iterTypes(inputType.getRank(),
                                     rewriter.getStringAttr("parallel"));

    // Create output init tensor attached to output CB
    Value init = buildInitTensor(rewriter, loc, inputType, input);
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outputCb);

    // Create compute op with passthrough body
    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{inputType}, ValueRange{input}, ValueRange{initAttached},
        rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

    // Build the body: just yield the input tile (passthrough)
    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = inputType.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc); // Input tile
    body->addArgument(tileType, loc); // Output tile (unused)

    rewriter.setInsertionPointToEnd(body);
    rewriter.create<YieldOp>(loc, body->getArgument(0));

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

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
#define TTL_BINARY_TILE_OP_SPECIAL(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  patterns.add<LowerBcastToCompute>(ctx);
  patterns.add<LowerTensorStoreToCompute>(ctx);
}

} // namespace mlir::tt::ttl
