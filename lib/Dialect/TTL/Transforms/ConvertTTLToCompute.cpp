// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
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

/// Build an indexing map for a tensor that may be broadcast to the output
/// shape. For dimensions where input is 1 and output is >1, the map uses
/// constant 0. For example, input [1,1] to output [1,2] gives map (d0, d1) ->
/// (0, 0).
static AffineMap buildBroadcastIndexingMap(RankedTensorType inputType,
                                           RankedTensorType outputType,
                                           MLIRContext *ctx) {
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int64_t rank = outputType.getRank();

  // Build the affine expressions for each dimension
  SmallVector<AffineExpr> exprs;
  for (int64_t i = 0; i < rank; ++i) {
    if (i < static_cast<int64_t>(inputShape.size()) && inputShape[i] == 1 &&
        outputShape[i] > 1) {
      // Broadcast dimension: always access index 0
      exprs.push_back(getAffineConstantExpr(0, ctx));
    } else {
      // Regular dimension: identity mapping
      exprs.push_back(getAffineDimExpr(i, ctx));
    }
  }

  return AffineMap::get(rank, 0, exprs, ctx);
}

/// Determine broadcast dimension for an input relative to the output.
/// Returns nullopt if no broadcast (shapes match).
/// Assumes 2D tile tensors: shape is [rows, cols] of tiles.
static std::optional<BcastDim> getBcastDim(RankedTensorType inputType,
                                           RankedTensorType outputType) {
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  if (inputShape == outputShape) {
    return std::nullopt; // No broadcast
  }

  // For 2D: [rows, cols]
  // row broadcast: input is [N, 1], output is [N, M] where M > 1
  // col broadcast: input is [1, M], output is [N, M] where N > 1
  // scalar broadcast: input is [1, 1], output is [N, M]
  if (inputType.getRank() < 2 || outputType.getRank() < 2) {
    return std::nullopt; // Can't determine for non-2D
  }

  int64_t inputRows = inputShape[inputShape.size() - 2];
  int64_t inputCols = inputShape[inputShape.size() - 1];
  int64_t outputRows = outputShape[outputShape.size() - 2];
  int64_t outputCols = outputShape[outputShape.size() - 1];

  // Determine which dimensions need broadcast based on what differs
  bool rowBcast = (inputRows == 1 && outputRows > 1);
  bool colBcast = (inputCols == 1 && outputCols > 1);

  if (rowBcast && colBcast) {
    // Both dimensions broadcast - scalar broadcast
    return BcastDim::scalar;
  }
  if (colBcast) {
    // Input is [N, 1], output is [N, M] - replicate single column across all
    // columns
    return BcastDim::col;
  }
  if (rowBcast) {
    // Input is [1, M], output is [N, M] - replicate single row across all rows
    return BcastDim::row;
  }

  return std::nullopt;
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

/// Find the CB that this operation's result will be attached to.
/// Looks for an attach_cb op that uses this operation's result.
static Value findOutputCB(Operation *op) {
  if (op->getNumResults() == 0) {
    return nullptr;
  }
  Value result = op->getResult(0);
  for (OpOperand &use : result.getUses()) {
    if (auto attachOp = dyn_cast<AttachCBOp>(use.getOwner())) {
      return attachOp.getCb();
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
/// The optional bcastDim specifies broadcast dimension when rhs is broadcast.
static Value emitTileOpFor(OpBuilder &b, Location loc, Operation *tensorOp,
                           ValueRange tileOperands, Type tileType,
                           BcastDimAttr bcastDim = nullptr) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0], tileOperands[1],  \
                             bcastDim);
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

  // Build indexing maps for inputs (broadcast-aware) and output (identity)
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  auto outputShape = type.getShape();
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    auto inputType = getTensorType(trace.rootInputs[i]);
    if (!inputType) {
      maps.push_back(AffineMapAttr::get(identityMap));
      continue;
    }
    auto inputShape = inputType.getShape();
    bool needsBroadcast = false;
    if (inputShape.size() == outputShape.size()) {
      for (size_t d = 0; d < inputShape.size(); ++d) {
        if (inputShape[d] != outputShape[d]) {
          needsBroadcast = true;
          break;
        }
      }
    }
    if (needsBroadcast) {
      // Input has different shape - build broadcast-aware indexing map
      maps.push_back(
          AffineMapAttr::get(buildBroadcastIndexingMap(inputType, type, ctx)));
    } else {
      // Same shape - identity map
      maps.push_back(AffineMapAttr::get(identityMap));
    }
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
    SmallVector<Value, 2> tileOperands;
    SmallVector<Value, 2> tensorOperands;
    for (Value operand : getElementwiseOperands(op)) {
      auto it = tensorToTile.find(operand);
      if (it == tensorToTile.end()) {
        return op->emitError("fusion failed: operand not mapped to tile value");
      }
      tileOperands.push_back(it->second);
      tensorOperands.push_back(operand);
    }

    // Detect broadcast for binary ops: check if rhs has smaller shape
    BcastDimAttr bcastDim = nullptr;
    if (tensorOperands.size() == 2) {
      auto resultType = getTensorType(op->getResult(0));
      auto rhsType = getTensorType(tensorOperands[1]);
      if (resultType && rhsType) {
        auto bcast = getBcastDim(rhsType, resultType);
        if (bcast.has_value()) {
          bcastDim = BcastDimAttr::get(rewriter.getContext(), *bcast);
        }
      }
    }

    Value tileResult =
        emitTileOpFor(rewriter, loc, op, tileOperands, tileType, bcastDim);
    if (!tileResult) {
      return op->emitError("fusion failed: unsupported op type");
    }

    tensorToTile[op->getResult(0)] = tileResult;
    finalResult = tileResult;
  }

  rewriter.create<YieldOp>(loc, ValueRange{finalResult});
  rewriter.replaceOp(sinkOp, computeOp.getResult(0));

  // Erase the fused ops (they're now inside the compute body as tile ops)
  for (Operation *op : trace.opsInOrder) {
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

  // Find the output CB. First check if there's an attach_cb that uses this
  // result, and use that CB. Otherwise, find an unused bind_cb.
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

  // Get rhs type for broadcast detection
  auto rhsType = getTensorType(rhs);
  auto outputShape = type.getShape();

  // Build indexing maps: identity for lhs, broadcast-aware for rhs if needed
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);

  // lhs always uses identity (we ensure lhs has the output shape in the DSL)
  maps.push_back(AffineMapAttr::get(identityMap));

  // rhs may need broadcast-aware indexing
  BcastDimAttr bcastDim = nullptr;
  if (rhsType) {
    auto rhsShape = rhsType.getShape();
    bool needsBroadcast = false;
    if (rhsShape.size() == outputShape.size()) {
      for (size_t d = 0; d < rhsShape.size(); ++d) {
        if (rhsShape[d] != outputShape[d]) {
          needsBroadcast = true;
          break;
        }
      }
    }
    if (needsBroadcast) {
      maps.push_back(
          AffineMapAttr::get(buildBroadcastIndexingMap(rhsType, type, ctx)));
      // Also set bcast_dim attribute for the tile op
      auto bcast = getBcastDim(rhsType, type);
      if (bcast.has_value()) {
        bcastDim = BcastDimAttr::get(ctx, *bcast);
      }
    } else {
      maps.push_back(AffineMapAttr::get(identityMap));
    }
  } else {
    maps.push_back(AffineMapAttr::get(identityMap));
  }
  // output
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
  // Create tile op with broadcast attribute if detected
  Value result = rewriter.create<TileOp>(loc, tileType, body->getArgument(0),
                                         body->getArgument(1), bcastDim);
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

  // Find the output CB. First check if there's an attach_cb that uses this
  // result, and use that CB. Otherwise, find an unused bind_cb.
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
}

} // namespace mlir::tt::ttl
