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
static Value emitTileOpFor(OpBuilder &b, Location loc, Operation *tensorOp,
                           ValueRange tileOperands, Type tileType) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##Op>(tensorOp))                                               \
    return b.create<TILE_OP>(loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
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
    for (Value operand : getElementwiseOperands(op)) {
      auto it = tensorToTile.find(operand);
      if (it == tensorToTile.end()) {
        return op->emitError("fusion failed: operand not mapped to tile value");
      }
      tileOperands.push_back(it->second);
    }

    Value tileResult = emitTileOpFor(rewriter, loc, op, tileOperands, tileType);
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

/// Build a ttl.compute op with tile_matmul in the body.
/// Matmul is different from binary ops: it reads a,b from CBs directly (not DST)
/// and accumulates into DST. The c operand is the accumulator/output.
static LogicalResult buildMatmulCompute(Operation *op, PatternRewriter &rewriter,
                                        Value a, Value b, Value c) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  // All inputs must be attached to CBs.
  Value aCb = getAttachedCB(a);
  Value bCb = getAttachedCB(b);
  Value cCb = getAttachedCB(c);
  if (!aCb || !bCb) {
    return op->emitError(
        "matmul inputs a and b must be attached to circular buffers via "
        "`ttl.attach_cb` or `ttl.cb_wait` before lowering to `ttl.compute`");
  }

  // Find the output CB. First check c's CB, then look for attach_cb on result,
  // finally fall back to unused bind_cb.
  Value outCb = cCb;
  if (!outCb) {
    outCb = findOutputCB(op);
  }
  if (!outCb) {
    auto unusedCBs = findUnusedBindCBs(op);
    if (unusedCBs.empty()) {
      return op->emitError("no circular buffer found for matmul output; ensure "
                           "c operand is attached to a CB or a ttl.bind_cb "
                           "exists for output");
    }
    outCb = unusedCBs.front().getResult();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps for all inputs and output.
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  maps.push_back(AffineMapAttr::get(identityMap)); // a
  maps.push_back(AffineMapAttr::get(identityMap)); // b
  maps.push_back(AffineMapAttr::get(identityMap)); // c (output)

  // Build iterator types: all parallel.
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  // Create init tensor and attach to output CB.
  Value init = buildInitTensor(rewriter, loc, type, a);
  Value initAttached =
      rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

  // Create ttl.compute op with a, b as inputs and c (init) as output.
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{a, b}, ValueRange{initAttached},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region with tile type block arguments.
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type scalarType = type.getElementType();
  Type tileType = ttcore::TileType::get(scalarType);
  body->addArgument(tileType, loc); // a tile
  body->addArgument(tileType, loc); // b tile
  body->addArgument(tileType, loc); // c tile (accumulator)

  rewriter.setInsertionPointToStart(body);
  // tile_matmul(a, b, c) -> result (same type as c, accumulated)
  Value result = rewriter.create<TileMatmulOp>(
      loc, tileType, body->getArgument(0), body->getArgument(1),
      body->getArgument(2));
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
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
// Generate type aliases for unary operations using tile ops
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  using Lower##TTL_OP = LowerUnaryToCompute<TTL_OP##Op, TILE_OP>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

/// Pattern for matmul: TTL matmul op -> ttl.compute with tile_matmul.
struct LowerMatmulToCompute : OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    return buildMatmulCompute(op.getOperation(), rewriter, op.getA(), op.getB(),
                              op.getC());
  }
};

/// Pattern for power: TTL power op -> ttl.compute with tile_power.
/// Power is a unary op with an additional scalar exponent attribute.
struct LowerPowerToCompute : OpRewritePattern<PowerOp> {
  using OpRewritePattern<PowerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PowerOp op,
                                PatternRewriter &rewriter) const override {
    auto type = getTensorType(op.getResult());
    if (!type) {
      return failure();
    }

    Value input = op.getInput();
    Value inputCb = getAttachedCB(input);
    if (!inputCb) {
      return op.emitError("power input must be attached to a circular buffer");
    }

    Value outCb = findOutputCB(op);
    if (!outCb) {
      auto unusedCBs = findUnusedBindCBs(op);
      if (unusedCBs.empty()) {
        return op.emitError("no unused bind_cb found for output");
      }
      outCb = unusedCBs.front().getResult();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Build identity indexing maps
    SmallVector<Attribute> maps;
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
    maps.push_back(AffineMapAttr::get(identityMap)); // input
    maps.push_back(AffineMapAttr::get(identityMap)); // output

    // Build iterator types: all parallel
    SmallVector<Attribute> iterTypes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      iterTypes.push_back(rewriter.getStringAttr("parallel"));
    }

    // Create init tensor and attach to output CB
    Value init = buildInitTensor(rewriter, loc, type, input);
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

    // Create ttl.compute op
    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{type}, ValueRange{input}, ValueRange{initAttached},
        rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

    // Build the body region with tile type block arguments
    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = type.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc); // input tile
    body->addArgument(tileType, loc); // output tile

    rewriter.setInsertionPointToStart(body);
    // Create tile_power with the exponent attribute
    Value result = rewriter.create<TilePowerOp>(loc, tileType,
                                                body->getArgument(0),
                                                op.getExponentAttr());
    rewriter.create<YieldOp>(loc, ValueRange{result});

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

/// Pattern for where: TTL where op -> ttl.compute with tile_where.
/// Where is a ternary op: result = condition ? true_val : false_val
struct LowerWhereToCompute : OpRewritePattern<WhereOp> {
  using OpRewritePattern<WhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    auto type = getTensorType(op.getResult());
    if (!type) {
      return failure();
    }

    Value cond = op.getCondition();
    Value trueVal = op.getTrueVal();
    Value falseVal = op.getFalseVal();

    // All inputs must be attached to CBs
    Value condCb = getAttachedCB(cond);
    Value trueCb = getAttachedCB(trueVal);
    Value falseCb = getAttachedCB(falseVal);
    if (!condCb || !trueCb || !falseCb) {
      return op.emitError(
          "where inputs must be attached to circular buffers");
    }

    Value outCb = findOutputCB(op);
    if (!outCb) {
      auto unusedCBs = findUnusedBindCBs(op);
      if (unusedCBs.empty()) {
        return op.emitError("no unused bind_cb found for output");
      }
      outCb = unusedCBs.front().getResult();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Build identity indexing maps for all inputs and output
    SmallVector<Attribute> maps;
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
    maps.push_back(AffineMapAttr::get(identityMap)); // condition
    maps.push_back(AffineMapAttr::get(identityMap)); // true_val
    maps.push_back(AffineMapAttr::get(identityMap)); // false_val
    maps.push_back(AffineMapAttr::get(identityMap)); // output

    // Build iterator types: all parallel
    SmallVector<Attribute> iterTypes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      iterTypes.push_back(rewriter.getStringAttr("parallel"));
    }

    // Create init tensor and attach to output CB
    Value init = buildInitTensor(rewriter, loc, type, cond);
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

    // Create ttl.compute op with 3 inputs
    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{type}, ValueRange{cond, trueVal, falseVal},
        ValueRange{initAttached}, rewriter.getArrayAttr(maps),
        rewriter.getArrayAttr(iterTypes));

    // Build the body region with tile type block arguments
    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = type.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc); // condition tile
    body->addArgument(tileType, loc); // true_val tile
    body->addArgument(tileType, loc); // false_val tile
    body->addArgument(tileType, loc); // output tile

    rewriter.setInsertionPointToStart(body);
    // Create tile_where with the three input tiles
    Value result = rewriter.create<TileWhereOp>(
        loc, tileType, body->getArgument(0), body->getArgument(1),
        body->getArgument(2));
    rewriter.create<YieldOp>(loc, ValueRange{result});

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

/// Generic pattern for lowering TTL reduce ops to ttl.compute with tile reduce.
/// Reduce ops read from CB (like matmul), require reduce_dim attribute.
template <typename SourceOp, typename TileOp>
struct LowerReduceToCompute : OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    auto type = getTensorType(op.getResult());
    if (!type) {
      return failure();
    }

    Value input = op.getInput();
    Value inputCb = getAttachedCB(input);
    if (!inputCb) {
      return op.emitError("reduce input must be attached to a circular buffer");
    }

    // Get scaler CB from the explicit scaler parameter
    Value scaler = op.getScaler();
    Value scalerCb = getAttachedCB(scaler);
    if (!scalerCb) {
      return op.emitError("reduce scaler must be attached to a circular buffer");
    }

    // Get output CB from the explicit output parameter (like transpose)
    Value output = op.getOutput();
    Value outCb = getAttachedCB(output);
    if (!outCb) {
      return op.emitError("reduce output must be attached to a circular buffer");
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Build identity indexing maps (one for each input + one for output)
    SmallVector<Attribute> maps;
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
    maps.push_back(AffineMapAttr::get(identityMap)); // input
    maps.push_back(AffineMapAttr::get(identityMap)); // scaler CB
    maps.push_back(AffineMapAttr::get(identityMap)); // output CB
    maps.push_back(AffineMapAttr::get(identityMap)); // init output

    // Build iterator types: all parallel
    SmallVector<Attribute> iterTypes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      iterTypes.push_back(rewriter.getStringAttr("parallel"));
    }

    // Create init tensor and attach to output CB
    Value init = buildInitTensor(rewriter, loc, type, input);
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

    // Create ttl.compute op with input, scaler, and output in inputs array
    // This follows transpose/matmul pattern
    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{type}, ValueRange{input, scaler, output},
        ValueRange{initAttached}, rewriter.getArrayAttr(maps),
        rewriter.getArrayAttr(iterTypes));

    // Build the body region with tile type block arguments
    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = type.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc); // input tile (arg 0)
    body->addArgument(tileType, loc); // scaler tile (arg 1)
    body->addArgument(tileType, loc); // output CB tile (arg 2)
    body->addArgument(tileType, loc); // init output tile (arg 3)

    rewriter.setInsertionPointToStart(body);
    // Create tile reduce op with input, scaler, output, and reduce_dim
    Value result = rewriter.create<TileOp>(loc, tileType, body->getArgument(0),
                                           body->getArgument(1),
                                           body->getArgument(2),
                                           op.getReduceDimAttr());
    rewriter.create<YieldOp>(loc, ValueRange{result});

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

// Generate type aliases for reduce op lowerings
#define TTL_REDUCE_TILE_OP(TTL_OP, TILE_OP, REDUCE_TYPE)                        \
  using Lower##TTL_OP = LowerReduceToCompute<TTL_OP##Op, TILE_OP>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

/// Pattern for transpose: TTL transpose op -> ttl.compute with tile_transpose.
/// Transpose reads from CB (like matmul/reduce), writes result to DST.
/// Follows the matmul pattern: both input and output go in inputs array.
struct LowerTransposeToCompute : OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto type = getTensorType(op.getResult());
    if (!type) {
      return failure();
    }

    Value input = op.getInput();
    Value inputCb = getAttachedCB(input);
    if (!inputCb) {
      return op.emitError("transpose input must be attached to a circular buffer");
    }

    // Get output CB from the explicit output parameter (like matmul's c)
    Value output = op.getOutput();
    Value outCb = getAttachedCB(output);
    if (!outCb) {
      return op.emitError("transpose output must be attached to a circular buffer");
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Build identity indexing maps (one for each input + one for output)
    SmallVector<Attribute> maps;
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
    maps.push_back(AffineMapAttr::get(identityMap)); // input
    maps.push_back(AffineMapAttr::get(identityMap)); // output CB (like matmul's b)
    maps.push_back(AffineMapAttr::get(identityMap)); // init output

    // Build iterator types: all parallel
    SmallVector<Attribute> iterTypes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      iterTypes.push_back(rewriter.getStringAttr("parallel"));
    }

    // Create init tensor and attach to output CB
    Value init = buildInitTensor(rewriter, loc, type, input);
    Value initAttached =
        rewriter.create<AttachCBOp>(loc, init.getType(), init, outCb);

    // Create ttl.compute op with both input and output in inputs array
    // This follows matmul pattern where a, b are inputs with their CBs
    auto computeOp = rewriter.create<ComputeOp>(
        loc, TypeRange{type}, ValueRange{input, output}, ValueRange{initAttached},
        rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

    // Build the body region with tile type block arguments
    Block *body = rewriter.createBlock(&computeOp.getBody());
    Type scalarType = type.getElementType();
    Type tileType = ttcore::TileType::get(scalarType);
    body->addArgument(tileType, loc); // input tile (arg 0)
    body->addArgument(tileType, loc); // output CB tile (arg 1) - for init lookup
    body->addArgument(tileType, loc); // init output tile (arg 2)

    rewriter.setInsertionPointToStart(body);
    // Create tile_transpose op with input and output
    Value result = rewriter.create<TileTransposeOp>(loc, tileType,
                                                    body->getArgument(0),
                                                    body->getArgument(1));
    rewriter.create<YieldOp>(loc, ValueRange{result});

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
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)     \
  patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Matmul pattern (not generated from .def file).
  patterns.add<LowerMatmulToCompute>(ctx);

  // Power pattern (unary with scalar exponent attribute).
  patterns.add<LowerPowerToCompute>(ctx);

  // Where pattern (ternary conditional selection).
  patterns.add<LowerWhereToCompute>(ctx);

  // Reduce patterns (generated from .def file).
#define TTL_REDUCE_TILE_OP(TTL_OP, TILE_OP, REDUCE_TYPE)                        \
  patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Transpose pattern (reads from CB, writes to DST).
  patterns.add<LowerTransposeToCompute>(ctx);
}

} // namespace mlir::tt::ttl
