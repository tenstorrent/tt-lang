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

  // Inputs must already be attached to CBs.
  Value lhsCb = getAttachedCB(lhs);
  Value rhsCb = getAttachedCB(rhs);
  if (!lhsCb || !rhsCb) {
    return op->emitError(
        "inputs must be attached to circular buffers via "
        "`ttl.attach_cb` or `ttl.cb_wait` before lowering to `ttl.compute`");
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

  // Input must already be attached to a CB.
  Value inputCb = getAttachedCB(input);
  if (!inputCb) {
    return op->emitError(
        "input must be attached to a circular buffer via "
        "`ttl.attach_cb` or `ttl.cb_wait` before lowering to `ttl.compute`");
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
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP)                                    \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
// Generate type aliases for unary operations using tile ops
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP)                                     \
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
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP) patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP) patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Matmul pattern (not generated from .def file).
  patterns.add<LowerMatmulToCompute>(ctx);
}

} // namespace mlir::tt::ttl
