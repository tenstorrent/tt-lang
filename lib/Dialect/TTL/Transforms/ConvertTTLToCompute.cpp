// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLCONVERTTTLTOCOMPUTE
#define GEN_PASS_DEF_TTLASSIGNDSTREGISTERS
#include "ttlang/Dialect/TTL/Passes.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttl {
namespace {

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

//===----------------------------------------------------------------------===//
// Lowering to ttl.compute with tile ops (primary path)
//===----------------------------------------------------------------------===//

/// Build a ttl.compute op with a single binary tile operation in the body.
template <typename TileOp>
static LogicalResult buildBinaryCompute(Operation *op,
                                        PatternRewriter &rewriter, Value lhs,
                                        Value rhs) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: (d0, d1, ...) -> (d0, d1, ...)
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  for (int i = 0; i < 3; ++i) { // lhs, rhs, output
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  Value init = buildInitTensor(rewriter, loc, type, lhs);

  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{lhs, rhs}, ValueRange{init},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region with block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type elemType = type.getElementType();
  body->addArgument(elemType, loc); // lhs tile/element
  body->addArgument(elemType, loc); // rhs tile/element
  body->addArgument(elemType, loc); // output tile/element

  rewriter.setInsertionPointToStart(body);
  Value result = rewriter.create<TileOp>(loc, elemType, body->getArgument(0),
                                         body->getArgument(1));
  rewriter.create<YieldOp>(loc, ValueRange{result});

  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

/// Build a ttl.compute op with a single unary tile operation in the body.
template <typename TileOp>
static LogicalResult buildUnaryCompute(Operation *op, PatternRewriter &rewriter,
                                       Value input) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps: (d0, d1, ...) -> (d0, d1, ...)
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  for (int i = 0; i < 2; ++i) { // input, output
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  Value init = buildInitTensor(rewriter, loc, type, input);

  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{input}, ValueRange{init},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region with block arguments
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type elemType = type.getElementType();
  body->addArgument(elemType, loc); // input tile/element
  body->addArgument(elemType, loc); // output tile/element

  rewriter.setInsertionPointToStart(body);
  Value result = rewriter.create<TileOp>(loc, elemType, body->getArgument(0));
  rewriter.create<YieldOp>(loc, ValueRange{result});

  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

/// Build a ttl.compute op with custom unary logic in the body.
template <typename BodyBuilder>
static LogicalResult
buildUnaryComputeCustom(Operation *op, PatternRewriter &rewriter, Value input,
                        BodyBuilder &&bodyBuilder) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Build identity indexing maps
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), ctx);
  for (int i = 0; i < 2; ++i) {
    maps.push_back(AffineMapAttr::get(identityMap));
  }

  // Build iterator types: all parallel
  SmallVector<Attribute> iterTypes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    iterTypes.push_back(rewriter.getStringAttr("parallel"));
  }

  Value init = buildInitTensor(rewriter, loc, type, input);

  // Create ttl.compute op
  auto computeOp = rewriter.create<ComputeOp>(
      loc, TypeRange{type}, ValueRange{input}, ValueRange{init},
      rewriter.getArrayAttr(maps), rewriter.getArrayAttr(iterTypes));

  // Build the body region
  Block *body = rewriter.createBlock(&computeOp.getBody());
  Type elemType = type.getElementType();
  body->addArgument(elemType, loc);
  body->addArgument(elemType, loc);

  rewriter.setInsertionPointToStart(body);
  Value result = bodyBuilder(rewriter, loc, body->getArgument(0));
  rewriter.create<YieldOp>(loc, ValueRange{result});

  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// Templated Elementwise Lowering Patterns (primary: ttl.compute with tile ops)
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
// Custom ops are defined manually below
#define TTL_UNARY_CUSTOM_TILE_OP(TTL_OP)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// Custom Lowering Patterns (operations requiring special handling)
//===----------------------------------------------------------------------===//

/// ReLU: lower to ttl.compute with ttl.tile_relu in the body.
struct LowerRelu : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryCompute<TileReluOp>(op.getOperation(), rewriter,
                                         op.getInput());
  }
};

/// Sigmoid: lower to ttl.compute with ttl.tile_sigmoid in the body.
struct LowerSigmoid : OpRewritePattern<SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryCompute<TileSigmoidOp>(op.getOperation(), rewriter,
                                            op.getInput());
  }
};

//===----------------------------------------------------------------------===//
// DST Register Assignment
//===----------------------------------------------------------------------===//

/// Annotate ttl.compute ops with DST register requirements.
/// Returns failure if any operation exceeds the capacity.
static LogicalResult annotateDST(func::FuncOp func, int64_t capacity) {
  bool error = false;
  func.walk([&](ComputeOp op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!resultType) {
      return;
    }
    int64_t tiles = resultType.getNumElements();
    op->setAttr(
        "ttl.dst_required",
        IntegerAttr::get(IntegerType::get(func.getContext(), 32), tiles));
    if (tiles > capacity) {
      error = true;
    }
  });
  return failure(error);
}

/// Remove DST marker ops using the provided rewriter.
static void eraseDSTMarkerOps(func::FuncOp func, IRRewriter &rewriter) {
  // Collect ops first to avoid iterator invalidation during erasure.
  SmallVector<AcquireDSTOp> acquireOps;
  SmallVector<ReleaseDSTOp> releaseOps;
  SmallVector<RequireDSTOp> requireOps;

  func.walk([&](Operation *op) {
    if (auto acquire = dyn_cast<AcquireDSTOp>(op)) {
      acquireOps.push_back(acquire);
    } else if (auto release = dyn_cast<ReleaseDSTOp>(op)) {
      releaseOps.push_back(release);
    } else if (auto require = dyn_cast<RequireDSTOp>(op)) {
      requireOps.push_back(require);
    }
  });

  // Replace AcquireDSTOp with constant 0 and erase.
  for (AcquireDSTOp op : acquireOps) {
    rewriter.setInsertionPoint(op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    rewriter.replaceOp(op, zero);
  }

  // Erase ReleaseDSTOp and RequireDSTOp.
  for (ReleaseDSTOp op : releaseOps) {
    rewriter.eraseOp(op);
  }
  for (RequireDSTOp op : requireOps) {
    rewriter.eraseOp(op);
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementations
//===----------------------------------------------------------------------===//

struct TTLConvertTTLToComputePass
    : public ::impl::TTLConvertTTLToComputeBase<TTLConvertTTLToComputePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    populateTTLToComputePatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

struct TTLAssignDSTRegistersPass
    : public ::impl::TTLAssignDSTRegistersBase<TTLAssignDSTRegistersPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    // Remove DST marker ops.
    eraseDSTMarkerOps(func, rewriter);

    // Annotate ttl.compute ops with DST requirements.
    if (failed(annotateDST(func, /*capacity=*/64))) {
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
#define TTL_UNARY_CUSTOM_TILE_OP(TTL_OP) patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
}

std::unique_ptr<Pass> createTTLConvertTTLToCompute() {
  return std::make_unique<TTLConvertTTLToComputePass>();
}

std::unique_ptr<Pass> createTTLAssignDSTRegisters() {
  return std::make_unique<TTLAssignDSTRegistersPass>();
}

} // namespace mlir::tt::ttl
