// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLCONVERTTTLTOCOMPUTE
#define GEN_PASS_DEF_TTLASSIGNDSTREGISTERS
#include "ttlang/Dialect/TTL/Passes.h.inc"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
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
// Lowering to ttl.compute with tile ops
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

//===----------------------------------------------------------------------===//
// DST Register Assignment
//===----------------------------------------------------------------------===//

/// Assign DST register indices to tile ops in the compute body.
/// Uses a simple sequential allocation strategy as a placeholder.
/// TODO: Implement DSTAllocationStrategy interface with pluggable algorithms
/// (linear-scan, graph-coloring, greedy).
static LogicalResult assignDSTRegisters(func::FuncOp func, int64_t capacity) {
  bool error = false;
  func.walk([&](ComputeOp op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!resultType) {
      return;
    }

    // Check if this compute op would exceed DST capacity (for tiling
    // decisions).
    int64_t tiles = resultType.getNumElements();
    if (tiles > capacity) {
      // TODO: Emit diagnostic about needing to tile for DST capacity.
      error = true;
    }

    // Simple sequential DST index assignment for tile ops in the body.
    // For now, all tile op results go to DST index 0 (in-place computation).
    // Binary ops load both inputs to DST indices 0 and 1.
    // TODO: Use liveness analysis and graph coloring for optimal allocation.
    op.getBody().walk([&](Operation *bodyOp) {
      if (bodyOp->getNumResults() > 0 && !bodyOp->hasAttr("dst_idx")) {
        StringRef opName = bodyOp->getName().getStringRef();
        if (opName.contains("tile_")) {
          // Assign DST index 0 to tile op results by default.
          // In-place operations reuse the same register.
          bodyOp->setAttr(
              "dst_idx",
              IntegerAttr::get(IntegerType::get(func.getContext(), 32), 0));
        }
      }
    });
  });
  return failure(error);
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

    // Assign DST registers and annotate ttl.compute ops.
    if (failed(assignDSTRegisters(func, /*capacity=*/64))) {
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
}

std::unique_ptr<Pass> createTTLConvertTTLToCompute() {
  return std::make_unique<TTLConvertTTLToComputePass>();
}

std::unique_ptr<Pass> createTTLAssignDSTRegisters() {
  return std::make_unique<TTLAssignDSTRegistersPass>();
}

} // namespace mlir::tt::ttl
