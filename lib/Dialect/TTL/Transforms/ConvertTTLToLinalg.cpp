// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#define GEN_PASS_DEF_TTLCONVERTTTLTOLINALG
#define GEN_PASS_DEF_TTLASSIGNDSTREGISTERS
#include "ttlang/Dialect/TTL/Passes.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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

template <typename ArithOp>
static LogicalResult buildBinaryGeneric(Operation *op,
                                        PatternRewriter &rewriter, Value lhs,
                                        Value rhs) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  SmallVector<AffineMap> maps(3, AffineMap::getMultiDimIdentityMap(
                                     type.getRank(), rewriter.getContext()));
  SmallVector<utils::IteratorType> iterTypes(type.getRank(),
                                             utils::IteratorType::parallel);

  Value init = buildInitTensor(rewriter, loc, type, lhs);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{type}, ValueRange{lhs, rhs}, ValueRange{init}, maps,
      iterTypes, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value res = b.create<ArithOp>(nestedLoc, args[0], args[1]);
        b.create<linalg::YieldOp>(nestedLoc, res);
      });
  rewriter.replaceOp(op, generic.getResult(0));
  return success();
}

template <typename MathOp>
static LogicalResult buildUnaryGeneric(Operation *op, PatternRewriter &rewriter,
                                       Value input) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  SmallVector<AffineMap> maps(2, AffineMap::getMultiDimIdentityMap(
                                     type.getRank(), rewriter.getContext()));
  SmallVector<utils::IteratorType> iterTypes(type.getRank(),
                                             utils::IteratorType::parallel);
  Value init = buildInitTensor(rewriter, loc, type, input);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{type}, ValueRange{input}, ValueRange{init}, maps,
      iterTypes, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value res = b.create<MathOp>(nestedLoc, args[0]);
        b.create<linalg::YieldOp>(nestedLoc, res);
      });
  rewriter.replaceOp(op, generic.getResult(0));
  return success();
}

template <typename UnaryBuilder>
static LogicalResult
buildUnaryGenericCustom(Operation *op, PatternRewriter &rewriter, Value input,
                        UnaryBuilder &&builder) {
  auto type = getTensorType(op->getResult(0));
  if (!type) {
    return failure();
  }

  Location loc = op->getLoc();
  SmallVector<AffineMap> maps(2, AffineMap::getMultiDimIdentityMap(
                                     type.getRank(), rewriter.getContext()));
  SmallVector<utils::IteratorType> iterTypes(type.getRank(),
                                             utils::IteratorType::parallel);
  Value init = buildInitTensor(rewriter, loc, type, input);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{type}, ValueRange{input}, ValueRange{init}, maps,
      iterTypes, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value res = builder(b, nestedLoc, args[0]);
        b.create<linalg::YieldOp>(nestedLoc, res);
      });
  rewriter.replaceOp(op, generic.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// Templated Elementwise Lowering Patterns
//===----------------------------------------------------------------------===//

/// Generic pattern for binary elementwise ops with direct arith/math mapping.
template <typename TTLOp, typename ArithOp>
struct LowerBinaryElementwise : OpRewritePattern<TTLOp> {
  using OpRewritePattern<TTLOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildBinaryGeneric<ArithOp>(op.getOperation(), rewriter, op.getLhs(),
                                       op.getRhs());
  }
};

/// Generic pattern for unary elementwise ops with direct math mapping.
template <typename TTLOp, typename MathOp>
struct LowerUnaryElementwise : OpRewritePattern<TTLOp> {
  using OpRewritePattern<TTLOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryGeneric<MathOp>(op.getOperation(), rewriter,
                                     op.getInput());
  }
};

//===----------------------------------------------------------------------===//
// Pattern Type Aliases - Generated from TTLElementwiseOps.def
//===----------------------------------------------------------------------===//

// Generate type aliases for binary operations: TTL_OP -> Lower##TTL_OP
#define TTL_BINARY_OP(TTL_OP, ARITH_OP)                                        \
  using Lower##TTL_OP = LowerBinaryElementwise<TTL_OP##Op, ARITH_OP>;
// Generate type aliases for simple unary operations
#define TTL_UNARY_OP(TTL_OP, MATH_OP)                                          \
  using Lower##TTL_OP = LowerUnaryElementwise<TTL_OP##Op, MATH_OP>;
// Custom ops are defined manually below
#define TTL_UNARY_CUSTOM_OP(TTL_OP)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// Custom Lowering Patterns (operations requiring special handling)
//===----------------------------------------------------------------------===//

/// ReLU requires custom lowering: max(x, 0) via compare + select.
struct LowerRelu : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryGenericCustom(
        op.getOperation(), rewriter, op.getInput(),
        [](OpBuilder &b, Location loc, Value v) {
          auto zero = b.create<arith::ConstantOp>(loc, v.getType(),
                                                  b.getZeroAttr(v.getType()));
          auto cmp =
              b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, v, zero);
          return b.create<arith::SelectOp>(loc, cmp, v, zero);
        });
  }
};

//===----------------------------------------------------------------------===//
// DST Register Assignment
//===----------------------------------------------------------------------===//

/// Annotate linalg.generic ops with DST register requirements.
/// Returns failure if any operation exceeds the capacity.
static LogicalResult annotateDST(func::FuncOp func, int64_t capacity) {
  bool error = false;
  func.walk([&](linalg::GenericOp op) {
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

struct TTLConvertTTLToLinalgPass
    : public ::impl::TTLConvertTTLToLinalgBase<TTLConvertTTLToLinalgPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    populateTTLToLinalgPatterns(patterns);
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

    // Annotate linalg.generic ops with DST requirements.
    if (failed(annotateDST(func, /*capacity=*/64))) {
      return signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void populateTTLToLinalgPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Register patterns generated from TTLElementwiseOps.def
#define TTL_BINARY_OP(TTL_OP, ARITH_OP) patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_OP(TTL_OP, MATH_OP) patterns.add<Lower##TTL_OP>(ctx);
#define TTL_UNARY_CUSTOM_OP(TTL_OP) patterns.add<Lower##TTL_OP>(ctx);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
}

std::unique_ptr<Pass> createTTLConvertTTLToLinalg() {
  return std::make_unique<TTLConvertTTLToLinalgPass>();
}

std::unique_ptr<Pass> createTTLAssignDSTRegisters() {
  return std::make_unique<TTLAssignDSTRegistersPass>();
}

} // namespace mlir::tt::ttl
