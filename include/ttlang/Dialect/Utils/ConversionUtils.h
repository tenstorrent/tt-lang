// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
#define TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/Twine.h"

namespace mlir::tt::ttl::utils {

/// Collect enclosing scf.for loops from innermost to outermost.
inline SmallVector<scf::ForOp> collectEnclosingLoops(Operation *op) {
  SmallVector<scf::ForOp> loops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      loops.push_back(forOp);
    }
  }
  return loops;
}

/// Compute linearized index from loop induction variables.
/// For loops with IVs [iv0, iv1, ...] and bounds [ub0, ub1, ...],
/// computes: iv0 * (ub1 * ub2 * ...) + iv1 * (ub2 * ...) + ...
inline Value linearizeLoopIndices(OpBuilder &builder, Location loc,
                                  SmallVectorImpl<scf::ForOp> &loops) {
  if (loops.empty()) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }

  Value result = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (size_t i = 0; i < loops.size(); ++i) {
    scf::ForOp loop = loops[loops.size() - 1 - i];
    Value stride = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (size_t j = i + 1; j < loops.size(); ++j) {
      scf::ForOp innerLoop = loops[loops.size() - 1 - j];
      stride =
          builder.create<arith::MulIOp>(loc, stride, innerLoop.getUpperBound());
    }
    Value term = builder.create<arith::MulIOp>(loc, loop.getInductionVar(), stride);
    result = builder.create<arith::AddIOp>(loc, result, term);
  }
  return result;
}

/// Compute linearized CB tile index from enclosing scf.for loops.
/// When cbShapeRank > 0, only the innermost cbShapeRank loops are used.
/// Returns constant 0 if not inside any loops.
inline Value computeCBTileIndexFromLoops(Operation *op, OpBuilder &builder,
                                         size_t cbShapeRank = 0) {
  SmallVector<scf::ForOp> loops = collectEnclosingLoops(op);

  if (cbShapeRank > 0 && loops.size() > cbShapeRank) {
    loops.resize(cbShapeRank);
  }

  for (scf::ForOp loop : loops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    assert(lb && *lb == 0 &&
           "computeCBTileIndexFromLoops: expected lower bound of 0");
    auto ub = getConstantIntValue(loop.getUpperBound());
    assert(ub && "computeCBTileIndexFromLoops: expected constant upper bound");
    auto step = getConstantIntValue(loop.getStep());
    assert(step && *step == 1 &&
           "computeCBTileIndexFromLoops: expected step of 1");
  }

  return linearizeLoopIndices(builder, op->getLoc(), loops);
}

/// Convert a TTL CircularBufferType value to a TTKernel CBType value.
/// If the value is already a TTKernel CB, returns it unchanged.
/// Uses the TypeConverter to materialize the conversion when provided,
/// otherwise creates an UnrealizedConversionCastOp directly.
inline FailureOr<Value>
convertTTLCBToTTKernel(Value cb, ConversionPatternRewriter &rewriter,
                       Location loc,
                       const TypeConverter *typeConverter = nullptr) {
  namespace ttk = mlir::tt::ttkernel;

  // Already converted.
  if (mlir::isa<ttk::CBType>(cb.getType())) {
    return cb;
  }

  // Convert TTL CB to TTKernel CB.
  auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(cb.getType());
  if (!ttlCbTy) {
    return failure();
  }

  Type ttkCbTy =
      ttk::CBType::get(ttlCbTy.getContext(), ttlCbTy.getTotalElements(),
                       ttlCbTy.getElementType());

  // Use type converter if provided, otherwise create cast directly.
  if (typeConverter) {
    Value result =
        typeConverter->materializeTargetConversion(rewriter, loc, ttkCbTy, cb);
    if (!result) {
      return failure();
    }
    return result;
  }

  auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, ttkCbTy, cb);
  return cast.getResult(0);
}

/// Runs applyPartialConversion while capturing the first diagnostic emitted
/// during conversion. Returns true on failure and populates `capturedDiag`
/// with either the captured diagnostic or a generic message that includes the
/// pass name.
inline bool
applyPartialConversionWithDiag(Operation *root, ConversionTarget &target,
                               const FrozenRewritePatternSet &patterns,
                               StringRef passName, std::string &capturedDiag) {
  bool failedConv = false;
  {
    ScopedDiagnosticHandler handler(root->getContext(), [&](Diagnostic &diag) {
      if (capturedDiag.empty()) {
        capturedDiag = diag.str();
      }
      return success();
    });
    failedConv = failed(applyPartialConversion(root, target, patterns));
  }

  if (failedConv && capturedDiag.empty()) {
    capturedDiag =
        (llvm::Twine(passName) + " failed during legalization").str();
  }
  return failedConv;
}

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
