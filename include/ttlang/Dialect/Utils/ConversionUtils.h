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
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/Twine.h"

namespace mlir::tt::ttl::utils {

/// Compute CB tile index from enclosing scf.for loops with an optional
/// stride transform applied to each constant stride before IR emission.
///
/// Only considers loops annotated by the compiler:
/// - Tile iteration loops (ttl.tile_loop): stride from attribute.
/// - Subblock loops (ttl.subblock_stride): stride from attribute.
/// Unmarked loops (user loops, external loops) are ignored.
///
/// The strideTransform is applied to each stride and tile_offset at C++
/// compile time (integer arithmetic on constants). This enables callers
/// to extract per-dimension components (e.g., stride/numCols for row index)
/// without generating runtime divui/remui operations.
///
/// When cbShapeRank > 0, only the innermost cbShapeRank tile loops are used.
/// Returns constant 0 if not inside any recognized loops.
///
/// Returns failure with diagnostics for unexpected loop structures (dynamic
/// bounds, non-zero lower bounds, etc.).
inline FailureOr<Value> computeCBTileIndexFromLoops(
    Operation *op, OpBuilder &builder, size_t cbShapeRank,
    llvm::function_ref<int64_t(int64_t)> strideTransform) {
  // Collect enclosing scf.for loops from innermost to outermost.
  SmallVector<scf::ForOp> allLoops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      allLoops.push_back(forOp);
    }
  }

  // Classify loops by attribute. Unmarked loops are ignored.
  SmallVector<scf::ForOp> tileLoops;
  SmallVector<scf::ForOp> subblockLoops;
  for (scf::ForOp loop : allLoops) {
    if (loop->hasAttr(kTileLoopAttrName)) {
      tileLoops.push_back(loop);
    } else if (loop->hasAttr(kSubblockStrideAttrName)) {
      subblockLoops.push_back(loop);
    }
    // Unmarked loops are ignored (user loops, external loops).
  }

  // Apply cbShapeRank clipping to tile loops only.
  if (cbShapeRank > 0 && tileLoops.size() > cbShapeRank) {
    tileLoops.resize(cbShapeRank);
  }

  // Validate tile loops.
  for (scf::ForOp loop : tileLoops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    if (!lb) {
      return op->emitOpError()
             << "enclosing tile loop has dynamic lower bound; "
             << "expected constant bounds from tile loops";
    }
    if (*lb != 0) {
      return op->emitOpError()
             << "enclosing tile loop has non-zero lower bound (" << *lb
             << "); expected lb=0 from tile loops";
    }
    auto ub = getConstantIntValue(loop.getUpperBound());
    if (!ub) {
      return op->emitOpError()
             << "enclosing tile loop has dynamic upper bound; "
             << "expected constant bounds from tile loops";
    }
  }

  // Validate subblock loops.
  for (scf::ForOp loop : subblockLoops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    if (!lb) {
      return op->emitOpError()
             << "enclosing subblock loop has dynamic lower bound; "
             << "expected constant bounds from subblock loops";
    }
    if (*lb != 0) {
      return op->emitOpError()
             << "enclosing subblock loop has non-zero lower bound (" << *lb
             << "); expected lb=0 from subblock loops";
    }
    auto step = getConstantIntValue(loop.getStep());
    if (!step) {
      return op->emitOpError() << "enclosing subblock loop has dynamic step; "
                               << "expected constant step from subblock loops";
    }
  }

  Location loc = op->getLoc();

  // Compute index: sum(IV * transform(stride)) + transform(tile_offset).
  // Tile loops processed outermost first for row-major ordering.
  Value result = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (scf::ForOp loop : llvm::reverse(tileLoops)) {
    auto strideAttr = loop->getAttrOfType<IntegerAttr>(kTileLoopAttrName);
    if (!strideAttr) {
      return op->emitOpError() << "enclosing tile loop missing stride value on "
                               << kTileLoopAttrName << " attribute";
    }
    int64_t stride = strideTransform(strideAttr.getInt());
    if (stride == 0) {
      continue;
    }
    Value term;
    if (stride == 1) {
      term = loop.getInductionVar();
    } else {
      Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
      term =
          builder.create<arith::MulIOp>(loc, loop.getInductionVar(), strideVal);
    }
    result = builder.create<arith::AddIOp>(loc, result, term);
  }

  // Add subblock offsets: IV * transform(stride) for each subblock loop.
  for (scf::ForOp loop : subblockLoops) {
    auto strideAttr = loop->getAttrOfType<IntegerAttr>(kSubblockStrideAttrName);
    int64_t stride = strideTransform(strideAttr.getInt());
    if (stride == 0) {
      continue;
    }
    Value offset;
    if (stride == 1) {
      offset = loop.getInductionVar();
    } else {
      Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
      offset =
          builder.create<arith::MulIOp>(loc, loop.getInductionVar(), strideVal);
    }
    result = builder.create<arith::AddIOp>(loc, result, offset);
  }

  // Add per-tile offset from unrolled emission.
  if (auto tileOffset = op->getAttrOfType<IntegerAttr>(kTileOffsetAttrName)) {
    int64_t offset = strideTransform(tileOffset.getInt());
    if (offset != 0) {
      Value offsetVal = builder.create<arith::ConstantIndexOp>(loc, offset);
      result = builder.create<arith::AddIOp>(loc, result, offsetVal);
    }
  }

  return result;
}

/// Convenience overload: identity transform (linearized index).
inline FailureOr<Value> computeCBTileIndexFromLoops(Operation *op,
                                                    OpBuilder &builder,
                                                    size_t cbShapeRank = 0) {
  return computeCBTileIndexFromLoops(op, builder, cbShapeRank,
                                     [](int64_t s) { return s; });
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
