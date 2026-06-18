// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
#define TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/Twine.h"

namespace mlir::tt::ttl::utils {

/// Convert a local DFB index (within a subblock) to a global DFB index (within
/// the full block) when `operand` traces to a tensor.extract_slice.
///
/// Delinearizes the local index into per-dimension coordinates using the
/// slice shape, adds the slice offsets per-dimension, and relinearizes
/// against the source (full) shape.
///
/// Example: local index 5 in a [3,2] slice at offset [0,2] of a [3,4] tensor:
///   delinearize(5, [3,2]) -> (2,1), add [0,2] -> (2,3),
///   linearize((2,3), [3,4]) -> 11.
///
/// Returns `localIndex` unchanged if no extract_slice is found.
inline Value addSliceOffset(Value operand, Value localIndex, OpBuilder &builder,
                            Location loc) {
  // Trace through tensor.extract (tile extraction from lower-to-loops).
  Value tensor = operand;
  if (auto extract = tensor.getDefiningOp<mlir::tensor::ExtractOp>()) {
    tensor = extract.getTensor();
  }
  // Skip past any `ttl.attach_cb` (SSA identity) so the next
  // `getDefiningOp` finds the extract_slice rather than the attach_cb.
  while (auto attach = tensor.getDefiningOp<mlir::tt::ttl::AttachCBOp>()) {
    tensor = attach.getTensor();
  }
  auto slice = tensor.getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!slice) {
    return localIndex;
  }
  auto sliceType = mlir::cast<RankedTensorType>(slice.getResult().getType());
  auto sourceType = mlir::cast<RankedTensorType>(slice.getSource().getType());

  // Delinearize local index into per-dimension coordinates within the slice.
  auto delinearized = affine::AffineDelinearizeIndexOp::create(
      builder, loc, localIndex, sliceType.getShape());

  // Add slice offsets per-dimension to produce global coordinates.
  SmallVector<Value> globalCoords;
  auto mixedOffsets = slice.getMixedOffsets();
  for (size_t d = 0; d < mixedOffsets.size(); ++d) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, mixedOffsets[d]);
    Value global =
        arith::AddIOp::create(builder, loc, delinearized.getResult(d), offset);
    globalCoords.push_back(global);
  }

  // Relinearize against the full source shape.
  return affine::AffineLinearizeIndexOp::create(builder, loc, globalCoords,
                                                sourceType.getShape());
}

/// Convert a TTL CircularBufferType value to a TTKernel CBType, or return
/// it unchanged if already converted.
inline FailureOr<Value>
convertTTLCBToTTKernel(Value cb, ConversionPatternRewriter &rewriter,
                       Location loc,
                       const TypeConverter *typeConverter = nullptr) {
  namespace ttk = mlir::tt::ttkernel;

  if (mlir::isa<ttk::CBType>(cb.getType())) {
    return cb;
  }

  auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(cb.getType());
  if (!ttlCbTy) {
    return failure();
  }

  Type ttkCbTy =
      ttk::CBType::get(ttlCbTy.getContext(), ttlCbTy.getTotalElements(),
                       ttlCbTy.getElementType());

  if (typeConverter) {
    Value result =
        typeConverter->materializeTargetConversion(rewriter, loc, ttkCbTy, cb);
    if (!result) {
      return failure();
    }
    return result;
  }

  auto cast = UnrealizedConversionCastOp::create(rewriter, loc, ttkCbTy, cb);
  return cast.getResult(0);
}

/// Materialize an integer value representing the bit pattern of a float-typed
/// SSA value. Handles three cases (checked in order):
///   1. arith.truncf (e.g. f32 -> bf16) -- recursively materialize the wider
///      source bits, then extract the upper target-width bits via shift+trunc.
///      bf16 is the upper 16 bits of the f32 IEEE-754 encoding.
///   2. unrealized_conversion_cast(iN -> fN) from a prior
///   RawElementReadLowering
///      -- unwrap to get the integer directly.
///   3. arith.constant float -- create the integer bit pattern.
inline FailureOr<Value> materializeIntBits(Value floatVal, Type intTy,
                                           OpBuilder &builder, Location loc) {
  if (auto truncOp = floatVal.getDefiningOp<arith::TruncFOp>()) {
    Value src = truncOp.getOperand();
    unsigned srcWidth = src.getType().getIntOrFloatBitWidth();
    unsigned dstWidth = floatVal.getType().getIntOrFloatBitWidth();
    auto srcIntTy = IntegerType::get(builder.getContext(), srcWidth);
    auto srcBits = materializeIntBits(src, srcIntTy, builder, loc);
    if (failed(srcBits)) {
      return failure();
    }
    Value shift = arith::ConstantIntOp::create(builder, loc,
                                               srcWidth - dstWidth, srcWidth);
    Value shifted = arith::ShRUIOp::create(builder, loc, *srcBits, shift);
    Value truncated = arith::TruncIOp::create(builder, loc, intTy, shifted);
    return truncated;
  }
  if (auto cast = floatVal.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1 &&
        cast.getInputs()[0].getType() == intTy) {
      return cast.getInputs()[0];
    }
  }
  if (auto constOp = floatVal.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = mlir::dyn_cast<FloatAttr>(constOp.getValue())) {
      APInt bits = floatAttr.getValue().bitcastToAPInt();
      Value intConst = arith::ConstantIntOp::create(
          builder, loc, bits.getZExtValue(), bits.getBitWidth());
      return intConst;
    }
  }
  return failure();
}

/// Run applyPartialConversion, capturing the first diagnostic on failure.
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
