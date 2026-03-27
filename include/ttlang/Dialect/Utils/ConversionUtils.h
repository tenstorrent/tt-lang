// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
#define TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
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

/// Element-wise scale: values[d] *= scales[d].
inline void scaleByBlockDims(MutableArrayRef<int64_t> values,
                             ArrayRef<int64_t> scales) {
  assert(scales.size() == values.size() &&
         "scales and values must have the same size");
  for (size_t d = 0; d < values.size(); ++d) {
    values[d] *= scales[d];
  }
}

/// Transform a linearized stride from the iteration domain into the operand's
/// coordinate space via an indexing map (compile-time, no IR emission).
///
/// A stride S encodes how much a linearized CB index advances per unit of a
/// loop IV. This function decomposes S into per-dimension steps (using the
/// iteration domain's row-major strides), applies the indexing map to select
/// the relevant dimensions, optionally scales by blockDims, and re-linearizes
/// in the operand shape.
///
/// For identity maps with no blockDims this is a no-op (returns S unchanged).
inline int64_t transformLinearizedStride(
    int64_t stride, ArrayRef<int64_t> iterDomainShape, AffineMap indexingMap,
    ArrayRef<int64_t> operandShape,
    std::optional<ArrayRef<int64_t>> blockDims = std::nullopt) {
  assert(indexingMap.getNumDims() == iterDomainShape.size() &&
         "indexing map domain rank must match iteration domain shape rank");
  assert(indexingMap.getNumResults() == operandShape.size() &&
         "indexing map result count must match operand shape rank");
  assert((!blockDims || blockDims->size() == operandShape.size()) &&
         "blockDims must match operand shape rank when provided");

  // Fast path: identity map with no block scaling is a no-op.
  if (indexingMap.isIdentity() && !blockDims) {
    return stride;
  }

  // Delinearize stride into per-dimension steps using the iteration domain's
  // row-major strides. For shape [M, N, K] with strides [N*K, K, 1]:
  //   stride=N*K -> components=[1, 0, 0] (one M-step)
  //   stride=K   -> components=[0, 1, 0] (one N-step)
  //   stride=1   -> components=[0, 0, 1] (one K-step)
  SmallVector<int64_t> domainStrides = computeStrides(iterDomainShape);
  SmallVector<int64_t> components(iterDomainShape.size());
  int64_t remaining = stride;
  for (size_t d = 0; d < iterDomainShape.size(); ++d) {
    if (domainStrides[d] > 0) {
      components[d] = remaining / domainStrides[d];
      remaining = remaining % domainStrides[d];
    }
  }
  assert(remaining == 0 &&
         "stride is not a multiple of any domain dimension stride");

  // Apply the indexing map: select dimensions referenced by the map.
  // For projected permutations, each result is an AffineDimExpr.
  SmallVector<int64_t> mappedComponents;
  mappedComponents.reserve(indexingMap.getNumResults());
  for (AffineExpr expr : indexingMap.getResults()) {
    auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr);
    assert(dimExpr && "expected projected permutation (AffineDimExpr results)");
    mappedComponents.push_back(components[dimExpr.getPosition()]);
  }

  // Scale by block dimensions to convert block-level steps to tile-level.
  if (blockDims) {
    scaleByBlockDims(mappedComponents, *blockDims);
  }

  // Re-linearize in the operand's row-major layout.
  SmallVector<int64_t> opStrides = computeStrides(operandShape);
  int64_t result = 0;
  for (size_t d = 0; d < mappedComponents.size(); ++d) {
    result += mappedComponents[d] * opStrides[d];
  }
  return result;
}

/// Emit `IV * stride` (with stride=0 and stride=1 optimizations).
inline Value emitIVTimesStride(OpBuilder &builder, Location loc, Value iv,
                               int64_t stride) {
  if (stride == 1) {
    return iv;
  }
  Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
  return arith::MulIOp::create(builder, loc, iv, strideVal);
}

/// Compute a CB tile index from enclosing loop structure and an indexing map.
///
/// This is the unified mechanism for all CB tile index computation. It
/// collects tile loops, subblock loops, and tile offsets from the enclosing
/// IR, transforms each stride through the indexing map (from iteration-domain
/// space to operand space), and sums the contributions.
///
/// For identity maps (elementwise ops), this produces the same result as the
/// loop strides directly. For non-identity maps (matmul, transpose, reduce,
/// broadcast), it correctly projects out irrelevant dimensions.
///
/// Assumes DMA kernels write tiles into CBs in row-major order (interleaved
/// layout). Sharded layouts with different CB tile orderings would require
/// a different linearization scheme.
///
/// Parameters:
///   - indexingMap: transforms iteration-domain coordinates to operand space
///   - iterDomainShape: full output tensor shape (for delinearizing strides)
///   - operandShape: this operand's tensor shape (for re-linearizing)
///   - cbShapeRank: retain only innermost N tile loops (0 = use all)
///   - blockDims: per-operand-dimension block sizes (std::nullopt = unit dims)
inline FailureOr<Value>
computeCBTileIndex(Operation *op, OpBuilder &builder, AffineMap indexingMap,
                   ArrayRef<int64_t> iterDomainShape,
                   ArrayRef<int64_t> operandShape, size_t cbShapeRank = 0,
                   std::optional<ArrayRef<int64_t>> blockDims = std::nullopt) {
  // Collect enclosing scf.for loops from innermost to outermost.
  SmallVector<scf::ForOp> allLoops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      allLoops.push_back(forOp);
    }
  }

  // Classify loops by attribute.
  SmallVector<scf::ForOp> tileLoops;
  SmallVector<scf::ForOp> subblockLoops;
  for (scf::ForOp loop : allLoops) {
    if (loop->hasAttr(kTileLoopStrideAttrName)) {
      tileLoops.push_back(loop);
    } else if (loop->hasAttr(kSubblockLoopStrideAttrName)) {
      subblockLoops.push_back(loop);
    }
    // Unmarked loops are ignored (user loops, external loops).
  }

  // Retain only the innermost cbShapeRank tile loops.
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
    if (!getConstantIntValue(loop.getUpperBound())) {
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
    if (!getConstantIntValue(loop.getStep())) {
      return op->emitOpError() << "enclosing subblock loop has dynamic step; "
                               << "expected constant step from subblock loops";
    }
  }

  Location loc = op->getLoc();
  Value result = arith::ConstantIndexOp::create(builder, loc, 0);

  // Tile loop contributions (outermost first).
  for (scf::ForOp loop : llvm::reverse(tileLoops)) {
    auto strideAttr = loop->getAttrOfType<IntegerAttr>(kTileLoopStrideAttrName);
    if (!strideAttr) {
      return op->emitOpError() << "enclosing tile loop missing stride value on "
                               << kTileLoopStrideAttrName << " attribute";
    }
    int64_t stride =
        transformLinearizedStride(strideAttr.getInt(), iterDomainShape,
                                  indexingMap, operandShape, blockDims);
    if (stride == 0) {
      continue;
    }
    Value term =
        emitIVTimesStride(builder, loc, loop.getInductionVar(), stride);
    result = arith::AddIOp::create(builder, loc, result, term);
  }

  // Subblock loop contributions. Order is irrelevant (addition is
  // commutative); each loop carries its own stride attribute.
  for (scf::ForOp loop : subblockLoops) {
    auto strideAttr =
        loop->getAttrOfType<IntegerAttr>(kSubblockLoopStrideAttrName);
    if (!strideAttr) {
      return op->emitOpError()
             << "enclosing subblock loop missing stride value on "
             << kSubblockLoopStrideAttrName << " attribute";
    }
    int64_t stride =
        transformLinearizedStride(strideAttr.getInt(), iterDomainShape,
                                  indexingMap, operandShape, blockDims);
    if (stride == 0) {
      continue;
    }
    Value term =
        emitIVTimesStride(builder, loc, loop.getInductionVar(), stride);
    result = arith::AddIOp::create(builder, loc, result, term);
  }

  // Per-tile offset from unrolled emission.
  if (auto tileOffset = op->getAttrOfType<IntegerAttr>(kTileOffsetAttrName)) {
    int64_t offset =
        transformLinearizedStride(tileOffset.getInt(), iterDomainShape,
                                  indexingMap, operandShape, blockDims);
    if (offset != 0) {
      Value offsetVal = arith::ConstantIndexOp::create(builder, loc, offset);
      result = arith::AddIOp::create(builder, loc, result, offsetVal);
    }
  }

  return result;
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

  auto cast = UnrealizedConversionCastOp::create(rewriter, loc, ttkCbTy, cb);
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
