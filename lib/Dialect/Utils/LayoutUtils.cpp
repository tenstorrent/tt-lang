// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/Utils/LayoutUtils.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::tt::ttl::utils {

/// Check if TTNNLayoutAttr represents a row-major layout.
static bool isRowMajorLayout(ttnn::TTNNLayoutAttr layoutAttr) {
  return layoutAttr.getLayout() == ttnn::Layout::RowMajor;
}

/// Check if TTNNLayoutAttr uses interleaved memory layout.
static bool isInterleavedMemoryLayout(ttnn::TTNNLayoutAttr layoutAttr) {
  auto memLayoutOpt = layoutAttr.getMemLayoutOpt();
  return memLayoutOpt &&
         (*memLayoutOpt == ttnn::TensorMemoryLayout::Interleaved);
}

/// Check if TTNNLayoutAttr has an identity affine map (no transposition).
static bool hasIdentityAffineMap(ttnn::TTNNLayoutAttr layoutAttr) {
  return layoutAttr.getLinear().isIdentity();
}

LayoutContiguityInfo analyzeLayoutContiguity(RankedTensorType tensorTy) {
  ArrayRef<int64_t> shape = tensorTy.getShape();
  int64_t elemBits = tensorTy.getElementType().getIntOrFloatBitWidth();
  int64_t elemByteWidth = elemBits / 8;

  // Compute total elements and size
  int64_t totalElements = 1;
  for (int64_t dim : shape) {
    totalElements *= dim;
  }
  int64_t totalSizeBytes = totalElements * elemByteWidth;

  // Compute row information (assuming 2D or higher)
  int64_t numRows = 1;
  int64_t rowSizeBytes = elemByteWidth;
  if (shape.size() >= 2) {
    // For 2D: rows = shape[0], cols = shape[1]
    // For ND: rows = product of all dims except last, cols = last dim
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      numRows *= shape[i];
    }
    rowSizeBytes = shape.back() * elemByteWidth;
  } else if (shape.size() == 1) {
    numRows = 1;
    rowSizeBytes = shape[0] * elemByteWidth;
  }

  int64_t rowStrideBytes = rowSizeBytes; // Default: assume contiguous

  // Default result (conservative fallback)
  ContiguityLevel level = ContiguityLevel::TileContiguous;
  bool isRowMajor = false;
  bool hasPadding = false;

  // Extract TTNNLayoutAttr if present
  auto encoding = tensorTy.getEncoding();
  if (!encoding) {
    // No encoding - conservative fallback
    return LayoutContiguityInfo{level, totalElements, totalSizeBytes,
                                rowSizeBytes, rowStrideBytes, numRows,
                                elemByteWidth, isRowMajor, hasPadding};
  }

  auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding);
  if (!layoutAttr) {
    // Not TTNNLayoutAttr - conservative fallback
    return LayoutContiguityInfo{level, totalElements, totalSizeBytes,
                                rowSizeBytes, rowStrideBytes, numRows,
                                elemByteWidth, isRowMajor, hasPadding};
  }

  // Check layout properties using helper functions
  isRowMajor = isRowMajorLayout(layoutAttr);
  bool isInterleaved = isInterleavedMemoryLayout(layoutAttr);
  bool isIdentityMap = hasIdentityAffineMap(layoutAttr);

  // TODO(#82): Handle all TTNNLayoutAttr variations (see issue).
  // TODO: Support sharded layouts with per-shard block transfers.
  // TODO: Support transposed/permuted layouts.

  // Determine contiguity level
  if (isRowMajor && isInterleaved && isIdentityMap) {
    // For now, assume interleaved row-major with identity map is fully contiguous
    // In reality, we might need to check shard layout for padding
    level = ContiguityLevel::FullyContiguous;
    hasPadding = false;
  } else if (isRowMajor && isIdentityMap) {
    // Row-major with identity map but not interleaved (might be sharded)
    // Conservative: assume rows are contiguous but may have padding
    level = ContiguityLevel::RowContiguous;
    hasPadding = true;
  } else {
    // Non-row-major, non-identity, or complex layout
    level = ContiguityLevel::TileContiguous;
  }

  return LayoutContiguityInfo{level, totalElements, totalSizeBytes,
                              rowSizeBytes, rowStrideBytes, numRows,
                              elemByteWidth, isRowMajor, hasPadding};
}

} // namespace mlir::tt::ttl::utils
