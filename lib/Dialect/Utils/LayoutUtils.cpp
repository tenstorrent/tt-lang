// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/Utils/LayoutUtils.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <climits>

namespace mlir::tt::ttl::utils {

LayoutContiguityInfo analyzeLayoutContiguity(RankedTensorType tensorTy) {
  LayoutContiguityInfo result{};
  ArrayRef<int64_t> shape = tensorTy.getShape();

  // Compute basic size info.
  int64_t elemBits = tensorTy.getElementType().getIntOrFloatBitWidth();
  result.elemByteWidth = elemBits / CHAR_BIT;
  result.totalElements = 1;
  for (int64_t dim : shape) {
    result.totalElements *= dim;
  }
  result.totalSizeBytes = result.totalElements * result.elemByteWidth;

  // Default to TileContiguous (current behavior) if no layout encoding.
  auto encoding = tensorTy.getEncoding();
  if (!encoding) {
    result.level = ContiguityLevel::TileContiguous;
    result.isRowMajor = true;
    result.hasPadding = false;
    if (shape.size() >= 2) {
      result.numRows = shape[shape.size() - 2];
      result.rowSizeBytes = shape.back() * result.elemByteWidth;
      result.rowStrideBytes = result.rowSizeBytes;
    } else {
      result.numRows = 1;
      result.rowSizeBytes = result.totalSizeBytes;
      result.rowStrideBytes = result.rowSizeBytes;
    }
    return result;
  }

  auto layout = mlir::dyn_cast<tt::ttnn::TTNNLayoutAttr>(encoding);
  if (!layout) {
    // Unknown encoding - fall back to tile transfers.
    result.level = ContiguityLevel::TileContiguous;
    return result;
  }

  // Check layout type.
  bool isTiled = layout.isTiled();
  result.isRowMajor = !isTiled;

  // Check memory layout (interleaved vs sharded).
  bool isInterleaved = true;
  if (auto memLayout = layout.getMemLayout()) {
    isInterleaved = !tt::ttnn::isShardedMemoryLayout(memLayout.getValue());
  }

  // Check affine map for identity (no permutation).
  bool hasIdentityMap = true;
  if (auto linearMap = layout.getLinear()) {
    // Identity map: (d0, d1, ...) -> (d0, d1, ...)
    hasIdentityMap = linearMap.isIdentity();
  }

  // Determine contiguity level based on layout properties.
  if (!isTiled && isInterleaved && hasIdentityMap) {
    // RowMajor + Interleaved + Identity = FullyContiguous
    result.level = ContiguityLevel::FullyContiguous;
    result.hasPadding = false;
  } else if (!isTiled && hasIdentityMap) {
    // RowMajor + Identity but sharded = RowContiguous
    result.level = ContiguityLevel::RowContiguous;
    result.hasPadding = true;
  } else if (isTiled) {
    // Tiled layout = TileContiguous (per-tile transfers)
    result.level = ContiguityLevel::TileContiguous;
    result.hasPadding = false;
  } else {
    // Complex layout (permuted map, etc) = NonContiguous
    result.level = ContiguityLevel::NonContiguous;
    result.hasPadding = true;
  }

  // Compute row info for RowContiguous transfers.
  if (shape.size() >= 2) {
    result.numRows = shape[shape.size() - 2];
    result.rowSizeBytes = shape.back() * result.elemByteWidth;
    // For sharded layouts, rowStrideBytes may differ from rowSizeBytes.
    // TODO(#138): Extract actual stride from sharding spec.
    result.rowStrideBytes = result.rowSizeBytes;
  } else {
    result.numRows = 1;
    result.rowSizeBytes = result.totalSizeBytes;
    result.rowStrideBytes = result.rowSizeBytes;
  }

  return result;
}

} // namespace mlir::tt::ttl::utils
