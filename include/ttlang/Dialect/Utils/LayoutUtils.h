// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_LAYOUTUTILS_H
#define TTLANG_DIALECT_UTILS_LAYOUTUTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttl::utils {

/// Lightweight stride/page computation for contiguous row-major tensors.
/// This is a placeholder until TTL tensor encodings provide real layout/tiling.
struct ContiguousLayoutInfo {
  int64_t rowStrideElems;
  int64_t colStrideElems;
  int64_t elemByteWidth;
  int64_t pageSizeBytes;
};

/// Contiguity classification for tensor layouts.
/// Determines the granularity of block transfers that can be used.
enum class ContiguityLevel {
  FullyContiguous, // Entire tensor is contiguous - single block transfer
  RowContiguous,   // Each row is contiguous - per-row block transfers
  TileContiguous,  // Only 32x32 tiles are contiguous - tile-level transfers
  NonContiguous    // Scattered/complex layout - error case
};

/// Extended layout analysis result with contiguity information.
/// Used to determine optimal data transfer strategy for ttl.copy lowering.
struct LayoutContiguityInfo {
  ContiguityLevel level;
  int64_t totalElements;  // Total elements in tensor
  int64_t totalSizeBytes; // Total size in bytes
  int64_t rowSizeBytes;   // Bytes per contiguous row
  int64_t rowStrideBytes; // Bytes between row starts (may include padding)
  int64_t numRows;        // Number of rows
  int64_t elemByteWidth;  // Bytes per element
  bool isRowMajor;
  bool hasPadding;
};

/// Analyze tensor layout contiguity from TTNNLayoutAttr.
///
/// Determines the optimal data transfer strategy by inspecting the tensor's
/// layout encoding. The analysis checks:
/// 1. Layout type (RowMajor vs Tile)
/// 2. Memory layout (Interleaved vs Sharded)
/// 3. Affine map (identity vs permuted)
///
/// Returns:
/// - FullyContiguous: Entire tensor can be transferred as one block
/// - RowContiguous: Each row can be transferred as a block (with padding)
/// - TileContiguous: Fall back to per-tile transfers (current behavior)
/// - NonContiguous: Complex/unsupported layout
LayoutContiguityInfo analyzeLayoutContiguity(RankedTensorType tensorTy);

inline ContiguousLayoutInfo computeContiguousLayout(RankedTensorType tensorTy) {
  ArrayRef<int64_t> shape = tensorTy.getShape();
  // TODO(ttl): Replace this contiguous fallback with stride/page derivation
  // from the tensor's layout encoding (e.g., TTNNLayoutAttr).
  // Issue: #82.
  int64_t rowStrideElems = shape.size() >= 2 ? shape.back() : 1;
  int64_t colStrideElems = 1;

  int64_t elemBits = tensorTy.getElementType().getIntOrFloatBitWidth();
  int64_t elemByteWidth = elemBits / 8;

  // TODO(#83): Derive page size from actual tiling/sharding when available.
  int64_t pageSizeBytes = elemByteWidth * rowStrideElems;

  return {rowStrideElems, colStrideElems, elemByteWidth, pageSizeBytes};
}

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_LAYOUTUTILS_H
