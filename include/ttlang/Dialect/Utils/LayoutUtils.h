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

inline ContiguousLayoutInfo computeContiguousLayout(RankedTensorType tensorTy) {
  ArrayRef<int64_t> shape = tensorTy.getShape();
  // TODO(ttl): Replace this contiguous fallback with stride/page derivation
  // from the tensor's layout encoding (e.g., TTNNLayoutAttr).
  // Issue: #000.
  int64_t rowStrideElems = shape.size() >= 2 ? shape.back() : 1;
  int64_t colStrideElems = 1;

  int64_t elemBits = tensorTy.getElementType().getIntOrFloatBitWidth();
  int64_t elemByteWidth = elemBits / 8;

  // TODO(ttl): Derive page size from actual tiling/sharding when available.
  // Issue: #000.
  int64_t pageSizeBytes = elemByteWidth * rowStrideElems;

  return {rowStrideElems, colStrideElems, elemByteWidth, pageSizeBytes};
}

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_LAYOUTUTILS_H
