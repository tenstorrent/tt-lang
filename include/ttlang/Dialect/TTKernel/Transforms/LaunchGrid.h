// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_TRANSFORMS_LAUNCHGRID_H
#define TTLANG_DIALECT_TTKERNEL_TRANSFORMS_LAUNCHGRID_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <utility>

namespace mlir::tt::ttkernel {

/// Returns the positive (x, y) extents of a `ttl.launch_grid` attribute.
inline FailureOr<std::pair<int64_t, int64_t>> readLaunchGrid(ArrayAttr attr) {
  if (!attr || attr.size() != 2) {
    return failure();
  }
  auto gridXAttr = dyn_cast<IntegerAttr>(attr[0]);
  auto gridYAttr = dyn_cast<IntegerAttr>(attr[1]);
  if (!gridXAttr || !gridYAttr) {
    return failure();
  }
  int64_t gridX = gridXAttr.getInt();
  int64_t gridY = gridYAttr.getInt();
  if (gridX <= 0 || gridY <= 0) {
    return failure();
  }
  return std::pair<int64_t, int64_t>{gridX, gridY};
}

} // namespace mlir::tt::ttkernel

#endif // TTLANG_DIALECT_TTKERNEL_TRANSFORMS_LAUNCHGRID_H
