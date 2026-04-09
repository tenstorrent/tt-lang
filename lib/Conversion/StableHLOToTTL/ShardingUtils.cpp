// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Conversion/StableHLOToTTL/ShardingUtils.h"

#include "shardy/dialect/sdy/ir/dialect.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::tt::ttl {

MeshInfo parseMesh(mlir::sdy::MeshOp meshOp) {
  MeshInfo info;
  for (auto axis : meshOp.getMeshAttr().getAxes()) {
    info.axes.push_back(axis.getName());
    info.shape.push_back(axis.getSize());
  }
  return info;
}

mlir::FailureOr<TensorShardInfo> parseTensorInfo(RankedTensorType localType,
                                                  Location loc) {
  if (!localType.hasStaticShape())
    return emitError(loc, "dynamic shapes not supported");

  auto elementType = localType.getElementType();
  if (!elementType.isBF16() && !elementType.isF16() && !elementType.isF32())
    return emitError(loc, "unsupported element type: ") << elementType;

  TensorShardInfo info;
  info.elementType = elementType;
  info.localShape = llvm::to_vector(localType.getShape());

  static constexpr int64_t kTileSize = 32;
  for (auto [i, dim] : llvm::enumerate(info.localShape)) {
    if (dim % kTileSize != 0)
      return emitError(loc, "local shape dimension ")
             << i << " (" << dim << ") not tile-aligned (must be multiple of "
             << kTileSize << ")";
    info.tileShape.push_back(dim / kTileSize);
  }

  return info;
}

} // namespace mlir::tt::ttl
