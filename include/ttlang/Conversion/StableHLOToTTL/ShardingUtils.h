// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_CONVERSION_STABLEHLOTOTLL_SHARDINGUTILS_H
#define TTLANG_CONVERSION_STABLEHLOTOTLL_SHARDINGUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::sdy {
class MeshOp;
class TensorShardingAttr;
} // namespace mlir::sdy

namespace mlir::tt::ttl {

/// Mesh topology extracted from sdy.mesh.
struct MeshInfo {
  llvm::SmallVector<int64_t> shape; // e.g., {2, 4}
  llvm::SmallVector<llvm::StringRef> axes; // e.g., {"x", "y"}
};

/// Per-tensor sharding info derived from sdy.manual_computation block args.
struct TensorShardInfo {
  llvm::SmallVector<int64_t> localShape; // per-core shape from block arg type
  llvm::SmallVector<int64_t> tileShape;  // localShape / 32
  Type elementType;
};

/// Parse sdy.mesh op into MeshInfo.
MeshInfo parseMesh(mlir::sdy::MeshOp meshOp);

/// Compute tile shape from a local tensor type. Errors if not tile-aligned
/// (all dims must be multiples of 32) or if the type has dynamic shapes.
mlir::FailureOr<TensorShardInfo>
parseTensorInfo(RankedTensorType localType, Location loc);

} // namespace mlir::tt::ttl

#endif // TTLANG_CONVERSION_STABLEHLOTOTLL_SHARDINGUTILS_H
