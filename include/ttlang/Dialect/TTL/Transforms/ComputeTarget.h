// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTETARGET_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTETARGET_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <string>

namespace mlir::tt::ttl {

/// Immutable LLK capabilities for one compute target.
class ComputeTargetEnvironment {
public:
  virtual ~ComputeTargetEnvironment() = default;

  static FailureOr<std::unique_ptr<ComputeTargetEnvironment>>
  get(Operation *operation, std::string &failureReason);

  virtual LogicalResult
  validateKernelTileType(ttcore::TileType tileType,
                         std::string &failureReason) const = 0;

  virtual LogicalResult
  validatePrimitiveDataType(ComputePrimitive primitive,
                            ttcore::TileType tileType,
                            std::string &failureReason) const = 0;

  virtual LogicalResult
  validatePrimitiveTileShape(ComputePrimitive primitive,
                             ttcore::TileType tileType,
                             std::string &failureReason) const = 0;

  virtual LogicalResult
  validatePassthroughTileType(ttcore::TileType tileType,
                              std::string &failureReason) const = 0;

  virtual LogicalResult
  validateMatmulTileTypes(ttcore::TileType lhsType, ttcore::TileType rhsType,
                          ttcore::TileType resultType, bool transposeRhs,
                          std::string &failureReason) const = 0;

  LogicalResult validateOperation(Operation *operation,
                                  std::string &failureReason) const;

protected:
  ComputeTargetEnvironment() = default;
};

/// Return the target-independent primitive implemented by a TTL operation.
std::optional<ComputePrimitive> getComputePrimitive(Operation *operation);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTETARGET_H
