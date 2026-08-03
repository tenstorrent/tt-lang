// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TILEEXECUTION_H
#define TTLANG_DIALECT_TTL_IR_TILEEXECUTION_H

#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttl {

/// Tile operation category used by configuration capability queries.
enum class TilePrimitive {
  Copy,
  ElementwiseBinary,
  ElementwiseUnary,
  Broadcast,
  Reduce,
  Transpose,
  Fill,
  Matmul,
  Store,
  DstIndex,
};

/// Hardware location from which a tile operand is consumed.
enum class TileOperandRoute {
  None,
  DataflowBuffer,
  Dst,
};

/// Target-independent category for optional full-fp32 accumulation.
enum class FullFp32AccumulationKind {
  Matmul,
  ReduceRow,
  ReduceColumn,
  ReduceScalar,
};

/// Target-independent execution semantics for one tile operation.
struct TileExecutionInfo {
  TilePrimitive primitive;
  llvm::SmallVector<TileOperandRoute, 4> operandRoutes;
  bool resultInDst = false;
  std::optional<FullFp32AccumulationKind> fullFp32Accumulation;
  /// Residual contents in a reused destination slot affect the result.
  bool accumulatesIntoDst = false;
};

/// Return strategies structurally permitted by the operation's SSA operands.
llvm::SmallVector<TileExecutionStrategy, 2>
getDefaultLegalTileExecutionStrategies(mlir::Operation *operation);

/// Return execution semantics for `operation` under `strategy`.
mlir::FailureOr<TileExecutionInfo> getDefaultTileExecutionInfo(
    mlir::Operation *operation,
    std::optional<TileExecutionStrategy> strategy = std::nullopt);

/// Verify that execution semantics define one route for every operand.
mlir::LogicalResult verifyTileExecutionInfo(mlir::Operation *operation,
                                            const TileExecutionInfo &info);

/// Return the strategy recorded on an operation with strategy alternatives.
mlir::FailureOr<TileExecutionStrategy>
getSelectedTileExecutionStrategy(mlir::Operation *operation);

/// Return execution semantics using the strategy recorded on the operation.
mlir::FailureOr<TileExecutionInfo>
getSelectedTileExecutionInfo(mlir::Operation *operation);

/// Return whether `operand` must be resident in DST when consumed.
bool isDstInput(mlir::OpOperand &operand);

/// Verify that every tile operation has complete execution semantics.
mlir::LogicalResult verifyTileExecutionSemantics(mlir::Operation *root);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_IR_TILEEXECUTION_H
