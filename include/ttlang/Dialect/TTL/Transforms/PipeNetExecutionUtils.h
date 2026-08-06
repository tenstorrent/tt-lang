//===- PipeNetExecutionUtils.h - PipeNet execution utilities ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for visiting PipeNet callbacks in their
// per-record execution order.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETEXECUTIONUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETEXECUTIONUTILS_H

#include "mlir/IR/Operation.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttl {

/// Whether a callback loop selects records by source or destination.
enum class PipeNetRecordSelection { Source, Destination };

/// A loop that executes one PipeNet callback for each matching record.
struct PipeNetRecordLoop {
  PipeNetRecordsAttr records;
  PipeNetRecordSelection selection;
};

/// The record selected by one active PipeNet callback loop.
struct ActivePipeNetRecord {
  Operation *loopOp = nullptr;
  std::uint64_t recordIndex = 0;
};

/// Return the active record selected by `loopOp`, if present.
std::optional<std::uint64_t>
getActivePipeNetRecordIndex(ArrayRef<ActivePipeNetRecord> activeRecords,
                            Operation *loopOp);

/// Visit operations in their execution order at `coord`.
///
/// PipeNet callback loops execute their complete body once for each matching
/// record. `resolveGeneratedRecordLoop` identifies loops created by lowering;
/// high-level PipeNet foreach operations are recognized directly.
/// Every region whose operations the callback interprets in execution order
/// must contain one block so lexical traversal represents that order.
///
/// `executionCountDivisor` is the product of matching record counts for the
/// enclosing callback loops. It is unknown if that product overflows.
WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<
        WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t> executionCountDivisor)>
        visitOperation);

/// Visit operations within an enclosing PipeNet callback execution.
///
/// This overload composes record traversal with another execution expansion,
/// such as direct function calls. `activeRecords` is restored before return.
WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<
        WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t> executionCountDivisor)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords,
    std::optional<std::uint64_t> executionCountDivisor);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETEXECUTIONUTILS_H
