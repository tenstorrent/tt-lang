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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>

namespace mlir::tt::ttl {

/// Whether a callback loop selects records by source or destination.
enum class PipeNetRecordSelection { Source, Destination };

/// A loop that executes one PipeNet callback for each matching record.
struct PipeNetRecordLoop {
  PipeNetRecordsAttr records;
  PipeNetRecordSelection selection;
  /// Empty when the loop induction value is the original record index.
  std::map<std::pair<LaunchExecutionLocation, std::uint64_t>, std::uint64_t>
      indirectInductionValues;
};

/// Return the loop induction value that selects `recordIndex` at `location`.
std::optional<std::uint64_t>
getPipeNetRecordLoopInductionValue(const PipeNetRecordLoop &recordLoop,
                                   const LaunchExecutionLocation &location,
                                   std::uint64_t recordIndex);

/// The record selected by one active PipeNet callback loop.
struct ActivePipeNetRecord {
  Operation *loopOp = nullptr;
  std::uint64_t recordIndex = 0;
};

/// Return the active record selected by `loopOp`, if present.
std::optional<std::uint64_t>
getActivePipeNetRecordIndex(ArrayRef<ActivePipeNetRecord> activeRecords,
                            Operation *loopOp);

/// Evaluate a selected-pipe accessor using one concrete record.
std::optional<llvm::APInt>
evaluateSelectedPipeRecordValue(Value value, PipeRecordAttr record);

/// Evaluate a selected-pipe accessor from its active callback record.
/// `resolveFunctionArgument` maps helper arguments to the active call site.
std::optional<llvm::APInt> evaluateActivePipeNetRecordValue(
    Value value, ArrayRef<ActivePipeNetRecord> activeRecords,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveFunctionArgument);

/// Execution facts for active callback records at one location.
struct ActivePipeNetExecution {
  /// False when at least one selected record cannot execute at the location.
  bool mayExecute = true;
  /// Product of matching record counts for the enclosing callback loops.
  /// Unknown when a loop or device comparison is unresolved, or on overflow.
  std::optional<std::uint64_t> countDivisor = 1;
};

/// Evaluate the selected records and matching-record count at `location`.
ActivePipeNetExecution evaluateActivePipeNetExecution(
    ArrayRef<ActivePipeNetRecord> activeRecords,
    const LaunchExecutionLocation &location,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop);

/// Visit operations in their execution order at `coord`.
///
/// PipeNet callback loops execute their complete body once for each matching
/// record. `resolveGeneratedRecordLoop` identifies loops created by lowering;
/// high-level PipeNet foreach operations are recognized directly.
/// Every region whose operations the callback interprets in execution order
/// must contain one block so lexical traversal represents that order.
WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>)>
        visitOperation);

/// Visit operations within an enclosing PipeNet callback execution.
///
/// This overload composes record traversal with another execution expansion,
/// such as direct function calls. `activeRecords` is restored before return.
WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETEXECUTIONUTILS_H
