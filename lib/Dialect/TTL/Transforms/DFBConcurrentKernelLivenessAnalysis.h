// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H

//===----------------------------------------------------------------------===//
// Concurrent Kernel DFB Liveness Analysis
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

/// Hardware processor that owns one DFB ring pointer.
enum class DFBPointerProcessor { Noc0, Noc1, Pack, Unpack };

/// Ring pointer advanced by a protocol effect.
enum class DFBPointerDirection { Read, Write };

/// Physical owner of one DFB ring pointer on one launched node.
struct DFBPointerOwner {
  LaunchNodeCoord node;
  DFBPointerProcessor processor = DFBPointerProcessor::Noc0;
  DFBPointerDirection direction = DFBPointerDirection::Read;

  bool operator==(const DFBPointerOwner &rhs) const {
    return node == rhs.node && processor == rhs.processor &&
           direction == rhs.direction;
  }

  bool operator!=(const DFBPointerOwner &rhs) const { return !(*this == rhs); }
};

/// Reason that a per-node lifetime did not prove a quiescent handoff.
enum class DFBQuiescenceFailureReason {
  None,
  MissingProtocolEffect,
  UnsupportedControlFlow,
  MismatchedTransaction,
  IncompleteUseOrder,
  UnknownPointerOwner,
};

/// Typed quiescence result retained for conflict evidence.
struct DFBQuiescenceProof {
  DFBQuiescenceFailureReason failure = DFBQuiescenceFailureReason::None;
  Operation *evidence = nullptr;

  bool proven() const { return failure == DFBQuiescenceFailureReason::None; }
};

/// One access event consumed by concurrent-liveness analysis.
///
/// A concrete lifecycle operation contributes one occurrence. An operation
/// with a protocol summary contributes one occurrence per effect, preserving
/// actions on different DFBs as distinct events. A dependency occurrence with
/// no effect contributes an opaque call-duration access.
struct DFBAccessOccurrence {
  /// Operation that performs the access; several occurrences may share it.
  Operation *operation = nullptr;

  /// Null for an opaque call-duration storage access.
  std::optional<DFBProtocolEffectKind> protocolEffect;

  /// Positive for protocol effects and zero for opaque accesses.
  int64_t numTiles = 0;

  /// Execution position among all protocol effects exposed by `operation`.
  unsigned sequenceIndex = 0;

  /// Launched nodes where this occurrence may execute.
  LaunchNodeDomain launchDomain;

  /// Operation that prevented a precise launch domain, or null when precise.
  Operation *unanalyzableDomainOperation = nullptr;
};

/// Execution count retained for one access in the allocation report.
struct DFBDiagnosticAccessOccurrence {
  /// Index into the logical lifecycle's access occurrence list.
  unsigned occurrenceIndex = 0;

  /// Exact executions on the node, or null when the count is unknown.
  std::optional<std::uint64_t> exactExecutionCount;

  bool operator==(const DFBDiagnosticAccessOccurrence &rhs) const {
    return occurrenceIndex == rhs.occurrenceIndex &&
           exactExecutionCount == rhs.exactExecutionCount;
  }
};

/// Per-node data collected only for the allocation report.
struct DFBPerNodeLifetimeDiagnostics {
  /// Execution-count row for every access considered on the launch node.
  SmallVector<DFBDiagnosticAccessOccurrence> occurrences;

  /// Every access whose entry belongs to the minimal lifetime frontier.
  SmallVector<unsigned> earliestAccessOccurrenceIndices;

  /// Accesses whose completion defines the retained terminal frontier.
  SmallVector<unsigned> terminalAccessOccurrenceIndices;

  bool operator==(const DFBPerNodeLifetimeDiagnostics &rhs) const {
    return occurrences == rhs.occurrences &&
           earliestAccessOccurrenceIndices ==
               rhs.earliestAccessOccurrenceIndices &&
           terminalAccessOccurrenceIndices ==
               rhs.terminalAccessOccurrenceIndices;
  }
};

/// Consecutive normalized transactions with one tile count.
struct DFBTransactionRun {
  std::uint64_t executionCount = 0;
  int64_t tilesPerExecution = 0;

  bool operator==(const DFBTransactionRun &rhs) const {
    return executionCount == rhs.executionCount &&
           tilesPerExecution == rhs.tilesPerExecution;
  }
};

/// Advances one physical ring cursor through finite transaction runs. Fails
/// when an acquire would cross the end of the physical allocation.
FailureOr<std::uint64_t>
advanceDFBTransactionCursor(ArrayRef<DFBTransactionRun> transactionRuns,
                            std::uint64_t physicalTileCount,
                            std::uint64_t initialOffset = 0);

/// Protocol state proved for one access interval between synchronized resets.
struct DFBLifecycleEpoch {
  SmallVector<unsigned> accessOccurrenceIndices;
  SmallVector<unsigned> earliestEntryEvents;
  SmallVector<unsigned> terminalCompletionEvents;
  SmallVector<DFBTransactionRun> transactionRuns;
  std::optional<DFBPointerOwner> writePointerOwner;
  std::optional<DFBPointerOwner> readPointerOwner;
  std::optional<int64_t> terminalResetOrdinal;
  bool terminalStateCanonical = false;
  DFBQuiescenceProof quiescence;
};

/// A selected reset write that overlaps a non-target logical DFB lifecycle.
struct DFBResetAllocationConflict {
  unsigned targetLogicalIndex = 0;
  unsigned overlappingLogicalIndex = 0;
  LaunchNodeCoord node;
  SynchronizedDFBResetAttr reset;
  Operation *resetOperation = nullptr;
  Operation *overlappingOperation = nullptr;
};

/// Immutable lifetime and hardware-state facts for one launched node.
struct DFBPerNodeLifetime {
  LaunchNodeCoord node;

  /// Whether an access may execute on `node`.
  bool mayBeActive = true;

  /// Whether all active accesses share one proved conditional execution.
  bool conditionalExecutionProven = false;

  /// Minimal access-entry event IDs under the proved happens-before relation.
  SmallVector<unsigned> earliestEntryEvents;

  /// Completion event IDs after which the DFB is quiescent.
  SmallVector<unsigned> terminalCompletionEvents;
  /// Normalized transaction runs in occurrence order.
  SmallVector<DFBTransactionRun> transactionRuns;
  std::optional<DFBPointerOwner> writePointerOwner;
  std::optional<DFBPointerOwner> readPointerOwner;
  SmallVector<DFBTransactionRun, 0> terminalTransactionRuns;
  std::optional<DFBPointerOwner> terminalWritePointerOwner;
  std::optional<DFBPointerOwner> terminalReadPointerOwner;
  bool terminalStateCanonical = false;
  SmallVector<DFBLifecycleEpoch, 0> resetEpochs;
  DFBQuiescenceProof quiescence;
};

/// Allocation-report data omitted from normal liveness analysis.
struct DFBLogicalLifecycleDiagnostics {
  SmallVector<DFBPerNodeLifetimeDiagnostics, 0> nodeLifetimeDiagnostics;
  SmallVector<DFBPerNodeLifetimeDiagnostics, 0> possibleNodeLifetimeDiagnostics;
};

/// Immutable protocol and per-node lifetime facts for one logical DFB.
struct DFBLogicalLifecycle {
  int64_t logicalId = 0;
  Type type;
  TensorBackingAttr tensorBacking;
  DFBAllocationGroupAttr allocationGroup;
  bool compilerCreated = false;
  SmallVector<BindCBOp> declarations;
  SmallVector<DFBAccessOccurrence> accesses;
  LaunchNodeDomain launchDomain;
  SmallVector<DFBPerNodeLifetime, 0> nodeLifetimes;
  SmallVector<DFBPerNodeLifetime, 0> possibleNodeLifetimes;
  std::unique_ptr<DFBLogicalLifecycleDiagnostics> allocationDiagnostics;
  bool bounded = false;
  bool conditionallyBounded = false;

  /// Returns the lifetime for `node`, or null when the DFB is inactive there.
  const DFBPerNodeLifetime *findNodeLifetime(LaunchNodeCoord node) const;

  /// Returns the possible-domain lifetime for `node`, or null when absent.
  const DFBPerNodeLifetime *
  findPossibleNodeLifetime(LaunchNodeCoord node) const;
};

/// Builds per-node cross-kernel happens-before and DFB quiescence facts.
class DFBConcurrentKernelLivenessAnalysis {
public:
  DFBConcurrentKernelLivenessAnalysis(Operation *operation,
                                      AnalysisManager &analysisManager);

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &analyses) {
    return !analyses.isPreserved<DFBConcurrentKernelLivenessAnalysis>() ||
           !analyses.isPreserved<DFBLogicalIdentityAnalysis>();
  }

  bool succeeded() const { return errorMessage.empty(); }
  StringRef getErrorMessage() const { return errorMessage; }
  Operation *getErrorOperation() const { return errorOperation; }

  ArrayRef<DFBLogicalLifecycle> getLogicalDFBLifecycles() const {
    return logicalDFBs;
  }

  ArrayRef<LaunchNodeCoord> getLaunchNodes() const { return launchNodes; }

  ArrayRef<DFBResetAllocationConflict> getResetAllocationConflicts() const {
    return resetAllocationConflicts;
  }

  /// Returns true when one indexed lifetime ends before another on `node`.
  bool isOrderedBefore(unsigned beforeIndex, unsigned afterIndex,
                       LaunchNodeCoord node) const;

  /// Returns ordering proved while treating unknown domains as possible.
  bool isConditionallyOrderedBefore(unsigned beforeIndex, unsigned afterIndex,
                                    LaunchNodeCoord node) const;

  /// Returns true when access events prove a reachability cycle.
  bool hasInconsistentOrder(unsigned lhsIndex, unsigned rhsIndex,
                            LaunchNodeCoord node) const;

  /// Returns inconsistent order while treating unknown domains as possible.
  bool hasConditionallyInconsistentOrder(unsigned lhsIndex, unsigned rhsIndex,
                                         LaunchNodeCoord node) const;

private:
  void analyze(Operation *operation,
               const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis);

  SmallVector<DFBLogicalLifecycle, 0> logicalDFBs;
  SmallVector<LaunchNodeCoord> launchNodes;
  SmallVector<SmallVector<llvm::BitVector>> orderedBeforeByNode;
  SmallVector<SmallVector<llvm::BitVector>> conditionallyOrderedBeforeByNode;
  SmallVector<DFBResetAllocationConflict> resetAllocationConflicts;
  SmallVector<SmallVector<llvm::BitVector>> inconsistentOrderByNode;
  SmallVector<SmallVector<llvm::BitVector>>
      conditionallyInconsistentOrderByNode;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
