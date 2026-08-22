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
  /// False when counterfactual analysis proves that no access executes.
  bool mayBeActive = true;

  /// Execution-count row for every access considered on the launch node.
  SmallVector<DFBDiagnosticAccessOccurrence> occurrences;

  /// Every access whose entry belongs to the minimal lifetime frontier.
  SmallVector<unsigned> earliestAccessOccurrenceIndices;

  /// Accesses whose completion defines the retained terminal frontier.
  SmallVector<unsigned> terminalAccessOccurrenceIndices;

  bool operator==(const DFBPerNodeLifetimeDiagnostics &rhs) const {
    return mayBeActive == rhs.mayBeActive && occurrences == rhs.occurrences &&
           earliestAccessOccurrenceIndices ==
               rhs.earliestAccessOccurrenceIndices &&
           terminalAccessOccurrenceIndices ==
               rhs.terminalAccessOccurrenceIndices;
  }
};

/// Immutable lifetime and hardware-state facts for one launched node.
struct DFBPerNodeLifetime {
  LaunchNodeCoord node;

  /// Minimal access-entry event IDs under the proved happens-before relation.
  SmallVector<unsigned> earliestEntryEvents;

  /// Completion event IDs after which the DFB is quiescent.
  SmallVector<unsigned> terminalCompletionEvents;

  /// One tile count per transaction tuple, paired by occurrence order.
  SmallVector<int64_t> transactionTileCounts;
  std::optional<DFBPointerOwner> writePointerOwner;
  std::optional<DFBPointerOwner> readPointerOwner;
  DFBQuiescenceProof quiescence;
};

/// Allocation-report data omitted from normal liveness analysis.
struct DFBLogicalLifecycleDiagnostics {
  SmallVector<DFBPerNodeLifetime, 0> counterfactualNodeLifetimes;
  SmallVector<DFBPerNodeLifetimeDiagnostics, 0> nodeLifetimeDiagnostics;
  SmallVector<DFBPerNodeLifetimeDiagnostics, 0>
      counterfactualNodeLifetimeDiagnostics;
};

/// Immutable protocol and per-node lifetime facts for one logical DFB.
struct DFBLogicalLifecycle {
  int64_t logicalId = 0;
  Type type;
  TensorBackingAttr tensorBacking;
  bool compilerCreated = false;
  SmallVector<BindCBOp> declarations;
  SmallVector<DFBAccessOccurrence> accesses;
  LaunchNodeDomain launchDomain;
  SmallVector<DFBPerNodeLifetime, 0> nodeLifetimes;
  std::unique_ptr<DFBLogicalLifecycleDiagnostics> allocationDiagnostics;
  bool bounded = false;

  /// Returns the lifetime for `node`, or null when the DFB is inactive there.
  const DFBPerNodeLifetime *findNodeLifetime(LaunchNodeCoord node) const;
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

  /// Returns true when one indexed lifetime ends before another on `node`.
  bool isOrderedBefore(unsigned beforeIndex, unsigned afterIndex,
                       LaunchNodeCoord node) const;

private:
  void analyze(Operation *operation,
               const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis);

  SmallVector<DFBLogicalLifecycle, 0> logicalDFBs;
  SmallVector<LaunchNodeCoord> launchNodes;
  SmallVector<SmallVector<llvm::BitVector>> orderedBeforeByNode;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
