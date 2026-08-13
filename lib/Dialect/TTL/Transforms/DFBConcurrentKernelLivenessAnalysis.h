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

/// Immutable occurrence of one logical DFB access.
struct DFBAccessOccurrence {
  Operation *operation = nullptr;
  std::optional<DFBProtocolEffectKind> protocolEffect;
  int64_t numTiles = 0;
  unsigned sequenceIndex = 0;
  LaunchNodeDomain launchDomain;
  Operation *unanalyzableDomainOperation = nullptr;
};

/// Execution count retained for one access in the debug report.
struct DFBPerNodeAccessOccurrence {
  unsigned occurrenceIndex = 0;
  std::optional<std::uint64_t> exactExecutionCount;
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
  SmallVector<DFBTransactionRun> transactionRuns;
  std::optional<DFBPointerOwner> writePointerOwner;
  std::optional<DFBPointerOwner> readPointerOwner;
  std::optional<int64_t> terminalResetOrdinal;
  bool terminalStateCanonical = false;
  DFBQuiescenceProof quiescence;
};

/// Immutable lifetime and hardware-state facts for one launched node.
struct DFBPerNodeLifetime {
  LaunchNodeCoord node;
  bool mayBeActive = true;
  bool conditionalExecutionProven = false;
  SmallVector<DFBPerNodeAccessOccurrence> reportedOccurrences;
  SmallVector<unsigned> earliestEntryEvents;
  SmallVector<unsigned> terminalCompletionEvents;
  SmallVector<unsigned> earliestAccessOccurrenceIndices;
  SmallVector<unsigned> terminalAccessOccurrenceIndices;
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

  /// Returns true when one indexed lifetime ends before another on `node`.
  bool isOrderedBefore(unsigned beforeIndex, unsigned afterIndex,
                       LaunchNodeCoord node) const;

  /// Returns ordering proved while treating unknown domains as possible.
  bool isConditionallyOrderedBefore(unsigned beforeIndex, unsigned afterIndex,
                                    LaunchNodeCoord node) const;

private:
  void analyze(Operation *operation,
               const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis);

  SmallVector<DFBLogicalLifecycle, 0> logicalDFBs;
  SmallVector<LaunchNodeCoord> launchNodes;
  SmallVector<SmallVector<llvm::BitVector>> orderedBeforeByNode;
  SmallVector<SmallVector<llvm::BitVector>> conditionallyOrderedBeforeByNode;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
