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

/// Protocol or opaque storage effect performed by one DFB access.
enum class DFBProtocolEffect { Reserve, Push, Wait, Pop, OpaqueAccess };

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
  RepeatedProtocolEffect,
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
  DFBProtocolEffect effect = DFBProtocolEffect::OpaqueAccess;
  LaunchNodeDomain launchDomain;
  Operation *unanalyzableDomainOperation = nullptr;
};

/// Immutable lifetime and hardware-state facts for one launched node.
struct DFBPerNodeLifetime {
  LaunchNodeCoord node;
  SmallVector<unsigned> occurrenceIndices;
  SmallVector<Operation *> earliestOperations;
  SmallVector<Operation *> terminalOperations;
  std::optional<int64_t> transactionTileCount;
  std::optional<DFBPointerOwner> writePointerOwner;
  std::optional<DFBPointerOwner> readPointerOwner;
  DFBQuiescenceProof quiescence;
};

/// Immutable protocol and per-node lifetime facts for one logical DFB.
struct DFBLogicalLifecycle {
  int64_t logicalId = 0;
  Type type;
  bool compilerCreated = false;
  SmallVector<BindCBOp> declarations;
  SmallVector<DFBAccessOccurrence> accesses;
  LaunchNodeDomain launchDomain;
  SmallVector<DFBPerNodeLifetime, 0> nodeLifetimes;
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
