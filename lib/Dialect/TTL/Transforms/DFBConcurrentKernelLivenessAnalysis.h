// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H

//===----------------------------------------------------------------------===//
// Concurrent Kernel DFB Liveness Analysis
//===----------------------------------------------------------------------===//
//
// This analysis constructs the cross-kernel event graph and derives the
// lifetime facts needed by physical DFB allocation. It does not modify IR or
// select physical indices.
//
//===----------------------------------------------------------------------===//

#include "DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

/// Immutable protocol and lifetime facts for one logical DFB.
struct DFBLogicalLifecycle {
  int64_t logicalId = 0;
  Type type;
  func::FuncOp producerKernel;
  func::FuncOp consumerKernel;
  bool compilerCreated = false;

  SmallVector<BindCBOp> declarations;
  SmallVector<Operation *> reserves;
  SmallVector<Operation *> pushes;
  SmallVector<Operation *> waits;
  SmallVector<Operation *> pops;
  SmallVector<Operation *> runtimeUses;

  std::optional<int64_t> transactionTileCount;
  SmallVector<unsigned> earliestEvents;
  SmallVector<unsigned> terminalEvents;
  bool bounded = false;
};

/// Builds cross-kernel happens-before facts for logical DFB lifetimes.
///
/// Each top-level kernel operation receives entry and completion events.
/// Program order and matched push-to-wait protocol edges establish the event
/// relation. The analysis supports any number of kernel sequences.
class DFBConcurrentKernelLivenessAnalysis {
public:
  DFBConcurrentKernelLivenessAnalysis(Operation *operation,
                                      AnalysisManager &analysisManager);

  /// Invalidates the result when its logical-identity dependency is invalid.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &analyses) {
    return !analyses.isPreserved<DFBConcurrentKernelLivenessAnalysis>() ||
           !analyses.isPreserved<DFBLogicalIdentityAnalysis>();
  }

  /// Returns true when the event and protocol facts satisfy input invariants.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed analysis.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the analysis diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns protocol and lifetime facts in module declaration order.
  ArrayRef<DFBLogicalLifecycle> getLogicalDFBLifecycles() const {
    return logicalDFBs;
  }

  /// Returns true when one indexed lifecycle ends before the other begins.
  ///
  /// Both indices refer to the array returned by `getLogicalDFBLifecycles()`.
  bool isOrderedBefore(unsigned beforeIndex, unsigned afterIndex) const;

private:
  void analyze(Operation *operation,
               const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis);

  SmallVector<DFBLogicalLifecycle, 0> logicalDFBs;
  SmallVector<llvm::BitVector> orderedBefore;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
