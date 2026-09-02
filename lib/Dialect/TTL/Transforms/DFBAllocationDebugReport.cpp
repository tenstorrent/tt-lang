// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAllocationDebugReport.h"

#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <tuple>

namespace mlir::tt::ttl {

namespace {

static llvm::StringRef getProtocolEffectName(DFBProtocolEffectKind effect) {
  switch (effect) {
  case DFBProtocolEffectKind::Reserve:
    return "reserve";
  case DFBProtocolEffectKind::Push:
    return "push";
  case DFBProtocolEffectKind::Wait:
    return "wait";
  case DFBProtocolEffectKind::Pop:
    return "pop";
  }
  llvm_unreachable("unknown DFB protocol effect");
}

static llvm::StringRef
getNonTransactionalAccessName(DFBNonTransactionalAccessKind access) {
  switch (access) {
  case DFBNonTransactionalAccessKind::Inspect:
    return "inspect";
  }
  llvm_unreachable("unknown DFB non-transactional access");
}

static llvm::StringRef
getLifecycleCompletionName(DFBLifecycleCompletionFailureReason failure) {
  switch (failure) {
  case DFBLifecycleCompletionFailureReason::None:
    return "complete";
  case DFBLifecycleCompletionFailureReason::MissingProtocolEffect:
    return "missing-protocol-effect";
  case DFBLifecycleCompletionFailureReason::UnsupportedControlFlow:
    return "unsupported-control-flow";
  case DFBLifecycleCompletionFailureReason::MismatchedTransaction:
    return "mismatched-transaction";
  case DFBLifecycleCompletionFailureReason::IncompleteUseOrder:
    return "incomplete-use-order";
  case DFBLifecycleCompletionFailureReason::UnknownPointerOwner:
    return "unknown-pointer-owner";
  }
  llvm_unreachable("unknown DFB lifecycle-completion failure");
}

static llvm::StringRef getPointerProcessorName(DFBPointerProcessor processor) {
  switch (processor) {
  case DFBPointerProcessor::Noc0:
    return "noc0";
  case DFBPointerProcessor::Noc1:
    return "noc1";
  case DFBPointerProcessor::Pack:
    return "pack";
  case DFBPointerProcessor::Unpack:
    return "unpack";
  }
  llvm_unreachable("unknown DFB pointer processor");
}

static llvm::StringRef getPointerDirectionName(DFBPointerDirection direction) {
  switch (direction) {
  case DFBPointerDirection::Read:
    return "read";
  case DFBPointerDirection::Write:
    return "write";
  }
  llvm_unreachable("unknown DFB pointer direction");
}

static void printNode(llvm::raw_ostream &output, LaunchNodeCoord node) {
  output << '(' << node.x << ',' << node.y << ')';
}

static void printDomain(llvm::raw_ostream &output,
                        const LaunchNodeDomain &domain) {
  if (!domain.known) {
    output << "unknown";
    return;
  }
  output << '{';
  llvm::interleaveComma(domain.nodes, output,
                        [&](LaunchNodeCoord node) { printNode(output, node); });
  output << '}';
}

static void printOperation(llvm::raw_ostream &output, Operation *operation) {
  if (!operation) {
    output << "none";
    return;
  }
  output << operation->getName().getStringRef();
  if (func::FuncOp kernel = operation->getParentOfType<func::FuncOp>()) {
    output << " kernel=@" << kernel.getSymName();
    if (Attribute thread = kernel->getAttr(kKernelThreadAttrName)) {
      output << " thread=";
      thread.print(output);
    }
    if (Attribute logicalKernel = kernel->getAttr(kLogicalKernelAttrName)) {
      output << " logical_kernel=";
      logicalKernel.print(output);
    }
    if (Attribute nocIndex = kernel->getAttr(kNocIndexAttrName)) {
      output << " noc_index=";
      nocIndex.print(output);
    }
  }
  output << " loc=";
  operation->getLoc().print(output);
}

static void printPointerOwner(llvm::raw_ostream &output,
                              const std::optional<DFBPointerOwner> &owner) {
  if (!owner) {
    output << "unknown";
    return;
  }
  printNode(output, owner->node);
  output << ':' << getPointerProcessorName(owner->processor) << ':'
         << getPointerDirectionName(owner->direction);
}

template <typename Range>
static void printValues(llvm::raw_ostream &output, const Range &values) {
  output << '[';
  llvm::interleaveComma(values, output);
  output << ']';
}

static void printAccesses(llvm::raw_ostream &output,
                          const DFBLogicalLifecycle &logicalDFB) {
  for (auto indexedAccess : llvm::enumerate(logicalDFB.accesses)) {
    const DFBAccessOccurrence &access = indexedAccess.value();
    output << "  access " << indexedAccess.index() << " effect=";
    if (const DFBProtocolEffectKind *protocolEffect =
            access.getProtocolEffect()) {
      output << getProtocolEffectName(*protocolEffect);
    } else {
      output << "none";
    }
    if (const DFBNonTransactionalAccessKind *nonTransactionalAccess =
            access.getNonTransactionalAccess()) {
      output << " non_transactional="
             << getNonTransactionalAccessName(*nonTransactionalAccess);
    }
    output << " tiles=" << access.numTiles
           << " sequence=" << access.sequenceIndex;
    if (access.opaqueExternalAccess) {
      output << " opaque_external=1";
    }
    output << " domain=";
    printDomain(output, access.launchDomain);
    output << " operation=";
    printOperation(output, access.operation);
    if (access.unanalyzableDomainOperation) {
      output << " unresolved_at=";
      printOperation(output, access.unanalyzableDomainOperation);
    }
    output << '\n';
  }
}

static void printOccurrences(llvm::raw_ostream &output,
                             const DFBPerNodeLifetimeDiagnostics &diagnostics) {
  output << '[';
  llvm::interleaveComma(diagnostics.occurrences, output,
                        [&](const DFBDiagnosticAccessOccurrence &occurrence) {
                          output << occurrence.occurrenceIndex << ':';
                          if (occurrence.exactExecutionCount) {
                            output << *occurrence.exactExecutionCount;
                          } else {
                            output << "unresolved";
                          }
                        });
  output << ']';
}

static void printTransactionRuns(llvm::raw_ostream &output,
                                 ArrayRef<DFBTransactionRun> transactionRuns) {
  constexpr std::uint64_t maxExpandedTransactions = 64;
  std::uint64_t remainingExpandedTransactions = maxExpandedTransactions;
  bool expandTransactions = true;
  for (const DFBTransactionRun &run : transactionRuns) {
    if (run.executionCount > remainingExpandedTransactions) {
      expandTransactions = false;
      break;
    }
    remainingExpandedTransactions -= run.executionCount;
  }

  output << '[';
  bool first = true;
  for (const DFBTransactionRun &run : transactionRuns) {
    if (!expandTransactions) {
      if (!first) {
        output << ", ";
      }
      first = false;
      output << "run(count=" << run.executionCount
             << ",tiles=" << run.tilesPerExecution << ')';
      continue;
    }
    for (std::uint64_t transaction = 0; transaction < run.executionCount;
         ++transaction) {
      if (!first) {
        output << ", ";
      }
      first = false;
      output << run.tilesPerExecution;
    }
  }
  output << ']';
}

static void printTransactions(llvm::raw_ostream &output,
                              const DFBPerNodeLifetime &lifetime) {
  printTransactionRuns(output, lifetime.transactionRuns);
  if (lifetime.writeCursorRuns != lifetime.transactionRuns ||
      lifetime.readCursorRuns != lifetime.transactionRuns) {
    output << " write_cursor_runs=";
    printTransactionRuns(output, lifetime.writeCursorRuns);
    output << " read_cursor_runs=";
    printTransactionRuns(output, lifetime.readCursorRuns);
  }
}

static bool resetCompletesOpaqueAccess(const DFBLogicalLifecycle &logicalDFB,
                                       const DFBLifecycleEpoch &epoch) {
  if (!epoch.terminalResetOrdinal) {
    return false;
  }
  return llvm::any_of(epoch.accessOccurrenceIndices, [&](unsigned accessIndex) {
    assert(accessIndex < logicalDFB.accesses.size());
    return logicalDFB.accesses[accessIndex].opaqueExternalAccess;
  });
}

static bool
hasOpaqueAccessCompletedByReset(const DFBLogicalLifecycle &logicalDFB,
                                const DFBPerNodeLifetime &lifetime) {
  return llvm::any_of(lifetime.epochs, [&](const DFBLifecycleEpoch &epoch) {
    return resetCompletesOpaqueAccess(logicalDFB, epoch);
  });
}

static void printLifecycleEpochs(llvm::raw_ostream &output,
                                 const DFBLogicalLifecycle &logicalDFB,
                                 const DFBPerNodeLifetime &lifetime) {
  output << '[';
  llvm::interleaveComma(
      lifetime.epochs, output, [&](const DFBLifecycleEpoch &epoch) {
        output << '{';
        if (epoch.executionCount > 1) {
          output << "executions=" << epoch.executionCount << ',';
        }
        output << "accesses=";
        printValues(output, epoch.accessOccurrenceIndices);
        output << ",transactions=";
        printTransactionRuns(output, epoch.transactionRuns);
        if (epoch.writeCursorRuns != epoch.transactionRuns ||
            epoch.readCursorRuns != epoch.transactionRuns) {
          output << ",write_cursor_runs=";
          printTransactionRuns(output, epoch.writeCursorRuns);
          output << ",read_cursor_runs=";
          printTransactionRuns(output, epoch.readCursorRuns);
        }
        output << ",write_owner=";
        printPointerOwner(output, epoch.writePointerOwner);
        output << ",read_owner=";
        printPointerOwner(output, epoch.readPointerOwner);
        if (epoch.terminalWritePointerOwner != epoch.writePointerOwner ||
            epoch.terminalReadPointerOwner != epoch.readPointerOwner) {
          output << ",terminal_write_owner=";
          printPointerOwner(output, epoch.terminalWritePointerOwner);
          output << ",terminal_read_owner=";
          printPointerOwner(output, epoch.terminalReadPointerOwner);
        }
        output << ",entry_reconfiguration=";
        if (epoch.entryReconfigurationOrdinal) {
          output << *epoch.entryReconfigurationOrdinal;
        } else {
          output << "initial";
        }
        output << ",active_configurations=[";
        llvm::interleaveComma(
            epoch.activeConfigurationEpochs, output,
            [&](std::optional<int64_t> reconfigurationOrdinal) {
              if (reconfigurationOrdinal) {
                output << *reconfigurationOrdinal;
              } else {
                output << "initial";
              }
            });
        output << ']';
        output << ",terminal_reset=";
        if (epoch.terminalResetOrdinal) {
          output << *epoch.terminalResetOrdinal;
        } else {
          output << "none";
        }
        if (resetCompletesOpaqueAccess(logicalDFB, epoch)) {
          output << ",opaque_protocol_reset=1";
        }
        if (epoch.inspectionOnly) {
          output << ",inspection_only=1";
        }
        output << ",terminal_reconfiguration=";
        if (epoch.terminalReconfigurationOrdinal) {
          output << *epoch.terminalReconfigurationOrdinal;
        } else {
          output << "none";
        }
        output << ",terminal_state="
               << (epoch.terminalStateCanonical ? "canonical" : "protocol")
               << '}';
      });
  output << ']';
}

static bool hasEqualLifecycleEpochs(ArrayRef<DFBLifecycleEpoch> lhs,
                                    ArrayRef<DFBLifecycleEpoch> rhs) {
  return llvm::equal(
      lhs, rhs,
      [](const DFBLifecycleEpoch &lhsEpoch, const DFBLifecycleEpoch &rhsEpoch) {
        return lhsEpoch.executionCount == rhsEpoch.executionCount &&
               lhsEpoch.accessOccurrenceIndices ==
                   rhsEpoch.accessOccurrenceIndices &&
               lhsEpoch.transactionRuns == rhsEpoch.transactionRuns &&
               lhsEpoch.writeCursorRuns == rhsEpoch.writeCursorRuns &&
               lhsEpoch.readCursorRuns == rhsEpoch.readCursorRuns &&
               lhsEpoch.writePointerOwner == rhsEpoch.writePointerOwner &&
               lhsEpoch.readPointerOwner == rhsEpoch.readPointerOwner &&
               lhsEpoch.terminalWritePointerOwner ==
                   rhsEpoch.terminalWritePointerOwner &&
               lhsEpoch.terminalReadPointerOwner ==
                   rhsEpoch.terminalReadPointerOwner &&
               lhsEpoch.activeConfigurationEpochs ==
                   rhsEpoch.activeConfigurationEpochs &&
               lhsEpoch.entryReconfigurationOrdinal ==
                   rhsEpoch.entryReconfigurationOrdinal &&
               lhsEpoch.terminalResetOrdinal == rhsEpoch.terminalResetOrdinal &&
               lhsEpoch.terminalReconfigurationOrdinal ==
                   rhsEpoch.terminalReconfigurationOrdinal &&
               lhsEpoch.inspectionOnly == rhsEpoch.inspectionOnly &&
               lhsEpoch.terminalStateCanonical ==
                   rhsEpoch.terminalStateCanonical &&
               lhsEpoch.completionProof.failure ==
                   rhsEpoch.completionProof.failure &&
               lhsEpoch.completionProof.evidence ==
                   rhsEpoch.completionProof.evidence;
      });
}

static bool
hasEqualPossibleFacts(const DFBPerNodeLifetime &lhs,
                      const DFBPerNodeLifetimeDiagnostics &lhsDiagnostics,
                      const DFBPerNodeLifetime &rhs,
                      const DFBPerNodeLifetimeDiagnostics &rhsDiagnostics) {
  if (lhs.completionProof.failure != rhs.completionProof.failure ||
      lhs.completionProof.evidence != rhs.completionProof.evidence ||
      lhs.mayBeActive != rhs.mayBeActive ||
      lhs.conditionalExecutionProven != rhs.conditionalExecutionProven ||
      lhs.transactionRuns != rhs.transactionRuns ||
      lhs.writeCursorRuns != rhs.writeCursorRuns ||
      lhs.readCursorRuns != rhs.readCursorRuns ||
      lhs.writePointerOwner != rhs.writePointerOwner ||
      lhs.readPointerOwner != rhs.readPointerOwner ||
      lhs.terminalTransactionRuns != rhs.terminalTransactionRuns ||
      lhs.terminalWriteCursorRuns != rhs.terminalWriteCursorRuns ||
      lhs.terminalReadCursorRuns != rhs.terminalReadCursorRuns ||
      lhs.terminalWritePointerOwner != rhs.terminalWritePointerOwner ||
      lhs.terminalReadPointerOwner != rhs.terminalReadPointerOwner ||
      lhs.inspectionOnly != rhs.inspectionOnly ||
      lhs.terminalStateCanonical != rhs.terminalStateCanonical ||
      !hasEqualLifecycleEpochs(lhs.epochs, rhs.epochs) ||
      !(lhsDiagnostics == rhsDiagnostics)) {
    return false;
  }
  return true;
}

static void
printLifetimeFacts(llvm::raw_ostream &output,
                   const DFBLogicalLifecycle &logicalDFB,
                   const DFBPerNodeLifetime &lifetime,
                   const DFBPerNodeLifetimeDiagnostics &diagnostics) {
  output << " evidence=";
  printOperation(output, lifetime.completionProof.evidence);
  output << " occurrences=";
  printOccurrences(output, diagnostics);
  output << " transactions=";
  printTransactions(output, lifetime);
  output << " write_owner=";
  printPointerOwner(output, lifetime.writePointerOwner);
  output << " read_owner=";
  printPointerOwner(output, lifetime.readPointerOwner);
  output << " earliest_accesses=";
  printValues(output, diagnostics.earliestAccessOccurrenceIndices);
  output << " terminal_accesses=";
  printValues(output, diagnostics.terminalAccessOccurrenceIndices);
  if (!lifetime.epochs.empty()) {
    output << " epochs=";
    printLifecycleEpochs(output, logicalDFB, lifetime);
  }
}

struct LifetimeWithDiagnostics {
  const DFBPerNodeLifetime *lifetime = nullptr;
  const DFBPerNodeLifetimeDiagnostics *diagnostics = nullptr;
};

struct PossibleLifetimeGroup {
  LifetimeWithDiagnostics representative;
  SmallVector<LaunchNodeCoord> nodes;
};

static void printPossibleLifetimes(
    llvm::raw_ostream &output, const DFBLogicalLifecycle &logicalDFB,
    const DFBLogicalLifecycleDiagnostics &allocationDiagnostics) {
  assert(logicalDFB.possibleNodeLifetimes.size() ==
             allocationDiagnostics.possibleNodeLifetimeDiagnostics.size() &&
         "possible lifetimes must have allocation-report data");
  SmallVector<PossibleLifetimeGroup> groups;
  for (auto lifetimeAndDiagnostics :
       llvm::zip_equal(logicalDFB.possibleNodeLifetimes,
                       allocationDiagnostics.possibleNodeLifetimeDiagnostics)) {
    const DFBPerNodeLifetime &lifetime = std::get<0>(lifetimeAndDiagnostics);
    const DFBPerNodeLifetimeDiagnostics &diagnostics =
        std::get<1>(lifetimeAndDiagnostics);
    auto groupIt = llvm::find_if(groups, [&](const auto &group) {
      return hasEqualPossibleFacts(*group.representative.lifetime,
                                   *group.representative.diagnostics, lifetime,
                                   diagnostics);
    });
    if (groupIt == groups.end()) {
      groups.push_back({{&lifetime, &diagnostics}, {lifetime.node}});
    } else {
      groupIt->nodes.push_back(lifetime.node);
    }
  }

  for (const PossibleLifetimeGroup &group : groups) {
    const DFBPerNodeLifetime &lifetime = *group.representative.lifetime;
    const DFBPerNodeLifetimeDiagnostics &diagnostics =
        *group.representative.diagnostics;
    output << "  possible_nodes lifecycle_completion="
           << getLifecycleCompletionName(lifetime.completionProof.failure)
           << " domain_assumption=unknown-possible may_be_active="
           << lifetime.mayBeActive
           << " conditional_execution=" << lifetime.conditionalExecutionProven;
    if (hasOpaqueAccessCompletedByReset(logicalDFB, lifetime)) {
      output << " opaque_protocol_reset=1";
    }
    if (lifetime.inspectionOnly) {
      output << " inspection_only=1";
    }
    output << " node_count=" << group.nodes.size();
    if (group.nodes.size() <= 8) {
      output << " nodes={";
      llvm::interleaveComma(group.nodes, output, [&](LaunchNodeCoord node) {
        printNode(output, node);
      });
      output << '}';
    } else {
      output << " exemplar=";
      printNode(output, lifetime.node);
    }
    printLifetimeFacts(output, logicalDFB, lifetime, diagnostics);
    output << '\n';
  }
}

static void printNodeLifetimes(llvm::raw_ostream &output,
                               const DFBLogicalLifecycle &logicalDFB) {
  if (!logicalDFB.allocationDiagnostics) {
    assert(logicalDFB.nodeLifetimes.empty() &&
           "authoritative lifetimes require completed analysis");
    return;
  }
  const DFBLogicalLifecycleDiagnostics &allocationDiagnostics =
      *logicalDFB.allocationDiagnostics;
  assert(logicalDFB.nodeLifetimes.size() ==
             allocationDiagnostics.nodeLifetimeDiagnostics.size() &&
         "exact lifetimes must have allocation-report data");
  for (auto [lifetime, diagnostics] :
       llvm::zip_equal(logicalDFB.nodeLifetimes,
                       allocationDiagnostics.nodeLifetimeDiagnostics)) {
    output << "  node ";
    printNode(output, lifetime.node);
    output << " lifecycle_completion="
           << getLifecycleCompletionName(lifetime.completionProof.failure)
           << " domain_assumption=exact conditional_execution="
           << lifetime.conditionalExecutionProven;
    if (hasOpaqueAccessCompletedByReset(logicalDFB, lifetime)) {
      output << " opaque_protocol_reset=1";
    }
    if (lifetime.inspectionOnly) {
      output << " inspection_only=1";
    }
    printLifetimeFacts(output, logicalDFB, lifetime, diagnostics);
    output << '\n';
  }
  printPossibleLifetimes(output, logicalDFB, allocationDiagnostics);
}

static void printConflictEvidence(llvm::raw_ostream &output,
                                  const DFBPhysicalConflictModel &model) {
  for (const DFBConflictEvidence &evidence : model.getEvidence()) {
    output << "DFB conflict lhs=" << evidence.lhsLogicalId
           << " rhs=" << evidence.rhsLogicalId
           << " reason=" << getDFBConflictReasonName(evidence.reason)
           << " node=";
    if (evidence.node) {
      printNode(output, *evidence.node);
    } else {
      output << "none";
    }
    output << " lhs_operation=";
    printOperation(output, evidence.lhsOperation);
    output << " rhs_operation=";
    printOperation(output, evidence.rhsOperation);
    output << '\n';
  }
}

} // namespace

void printDFBAllocationDebugReport(
    llvm::raw_ostream &output,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflictModel) {
  output << "DFB allocation liveness report\n";
  for (const DFBLogicalLifecycle &logicalDFB :
       liveness.getLogicalDFBLifecycles()) {
    output << "DFB logical_id=" << logicalDFB.logicalId
           << " bounded=" << logicalDFB.bounded
           << " compiler_created=" << logicalDFB.compilerCreated
           << " conditionally_bounded=" << logicalDFB.conditionallyBounded;
    if (logicalDFB.hasOpaqueExternalAccess) {
      output << " opaque_external_access=1";
    }
    output << " access_completion_proven=" << logicalDFB.accessCompletionProven
           << " allocation_group=";
    if (logicalDFB.allocationGroup) {
      Attribute allocationGroup = logicalDFB.allocationGroup;
      allocationGroup.print(output);
    } else {
      output << "none";
    }
    output << " type=";
    logicalDFB.type.print(output);
    output << " tensor_backing=";
    if (logicalDFB.tensorBacking) {
      Attribute tensorBacking = logicalDFB.tensorBacking;
      tensorBacking.print(output);
    } else {
      output << "none";
    }
    output << " domain=";
    printDomain(output, logicalDFB.launchDomain);
    output << '\n';
    printAccesses(output, logicalDFB);
    printNodeLifetimes(output, logicalDFB);
  }
  printConflictEvidence(output, conflictModel);
  output << "DFB allocation liveness report end\n";
}

void printDFBStorageConflictDebugReport(
    llvm::raw_ostream &output,
    const DFBPhysicalConflictModel &storageConflictModel) {
  output << "DFB storage conflict report\n";
  printConflictEvidence(output, storageConflictModel);
  output << "DFB storage conflict report end\n";
}

} // namespace mlir::tt::ttl
