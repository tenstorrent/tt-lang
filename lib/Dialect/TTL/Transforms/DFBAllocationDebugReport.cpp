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
getQuiescenceFailureName(DFBQuiescenceFailureReason failure) {
  switch (failure) {
  case DFBQuiescenceFailureReason::None:
    return "none";
  case DFBQuiescenceFailureReason::MissingProtocolEffect:
    return "missing-protocol-effect";
  case DFBQuiescenceFailureReason::UnsupportedControlFlow:
    return "unsupported-control-flow";
  case DFBQuiescenceFailureReason::MismatchedTransaction:
    return "mismatched-transaction";
  case DFBQuiescenceFailureReason::IncompleteUseOrder:
    return "incomplete-use-order";
  case DFBQuiescenceFailureReason::UnknownPointerOwner:
    return "unknown-pointer-owner";
  }
  llvm_unreachable("unknown DFB quiescence failure");
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
    if (access.protocolEffect) {
      output << getProtocolEffectName(*access.protocolEffect);
    } else {
      output << "none";
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
                             const DFBPerNodeLifetime &lifetime) {
  output << '[';
  llvm::interleaveComma(lifetime.reportedOccurrences, output,
                        [&](const DFBPerNodeAccessOccurrence &occurrence) {
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

static void printResetEpochs(llvm::raw_ostream &output,
                             const DFBPerNodeLifetime &lifetime) {
  output << '[';
  llvm::interleaveComma(
      lifetime.resetEpochs, output, [&](const DFBLifecycleEpoch &epoch) {
        output << "{accesses=";
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
        output << ",terminal_reset=";
        if (epoch.terminalResetOrdinal) {
          output << *epoch.terminalResetOrdinal;
        } else {
          output << "none";
        }
        if (epoch.resetCanonicalizedOpaqueProtocol) {
          output << ",opaque_protocol_reset=1";
        }
        output << ",terminal_state="
               << (epoch.terminalStateCanonical ? "canonical" : "protocol")
               << '}';
      });
  output << ']';
}

static bool hasEqualResetEpochs(ArrayRef<DFBLifecycleEpoch> lhs,
                                ArrayRef<DFBLifecycleEpoch> rhs) {
  return llvm::equal(
      lhs, rhs,
      [](const DFBLifecycleEpoch &lhsEpoch, const DFBLifecycleEpoch &rhsEpoch) {
        return lhsEpoch.accessOccurrenceIndices ==
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
               lhsEpoch.terminalResetOrdinal == rhsEpoch.terminalResetOrdinal &&
               lhsEpoch.resetCanonicalizedOpaqueProtocol ==
                   rhsEpoch.resetCanonicalizedOpaqueProtocol &&
               lhsEpoch.terminalStateCanonical ==
                   rhsEpoch.terminalStateCanonical &&
               lhsEpoch.quiescence.failure == rhsEpoch.quiescence.failure &&
               lhsEpoch.quiescence.evidence == rhsEpoch.quiescence.evidence;
      });
}

static bool hasEqualPossibleFacts(const DFBPerNodeLifetime &lhs,
                                  const DFBPerNodeLifetime &rhs) {
  if (lhs.quiescence.failure != rhs.quiescence.failure ||
      lhs.quiescence.evidence != rhs.quiescence.evidence ||
      lhs.mayBeActive != rhs.mayBeActive ||
      lhs.conditionalExecutionProven != rhs.conditionalExecutionProven ||
      lhs.reportedOccurrences.size() != rhs.reportedOccurrences.size() ||
      lhs.earliestAccessOccurrenceIndices !=
          rhs.earliestAccessOccurrenceIndices ||
      lhs.terminalAccessOccurrenceIndices !=
          rhs.terminalAccessOccurrenceIndices ||
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
      lhs.resetCanonicalizedOpaqueProtocol !=
          rhs.resetCanonicalizedOpaqueProtocol ||
      lhs.terminalStateCanonical != rhs.terminalStateCanonical ||
      !hasEqualResetEpochs(lhs.resetEpochs, rhs.resetEpochs)) {
    return false;
  }
  return llvm::equal(lhs.reportedOccurrences, rhs.reportedOccurrences,
                     [](const DFBPerNodeAccessOccurrence &lhsOccurrence,
                        const DFBPerNodeAccessOccurrence &rhsOccurrence) {
                       return lhsOccurrence.occurrenceIndex ==
                                  rhsOccurrence.occurrenceIndex &&
                              lhsOccurrence.exactExecutionCount ==
                                  rhsOccurrence.exactExecutionCount;
                     });
}

struct PossibleLifetimeGroup {
  const DFBPerNodeLifetime *representative = nullptr;
  SmallVector<LaunchNodeCoord> nodes;
};

static void printPossibleLifetimes(
    llvm::raw_ostream &output,
    llvm::ArrayRef<const DFBPerNodeLifetime *> possibleLifetimes) {
  SmallVector<PossibleLifetimeGroup> groups;
  for (const DFBPerNodeLifetime *lifetime : possibleLifetimes) {
    auto groupIt = llvm::find_if(groups, [&](const auto &group) {
      return hasEqualPossibleFacts(*group.representative, *lifetime);
    });
    if (groupIt == groups.end()) {
      groups.push_back({lifetime, {lifetime->node}});
    } else {
      groupIt->nodes.push_back(lifetime->node);
    }
  }

  for (const PossibleLifetimeGroup &group : groups) {
    const DFBPerNodeLifetime &lifetime = *group.representative;
    output << "  possible_nodes quiescence="
           << getQuiescenceFailureName(lifetime.quiescence.failure)
           << " domain_assumption=unknown-possible may_be_active="
           << lifetime.mayBeActive
           << " conditional_execution=" << lifetime.conditionalExecutionProven;
    if (lifetime.resetCanonicalizedOpaqueProtocol) {
      output << " opaque_protocol_reset=1";
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
    output << " evidence=";
    printOperation(output, lifetime.quiescence.evidence);
    output << " occurrences=";
    printOccurrences(output, lifetime);
    output << " transactions=";
    printTransactions(output, lifetime);
    output << " write_owner=";
    printPointerOwner(output, lifetime.writePointerOwner);
    output << " read_owner=";
    printPointerOwner(output, lifetime.readPointerOwner);
    output << " earliest_accesses=";
    printValues(output, lifetime.earliestAccessOccurrenceIndices);
    output << " terminal_accesses=";
    printValues(output, lifetime.terminalAccessOccurrenceIndices);
    if (!lifetime.resetEpochs.empty()) {
      output << " reset_epochs=";
      printResetEpochs(output, lifetime);
    }
    output << '\n';
  }
}

static void printNodeLifetimes(llvm::raw_ostream &output,
                               const DFBLogicalLifecycle &logicalDFB) {
  for (const DFBPerNodeLifetime &lifetime : logicalDFB.nodeLifetimes) {
    output << "  node ";
    printNode(output, lifetime.node);
    output << " quiescence="
           << getQuiescenceFailureName(lifetime.quiescence.failure)
           << " domain_assumption=exact conditional_execution="
           << lifetime.conditionalExecutionProven;
    if (lifetime.resetCanonicalizedOpaqueProtocol) {
      output << " opaque_protocol_reset=1";
    }
    output << " evidence=";
    printOperation(output, lifetime.quiescence.evidence);
    output << " occurrences=";
    printOccurrences(output, lifetime);
    output << " transactions=";
    printTransactions(output, lifetime);
    output << " write_owner=";
    printPointerOwner(output, lifetime.writePointerOwner);
    output << " read_owner=";
    printPointerOwner(output, lifetime.readPointerOwner);
    output << " earliest_accesses=";
    printValues(output, lifetime.earliestAccessOccurrenceIndices);
    output << " terminal_accesses=";
    printValues(output, lifetime.terminalAccessOccurrenceIndices);
    if (!lifetime.resetEpochs.empty()) {
      output << " reset_epochs=";
      printResetEpochs(output, lifetime);
    }
    output << '\n';
  }
  SmallVector<const DFBPerNodeLifetime *> possibleLifetimes;
  for (const DFBPerNodeLifetime &lifetime : logicalDFB.possibleNodeLifetimes) {
    possibleLifetimes.push_back(&lifetime);
  }
  printPossibleLifetimes(output, possibleLifetimes);
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
    output << " allocation_group=";
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

} // namespace mlir::tt::ttl
