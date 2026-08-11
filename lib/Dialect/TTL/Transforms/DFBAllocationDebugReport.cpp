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

static llvm::StringRef getConflictReasonName(DFBConflictReason reason) {
  switch (reason) {
  case DFBConflictReason::DescriptorMismatch:
    return "descriptor-mismatch";
  case DFBConflictReason::StorageMismatch:
    return "storage-mismatch";
  case DFBConflictReason::UnknownLaunchNodeDomain:
    return "unknown-launch-node-domain";
  case DFBConflictReason::UnprovenQuiescence:
    return "unproven-quiescence";
  case DFBConflictReason::TransactionMismatch:
    return "transaction-mismatch";
  case DFBConflictReason::PointerOwnerMismatch:
    return "pointer-owner-mismatch";
  case DFBConflictReason::ConcurrentLifetime:
    return "concurrent-lifetime";
  }
  llvm_unreachable("unknown DFB conflict reason");
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

static void printNode(llvm::raw_ostream &os, LaunchNodeCoord node) {
  os << '(' << node.x << ',' << node.y << ')';
}

static void printDomain(llvm::raw_ostream &os, const LaunchNodeDomain &domain) {
  if (!domain.known) {
    os << "unknown";
    return;
  }
  os << '{';
  llvm::interleaveComma(domain.nodes, os,
                        [&](LaunchNodeCoord node) { printNode(os, node); });
  os << '}';
}

static void printOperation(llvm::raw_ostream &os, Operation *operation) {
  if (!operation) {
    os << "none";
    return;
  }
  os << operation->getName().getStringRef();
  if (func::FuncOp kernel = operation->getParentOfType<func::FuncOp>()) {
    os << " kernel=@" << kernel.getSymName();
    if (Attribute thread = kernel->getAttr(kKernelThreadAttrName)) {
      os << " thread=";
      thread.print(os);
    }
    if (Attribute logicalKernel = kernel->getAttr(kLogicalKernelAttrName)) {
      os << " logical_kernel=";
      logicalKernel.print(os);
    }
    if (Attribute nocIndex = kernel->getAttr(kNocIndexAttrName)) {
      os << " noc_index=";
      nocIndex.print(os);
    }
  }
  os << " loc=";
  operation->getLoc().print(os);
}

static void printPointerOwner(llvm::raw_ostream &os,
                              const std::optional<DFBPointerOwner> &owner) {
  if (!owner) {
    os << "unknown";
    return;
  }
  printNode(os, owner->node);
  os << ':' << getPointerProcessorName(owner->processor) << ':'
     << getPointerDirectionName(owner->direction);
}

static void printUnsignedValues(llvm::raw_ostream &os,
                                llvm::ArrayRef<unsigned> values) {
  os << '[';
  llvm::interleaveComma(values, os);
  os << ']';
}

static void printTileCounts(llvm::raw_ostream &os,
                            llvm::ArrayRef<int64_t> values) {
  os << '[';
  llvm::interleaveComma(values, os);
  os << ']';
}

static void printAccesses(llvm::raw_ostream &os,
                          const DFBLogicalLifecycle &logicalDFB) {
  for (auto indexedAccess : llvm::enumerate(logicalDFB.accesses)) {
    const DFBAccessOccurrence &access = indexedAccess.value();
    os << "  access " << indexedAccess.index() << " effect=";
    if (access.protocolEffect) {
      os << getProtocolEffectName(*access.protocolEffect);
    } else {
      os << "none";
    }
    os << " tiles=" << access.numTiles << " sequence=" << access.sequenceIndex
       << " domain=";
    printDomain(os, access.launchDomain);
    os << " operation=";
    printOperation(os, access.operation);
    if (access.unanalyzableDomainOperation) {
      os << " unresolved_at=";
      printOperation(os, access.unanalyzableDomainOperation);
    }
    os << '\n';
  }
}

static void printOccurrences(llvm::raw_ostream &os,
                             const DFBPerNodeLifetime &lifetime) {
  os << '[';
  llvm::interleaveComma(lifetime.occurrences, os,
                        [&](const DFBPerNodeAccessOccurrence &occurrence) {
                          os << occurrence.occurrenceIndex << ':';
                          if (occurrence.exactExecutionCount) {
                            os << *occurrence.exactExecutionCount;
                          } else {
                            os << "unresolved";
                          }
                        });
  os << ']';
}

static bool hasEqualPossibleDomainFacts(const DFBPerNodeLifetime &lhs,
                                        const DFBPerNodeLifetime &rhs) {
  if (lhs.quiescence.failure != rhs.quiescence.failure ||
      lhs.quiescence.evidence != rhs.quiescence.evidence ||
      lhs.mayBeActive != rhs.mayBeActive ||
      lhs.conditionalExecutionProven != rhs.conditionalExecutionProven ||
      lhs.occurrences.size() != rhs.occurrences.size() ||
      lhs.transactionTileCounts != rhs.transactionTileCounts ||
      lhs.writePointerOwner != rhs.writePointerOwner ||
      lhs.readPointerOwner != rhs.readPointerOwner) {
    return false;
  }
  return llvm::equal(lhs.occurrences, rhs.occurrences,
                     [](const DFBPerNodeAccessOccurrence &lhsOccurrence,
                        const DFBPerNodeAccessOccurrence &rhsOccurrence) {
                       return lhsOccurrence.occurrenceIndex ==
                                  rhsOccurrence.occurrenceIndex &&
                              lhsOccurrence.exactExecutionCount ==
                                  rhsOccurrence.exactExecutionCount;
                     });
}

struct PossibleDomainLifetimeGroup {
  const DFBPerNodeLifetime *representative = nullptr;
  SmallVector<LaunchNodeCoord> nodes;
};

static void printPossibleDomainLifetimes(
    llvm::raw_ostream &os,
    llvm::ArrayRef<const DFBPerNodeLifetime *> possibleDomainLifetimes) {
  SmallVector<PossibleDomainLifetimeGroup> groups;
  for (const DFBPerNodeLifetime *lifetime : possibleDomainLifetimes) {
    auto groupIt = llvm::find_if(groups, [&](const auto &group) {
      return hasEqualPossibleDomainFacts(*group.representative, *lifetime);
    });
    if (groupIt == groups.end()) {
      groups.push_back({lifetime, {lifetime->node}});
    } else {
      groupIt->nodes.push_back(lifetime->node);
    }
  }

  for (const PossibleDomainLifetimeGroup &group : groups) {
    const DFBPerNodeLifetime &lifetime = *group.representative;
    os << "  diagnostic_nodes quiescence="
       << getQuiescenceFailureName(lifetime.quiescence.failure)
       << " domain_assumption=unknown-possible may_be_active="
       << lifetime.mayBeActive
       << " conditional_execution=" << lifetime.conditionalExecutionProven
       << " node_count=" << group.nodes.size();
    if (group.nodes.size() <= 8) {
      os << " nodes={";
      llvm::interleaveComma(group.nodes, os,
                            [&](LaunchNodeCoord node) { printNode(os, node); });
      os << '}';
    } else {
      os << " exemplar=";
      printNode(os, lifetime.node);
    }
    os << " evidence=";
    printOperation(os, lifetime.quiescence.evidence);
    os << " occurrences=";
    printOccurrences(os, lifetime);
    os << " transactions=";
    printTileCounts(os, lifetime.transactionTileCounts);
    os << " write_owner=";
    printPointerOwner(os, lifetime.writePointerOwner);
    os << " read_owner=";
    printPointerOwner(os, lifetime.readPointerOwner);
    os << '\n';
  }
}

static void printNodeLifetimes(llvm::raw_ostream &os,
                               const DFBLogicalLifecycle &logicalDFB) {
  SmallVector<const DFBPerNodeLifetime *> possibleDomainLifetimes;
  for (const DFBPerNodeLifetime &lifetime : logicalDFB.nodeLifetimes) {
    if (lifetime.includesUnknownDomains) {
      possibleDomainLifetimes.push_back(&lifetime);
      continue;
    }
    os << "  node ";
    printNode(os, lifetime.node);
    os << " quiescence="
       << getQuiescenceFailureName(lifetime.quiescence.failure)
       << " domain_assumption=exact"
       << " may_be_active=" << lifetime.mayBeActive
       << " conditional_execution=" << lifetime.conditionalExecutionProven
       << " evidence=";
    printOperation(os, lifetime.quiescence.evidence);
    os << " occurrences=";
    printOccurrences(os, lifetime);
    os << " transactions=";
    printTileCounts(os, lifetime.transactionTileCounts);
    os << " write_owner=";
    printPointerOwner(os, lifetime.writePointerOwner);
    os << " read_owner=";
    printPointerOwner(os, lifetime.readPointerOwner);
    os << " earliest=";
    printUnsignedValues(os, lifetime.earliestEntryEvents);
    os << " terminal=";
    printUnsignedValues(os, lifetime.terminalCompletionEvents);
    os << '\n';
  }
  printPossibleDomainLifetimes(os, possibleDomainLifetimes);
}

static void printConflictEvidence(llvm::raw_ostream &os,
                                  const DFBPhysicalConflictModel &model) {
  for (const DFBConflictEvidence &evidence : model.getEvidence()) {
    os << "DFB conflict lhs=" << evidence.lhsLogicalId
       << " rhs=" << evidence.rhsLogicalId
       << " reason=" << getConflictReasonName(evidence.reason) << " node=";
    if (evidence.node) {
      printNode(os, *evidence.node);
    } else {
      os << "none";
    }
    os << " lhs_operation=";
    printOperation(os, evidence.lhsOperation);
    os << " rhs_operation=";
    printOperation(os, evidence.rhsOperation);
    os << '\n';
  }
}

} // namespace

void printDFBAllocationDebugReport(
    llvm::raw_ostream &os, const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflictModel) {
  os << "DFB allocation liveness report\n";
  for (const DFBLogicalLifecycle &logicalDFB :
       liveness.getLogicalDFBLifecycles()) {
    os << "DFB logical_id=" << logicalDFB.logicalId
       << " bounded=" << logicalDFB.bounded
       << " compiler_created=" << logicalDFB.compilerCreated
       << " conditionally_bounded=" << logicalDFB.conditionallyBounded
       << " type=";
    logicalDFB.type.print(os);
    os << " tensor_backing=";
    if (logicalDFB.tensorBacking) {
      Attribute tensorBacking = logicalDFB.tensorBacking;
      tensorBacking.print(os);
    } else {
      os << "none";
    }
    os << " domain=";
    printDomain(os, logicalDFB.launchDomain);
    os << '\n';
    printAccesses(os, logicalDFB);
    printNodeLifetimes(os, logicalDFB);
  }
  printConflictEvidence(os, conflictModel);
  os << "DFB allocation liveness report end\n";
}

} // namespace mlir::tt::ttl
