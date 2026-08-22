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
           << " sequence=" << access.sequenceIndex << " domain=";
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

static bool
hasEqualDiagnosticFacts(const DFBPerNodeLifetime &lhs,
                        const DFBPerNodeLifetimeDiagnostics &lhsDiagnostics,
                        const DFBPerNodeLifetime &rhs,
                        const DFBPerNodeLifetimeDiagnostics &rhsDiagnostics) {
  if (lhs.quiescence.failure != rhs.quiescence.failure ||
      lhs.quiescence.evidence != rhs.quiescence.evidence ||
      lhs.transactionTileCounts != rhs.transactionTileCounts ||
      lhs.writePointerOwner != rhs.writePointerOwner ||
      lhs.readPointerOwner != rhs.readPointerOwner ||
      !(lhsDiagnostics == rhsDiagnostics)) {
    return false;
  }
  return true;
}

static void
printLifetimeFacts(llvm::raw_ostream &output,
                   const DFBPerNodeLifetime &lifetime,
                   const DFBPerNodeLifetimeDiagnostics &diagnostics) {
  output << " evidence=";
  printOperation(output, lifetime.quiescence.evidence);
  output << " occurrences=";
  printOccurrences(output, diagnostics);
  output << " transactions=";
  printValues(output, lifetime.transactionTileCounts);
  output << " write_owner=";
  printPointerOwner(output, lifetime.writePointerOwner);
  output << " read_owner=";
  printPointerOwner(output, lifetime.readPointerOwner);
  output << " earliest_accesses=";
  printValues(output, diagnostics.earliestAccessOccurrenceIndices);
  output << " terminal_accesses=";
  printValues(output, diagnostics.terminalAccessOccurrenceIndices);
}

struct LifetimeWithDiagnostics {
  const DFBPerNodeLifetime *lifetime = nullptr;
  const DFBPerNodeLifetimeDiagnostics *diagnostics = nullptr;
};

struct DiagnosticLifetimeGroup {
  LifetimeWithDiagnostics representative;
  SmallVector<LaunchNodeCoord> nodes;
};

static void printDiagnosticLifetimes(
    llvm::raw_ostream &output,
    const DFBLogicalLifecycleDiagnostics &allocationDiagnostics) {
  assert(
      allocationDiagnostics.counterfactualNodeLifetimes.size() ==
          allocationDiagnostics.counterfactualNodeLifetimeDiagnostics.size() &&
      "counterfactual lifetimes must have allocation-report data");
  SmallVector<DiagnosticLifetimeGroup> groups;
  for (auto lifetimeAndDiagnostics : llvm::zip_equal(
           allocationDiagnostics.counterfactualNodeLifetimes,
           allocationDiagnostics.counterfactualNodeLifetimeDiagnostics)) {
    const DFBPerNodeLifetime &lifetime = std::get<0>(lifetimeAndDiagnostics);
    const DFBPerNodeLifetimeDiagnostics &diagnostics =
        std::get<1>(lifetimeAndDiagnostics);
    auto groupIt = llvm::find_if(groups, [&](const auto &group) {
      return hasEqualDiagnosticFacts(*group.representative.lifetime,
                                     *group.representative.diagnostics,
                                     lifetime, diagnostics);
    });
    if (groupIt == groups.end()) {
      groups.push_back({{&lifetime, &diagnostics}, {lifetime.node}});
    } else {
      groupIt->nodes.push_back(lifetime.node);
    }
  }

  for (const DiagnosticLifetimeGroup &group : groups) {
    const DFBPerNodeLifetime &lifetime = *group.representative.lifetime;
    const DFBPerNodeLifetimeDiagnostics &diagnostics =
        *group.representative.diagnostics;
    output << "  diagnostic_nodes quiescence="
           << getQuiescenceFailureName(lifetime.quiescence.failure)
           << " domain_assumption=unknown-may-be-active may_be_active="
           << diagnostics.mayBeActive << " node_count=" << group.nodes.size();
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
    printLifetimeFacts(output, lifetime, diagnostics);
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
    output << " quiescence="
           << getQuiescenceFailureName(lifetime.quiescence.failure)
           << " domain_assumption=exact";
    printLifetimeFacts(output, lifetime, diagnostics);
    output << '\n';
  }
  printDiagnosticLifetimes(output, allocationDiagnostics);
}

static void printConflictEvidence(llvm::raw_ostream &output,
                                  const DFBPhysicalConflictModel &model) {
  for (const DFBConflictEvidence &evidence : model.getEvidence()) {
    output << "DFB conflict lhs=" << evidence.lhsLogicalId
           << " rhs=" << evidence.rhsLogicalId
           << " reason=" << getConflictReasonName(evidence.reason) << " node=";
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
           << " compiler_created=" << logicalDFB.compilerCreated << " type=";
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
