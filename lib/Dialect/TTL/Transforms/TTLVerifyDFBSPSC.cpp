// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify DFB SPSC
//===----------------------------------------------------------------------===//
//
// Rejects waits without a push and modules in which a logical dataflow buffer
// has more than one producer or consumer kernel active on the same launched
// node. Logical identity remains distinct when non-overlapping DFBs share a
// physical `cb_index`. tt-metal CBs are single-producer single-consumer at the
// API level; see `docs/development/DFBManagement.md` for the rationale.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "DFBVerification.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYDFBSPSC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// Caches one operation's launch domain because all its protocol actions execute
// on the same launched nodes.
struct ProtocolActionDomain {
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

// A kernel thread that produces or consumes a dataflow buffer.
//
// Multiple actions in the same thread are merged because SPSC is a thread-level
// property, not an operation-level property.
struct DFBParticipant {
  func::FuncOp thread;
  Operation *op = nullptr;
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

// Producers or consumers for one logical dataflow buffer.
struct DFBParticipantSet {
  llvm::SmallMapVector<func::FuncOp, DFBParticipant, 2> participants;
};

// Analysis state shared by the dataflow solver and the verifier pass.
struct ModuleState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, ProtocolActionDomain> protocolActionDomains;
};

void recordProtocolActionDomain(Operation *op, const LaunchNodeDomain &domain,
                                Operation *unanalyzableOp, ModuleState &state) {
  auto access = dyn_cast<DFBAccessOpInterface>(op);
  if (!access || access.getDFBProtocolEffects().empty()) {
    return;
  }
  state.protocolActionDomains[op] = {domain, unanalyzableOp};
}

// SPSC counts kernel threads, so repeated actions from one thread form one
// participant with the union of their launch domains.
void addParticipant(DFBParticipantSet &set, func::FuncOp thread, Operation *op,
                    const LaunchNodeDomain &domain, Operation *unanalyzableOp) {
  DFBParticipant participant{thread, op, domain, unanalyzableOp};
  auto [it, inserted] = set.participants.insert({thread, participant});
  if (inserted) {
    return;
  }
  DFBParticipant &existing = it->second;
  existing.domain = existing.domain.unionWith(domain);
  existing.op = pickEarlierBySourceLoc(existing.op, op);
  existing.unanalyzableOp =
      pickEarlierBySourceLoc(existing.unanalyzableOp, unanalyzableOp);
}

void attachCommonNotes(InFlightDiagnostic &diag, Operation *bindSite,
                       llvm::StringRef role) {
  diag.attachNote() << "tt-metal CBs are single-producer single-consumer; "
                       "allocate one DFB per "
                    << role;
  if (bindSite) {
    diag.attachNote(bindSite->getLoc()) << "dataflow buffer declared here";
  }
}

struct DFBProtocolPresence {
  bool hasAcquisitionAction = false;
  bool hasUnknownUserDFBAccess = false;
  llvm::DenseSet<int64_t> pushedDFBs;
  llvm::DenseSet<int64_t> dfbsWithOpaqueAccess;
  llvm::SmallMapVector<int64_t, Operation *, 4> firstWaitByDFB;
};

DFBProtocolPresence collectDFBProtocolPresence(ModuleOp module) {
  DFBProtocolPresence presence;
  module.walk([&](Operation *op) {
    auto access = dyn_cast<DFBAccessOpInterface>(op);
    if (!access || !getEnclosingKernelThread(op)) {
      return;
    }
    if (auto opaqueCall = dyn_cast<OpaqueCallOp>(op)) {
      SmallVector<Value> dependencies = opaqueCall.getDFBDependencyOperands();
      for (unsigned dependencyIndex :
           getOpaqueDFBDependencyIndices(opaqueCall)) {
        FailureOr<int64_t> dfbId = getDFBId(dependencies[dependencyIndex]);
        assert(succeeded(dfbId) && "DFB identities were verified");
        presence.dfbsWithOpaqueAccess.insert(*dfbId);
      }
      presence.hasUnknownUserDFBAccess |= opaqueCall.hasUnknownDFBAccess();
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      FailureOr<int64_t> dfbId = getDFBId(effect.dfb);
      assert(succeeded(dfbId) && "DFB identities were verified");
      switch (effect.kind) {
      case DFBProtocolEffectKind::Reserve:
        presence.hasAcquisitionAction = true;
        break;
      case DFBProtocolEffectKind::Push:
        presence.pushedDFBs.insert(*dfbId);
        break;
      case DFBProtocolEffectKind::Wait:
        presence.hasAcquisitionAction = true;
        presence.firstWaitByDFB.try_emplace(*dfbId, op);
        break;
      case DFBProtocolEffectKind::Pop:
        break;
      }
    }
  });
  return presence;
}

LogicalResult verifyDFBWaitsHavePushes(
    const DFBProtocolPresence &presence,
    const llvm::DenseMap<int64_t, Operation *> &bindSites) {
  bool sawError = false;
  for (auto [dfbId, waitOp] : presence.firstWaitByDFB) {
    Operation *bindSite = bindSites.lookup(dfbId);
    auto bindOp = dyn_cast_or_null<BindCBOp>(bindSite);
    bool hasPossibleExternalProducer =
        presence.dfbsWithOpaqueAccess.contains(dfbId) ||
        (presence.hasUnknownUserDFBAccess && bindOp &&
         isUserManagedDFB(bindOp.getResult()));
    if (presence.pushedDFBs.contains(dfbId) || hasPossibleExternalProducer) {
      continue;
    }
    InFlightDiagnostic diag = waitOp->emitError()
                              << "logical DFB " << dfbId
                              << " is waited on but no kernel thread pushes "
                                 "it";
    diag.attachNote()
        << "a DFB wait blocks until a matching push publishes data";
    if (bindSite) {
      diag.attachNote(bindSite->getLoc()) << "dataflow buffer declared here";
    }
    sawError = true;
  }
  return failure(sawError);
}

void emitOverlapError(int64_t logicalId, const DFBParticipant &lhs,
                      const DFBParticipant &rhs,
                      const LaunchNodeDomain &overlap, Operation *bindSite,
                      llvm::StringRef role, llvm::StringRef verbedHere) {
  InFlightDiagnostic diag = lhs.op->emitError()
                            << "logical DFB " << logicalId << " has multiple "
                            << role
                            << " kernels active on the same launched node";
  if (!overlap.nodes.empty()) {
    LaunchNodeCoord example = *overlap.nodes.begin();
    diag.attachNote() << "example overlapping node: core_x=" << example.x
                      << ", core_y=" << example.y;
  }
  diag.attachNote(rhs.op->getLoc()) << "also " << verbedHere << " here";
  attachCommonNotes(diag, bindSite, role);
}

void emitUnknownDomainError(int64_t logicalId, const DFBParticipantSet &set,
                            Operation *bindSite, llvm::StringRef role,
                            llvm::StringRef verbedHere) {
  auto unknownIt = llvm::find_if(set.participants, [](const auto &entry) {
    return !entry.second.domain.known;
  });
  assert(unknownIt != set.participants.end() &&
         "expected at least one unknown participant domain");

  const DFBParticipant &primary = unknownIt->second;
  InFlightDiagnostic diag = primary.op->emitError()
                            << "logical DFB " << logicalId << " has multiple "
                            << role
                            << " kernels, but SPSC could not be statically "
                               "proven";
  if (primary.unanalyzableOp) {
    diag.attachNote(primary.unanalyzableOp->getLoc())
        << "this expression is not statically analyzable";
  }
  for (auto &entry : set.participants) {
    const DFBParticipant &participant = entry.second;
    if (participant.op == primary.op) {
      continue;
    }
    diag.attachNote(participant.op->getLoc())
        << "also " << verbedHere << " here";
  }
  attachCommonNotes(diag, bindSite, role);
}

bool verifyParticipantSet(int64_t logicalId, const DFBParticipantSet &set,
                          Operation *bindSite, llvm::StringRef role,
                          llvm::StringRef verbedHere) {
  if (set.participants.size() <= 1) {
    return false;
  }

  for (auto lhsIt = set.participants.begin(), end = set.participants.end();
       lhsIt != end; ++lhsIt) {
    const DFBParticipant &lhs = lhsIt->second;
    for (auto rhsIt = std::next(lhsIt); rhsIt != end; ++rhsIt) {
      const DFBParticipant &rhs = rhsIt->second;
      LaunchNodeDomain overlap = lhs.domain.intersectWith(rhs.domain);
      if (overlap.known && !overlap.nodes.empty()) {
        emitOverlapError(logicalId, lhs, rhs, overlap, bindSite, role,
                         verbedHere);
        return true;
      }
    }
  }

  if (llvm::any_of(set.participants, [](const auto &entry) {
        return !entry.second.domain.known;
      })) {
    emitUnknownDomainError(logicalId, set, bindSite, role, verbedHere);
    return true;
  }
  return false;
}

struct TTLVerifyDFBSPSCPass
    : public impl::TTLVerifyDFBSPSCBase<TTLVerifyDFBSPSCPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(verifyResolvedDFBIdentities(module, getArgument()))) {
      signalPassFailure();
      return;
    }

    llvm::DenseMap<int64_t, Operation *> bindSites;
    llvm::DenseMap<int64_t, int64_t> physicalIndices;
    bool hasInconsistentIndex = false;

    // Finalization normally guarantees this mapping. The check remains here
    // because the verifier also supports direct invocation on finalized IR.
    module.walk([&](BindCBOp bindOp) {
      FailureOr<int64_t> dfbId = getDFBId(bindOp.getResult());
      assert(succeeded(dfbId) && "DFB identities were verified");
      int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
      bindSites.try_emplace(*dfbId, bindOp);
      auto [indexIt, inserted] = physicalIndices.try_emplace(*dfbId, cbIndex);
      if (!inserted && indexIt->second != cbIndex) {
        bindOp.emitOpError() << "logical DFB " << *dfbId
                             << " has inconsistent finalized cb_index values "
                             << indexIt->second << " and " << cbIndex;
        hasInconsistentIndex = true;
      }
    });

    if (hasInconsistentIndex) {
      signalPassFailure();
      return;
    }

    DFBProtocolPresence protocolPresence = collectDFBProtocolPresence(module);
    if (!protocolPresence.hasAcquisitionAction) {
      return;
    }

    ModuleState state;
    state.initialize(module);
    if (!state.hasLaunchGrid) {
      module.emitError()
          << "ttl-verify-dfb-spsc requires a `ttl.launch_grid` module "
             "attribute (an i64 array of length 2 with positive entries) "
             "when verifying DFB acquire actions";
      signalPassFailure();
      return;
    }

    if (failed(verifyDFBWaitsHavePushes(protocolPresence, bindSites))) {
      signalPassFailure();
      return;
    }

    if (isDFBProtocolDomainVerificationRelaxed()) {
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.narrowPipeNetScopes = true;
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation *unanalyzableOp) {
      recordProtocolActionDomain(op, domain, unanalyzableOp, state);
    };
    solver.load<LaunchNodeDomainAnalysis>(state, options);
    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }
    if (state.sawError) {
      signalPassFailure();
      return;
    }

    llvm::MapVector<int64_t, DFBParticipantSet> producersByDFB;
    llvm::MapVector<int64_t, DFBParticipantSet> consumersByDFB;

    auto record = [&](llvm::MapVector<int64_t, DFBParticipantSet> &perDFB,
                      Operation *op, Value cb) {
      func::FuncOp thread = getEnclosingKernelThread(op);
      if (!thread) {
        return;
      }
      FailureOr<int64_t> dfbId = getDFBId(cb);
      assert(succeeded(dfbId) && "DFB identities were verified");
      auto domainIt = state.protocolActionDomains.find(op);
      ProtocolActionDomain actionDomain =
          domainIt == state.protocolActionDomains.end()
              ? ProtocolActionDomain{LaunchNodeDomain::unknown(), op}
              : domainIt->second;
      addParticipant(perDFB[*dfbId], thread, op, actionDomain.domain,
                     actionDomain.unanalyzableOp);
    };

    module.walk([&](Operation *op) {
      auto access = dyn_cast<DFBAccessOpInterface>(op);
      if (!access) {
        return;
      }
      for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
        if (isProducerDFBProtocolEffect(effect.kind)) {
          record(producersByDFB, op, effect.dfb);
        } else if (isConsumerDFBProtocolEffect(effect.kind)) {
          record(consumersByDFB, op, effect.dfb);
        }
      }
    });

    bool sawError = false;
    for (auto &entry : producersByDFB) {
      sawError |= verifyParticipantSet(
          entry.first, entry.second, bindSites.lookup(entry.first), "producer",
          "performed a producer action");
    }
    for (auto &entry : consumersByDFB) {
      sawError |= verifyParticipantSet(
          entry.first, entry.second, bindSites.lookup(entry.first), "consumer",
          "performed a consumer action");
    }

    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
