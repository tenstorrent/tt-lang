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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "DFBProtocolDomainAnalysis.h"
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

// Every waited DFB needs a compiler-visible push or an opaque call that may
// contain one.
LogicalResult
verifyDFBWaitsHavePushes(ModuleOp module,
                         const llvm::DenseMap<int64_t, BindCBOp> &bindSites) {
  llvm::DenseSet<int64_t> pushedDFBs;
  llvm::SmallMapVector<int64_t, Operation *, 4> firstWaitByDFB;
  module.walk([&](DFBAccessOpInterface access) {
    if (!getEnclosingKernelThread(access)) {
      return;
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      FailureOr<int64_t> dfbId = getDFBId(effect.dfb);
      assert(succeeded(dfbId) && "DFB identities were verified");
      if (effect.kind == DFBProtocolEffectKind::Push) {
        pushedDFBs.insert(*dfbId);
      } else if (effect.kind == DFBProtocolEffectKind::Wait) {
        firstWaitByDFB.try_emplace(*dfbId, access);
      }
    }
  });
  llvm::DenseMap<int64_t, SmallVector<OpaqueCallOp>> opaqueProtocolCalls =
      collectDFBsWithOpaqueProtocolActions(module, bindSites);

  bool sawError = false;
  for (auto [dfbId, waitOp] : firstWaitByDFB) {
    if (pushedDFBs.contains(dfbId) || opaqueProtocolCalls.contains(dfbId)) {
      continue;
    }
    InFlightDiagnostic diag = waitOp->emitError()
                              << "logical DFB " << dfbId
                              << " is waited on but no kernel thread pushes "
                                 "it";
    diag.attachNote()
        << "a DFB wait blocks until a matching push publishes data";
    if (BindCBOp bindSite = bindSites.lookup(dfbId)) {
      diag.attachNote(bindSite.getLoc()) << "dataflow buffer declared here";
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

    FailureOr<llvm::DenseMap<int64_t, BindCBOp>> bindSites =
        collectFinalizedDFBBindSites(module);
    if (failed(bindSites)) {
      signalPassFailure();
      return;
    }
    if (!hasDFBProtocolEffect(module, /*acquisitionsOnly=*/true)) {
      return;
    }

    DFBProtocolDomainState state;
    if (failed(
            initializeDFBProtocolDomainState(module, getArgument(), state))) {
      signalPassFailure();
      return;
    }

    if (failed(verifyDFBWaitsHavePushes(module, *bindSites))) {
      signalPassFailure();
      return;
    }

    if (applyDFBProtocolDomainVerificationRelaxation(module)) {
      return;
    }

    if (failed(analyzeDFBProtocolDomains(module, state))) {
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
      DFBProtocolActionDomain actionDomain = state.getProtocolActionDomain(op);
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
          entry.first, entry.second, bindSites->lookup(entry.first), "producer",
          "performed a producer action");
    }
    for (auto &entry : consumersByDFB) {
      sawError |= verifyParticipantSet(
          entry.first, entry.second, bindSites->lookup(entry.first), "consumer",
          "performed a consumer action");
    }

    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
