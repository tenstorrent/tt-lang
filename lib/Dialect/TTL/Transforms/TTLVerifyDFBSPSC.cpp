// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify DFB SPSC
//===----------------------------------------------------------------------===//
//
// Rejects modules in which a dataflow buffer (identified by its `cb_index`) has
// more than one producer or consumer kernel thread active on the same launched
// node. tt-metal CBs are single-producer single-consumer at the API level; see
// `docs/development/DFBManagement.md` for the rationale.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYDFBSPSC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Domain fact recorded for one dataflow buffer acquire operation.
struct AcquireDomain {
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

/// A kernel thread that produces or consumes a dataflow buffer.
///
/// Multiple acquires in the same thread are merged into one participant because
/// SPSC is a thread-level property, not an operation-level property.
struct DFBParticipant {
  func::FuncOp thread;
  Operation *op = nullptr;
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

/// Producers or consumers for one finalized dataflow buffer index.
struct DFBParticipantSet {
  llvm::SmallMapVector<func::FuncOp, DFBParticipant, 2> participants;
};

/// Analysis state shared by the dataflow solver and the verifier pass.
struct ModuleState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, AcquireDomain> acquireDomains;
};

/// Record the launch-node domain that reaches a producer or consumer acquire.
void recordAcquireDomain(Operation *op, const LaunchNodeDomain &domain,
                         Operation *unanalyzableOp, ModuleState &state) {
  if (!isa<CBReserveOp, CBWaitOp>(op)) {
    return;
  }
  state.acquireDomains[op] = {domain, unanalyzableOp};
}

/// Add a thread participant, merging repeated acquires from the same thread.
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

/// Attach notes shared by producer and consumer SPSC diagnostics.
void attachCommonNotes(InFlightDiagnostic &diag, Operation *bindSite,
                       llvm::StringRef role) {
  diag.attachNote() << "tt-metal CBs are single-producer single-consumer; "
                       "allocate one DFB per "
                    << role;
  if (bindSite) {
    diag.attachNote(bindSite->getLoc()) << "dataflow buffer declared here";
  }
}

/// Emit the error for two participant domains with a proven common launch node.
void emitOverlapError(int64_t cbIndex, const DFBParticipant &lhs,
                      const DFBParticipant &rhs,
                      const LaunchNodeDomain &overlap, Operation *bindSite,
                      llvm::StringRef role, llvm::StringRef verbedHere) {
  InFlightDiagnostic diag = lhs.op->emitError()
                            << "dataflow buffer cb_index=" << cbIndex << " has "
                            << "multiple " << role
                            << " threads active on the same launched node";
  if (!overlap.nodes.empty()) {
    LaunchNodeCoord example = *overlap.nodes.begin();
    diag.attachNote() << "example overlapping node: core_x=" << example.x
                      << ", core_y=" << example.y;
  }
  diag.attachNote(rhs.op->getLoc()) << "also " << verbedHere << " here";
  attachCommonNotes(diag, bindSite, role);
}

/// Emit the conservative error used when a domain-dependent predicate could not
/// be evaluated statically.
void emitUnknownDomainError(int64_t cbIndex, const DFBParticipantSet &set,
                            Operation *bindSite, llvm::StringRef role,
                            llvm::StringRef verbedHere) {
  auto unknownIt = llvm::find_if(set.participants, [](const auto &entry) {
    return !entry.second.domain.known;
  });
  assert(unknownIt != set.participants.end() &&
         "expected at least one unknown participant domain");

  const DFBParticipant &primary = unknownIt->second;
  InFlightDiagnostic diag =
      primary.op->emitError()
      << "dataflow buffer cb_index=" << cbIndex << " has multiple " << role
      << " threads, but SPSC could not be statically proven";
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

/// Verify one dataflow buffer role after participants have been coalesced by
/// kernel thread.
bool verifyParticipantSet(int64_t cbIndex, const DFBParticipantSet &set,
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
        emitOverlapError(cbIndex, lhs, rhs, overlap, bindSite, role,
                         verbedHere);
        return true;
      }
    }
  }

  if (llvm::any_of(set.participants, [](const auto &entry) {
        return !entry.second.domain.known;
      })) {
    emitUnknownDomainError(cbIndex, set, bindSite, role, verbedHere);
    return true;
  }
  return false;
}

struct TTLVerifyDFBSPSCPass
    : public impl::TTLVerifyDFBSPSCBase<TTLVerifyDFBSPSCPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleState state;
    state.initialize(module);

    bool hasAcquire = false;
    llvm::DenseMap<int64_t, Operation *> bindSites;

    module.walk([&](Operation *op) {
      if (isa<CBReserveOp, CBWaitOp>(op) && getEnclosingKernelThread(op)) {
        hasAcquire = true;
      } else if (auto bindOp = dyn_cast<BindCBOp>(op)) {
        std::optional<int64_t> idx = getCBIndex(bindOp.getResult());
        if (idx.has_value()) {
          bindSites.try_emplace(*idx, op);
        }
      }
    });

    if (!hasAcquire) {
      return;
    }
    if (!state.hasLaunchGrid) {
      module.emitError()
          << "ttl-verify-dfb-spsc requires a `ttl.launch_grid` module "
             "attribute (an i64 array of length 2 with positive entries) "
             "when verifying DFB acquire ops";
      signalPassFailure();
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.narrowPipeNetScopes = true;
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation *unanalyzableOp) {
      recordAcquireDomain(op, domain, unanalyzableOp, state);
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

    llvm::MapVector<int64_t, DFBParticipantSet> producers;
    llvm::MapVector<int64_t, DFBParticipantSet> consumers;

    auto record = [&](llvm::MapVector<int64_t, DFBParticipantSet> &perCB,
                      Operation *op, Value cb) {
      func::FuncOp thread = getEnclosingKernelThread(op);
      if (!thread) {
        return;
      }
      std::optional<int64_t> idx = getCBIndex(cb);
      assert(idx.has_value() &&
             "ttl-verify-dfb-spsc requires finalized cb_index; run "
             "ttl-finalize-dfb-indices first");
      auto domainIt = state.acquireDomains.find(op);
      AcquireDomain acquireDomain =
          domainIt == state.acquireDomains.end()
              ? AcquireDomain{LaunchNodeDomain::unknown(), op}
              : domainIt->second;
      addParticipant(perCB[*idx], thread, op, acquireDomain.domain,
                     acquireDomain.unanalyzableOp);
    };

    module.walk([&](Operation *op) {
      if (auto reserveOp = dyn_cast<CBReserveOp>(op)) {
        record(producers, op, reserveOp.getCb());
      } else if (auto waitOp = dyn_cast<CBWaitOp>(op)) {
        record(consumers, op, waitOp.getCb());
      }
    });

    bool sawError = false;
    for (auto &entry : producers) {
      sawError |= verifyParticipantSet(entry.first, entry.second,
                                       bindSites.lookup(entry.first),
                                       "producer", "reserved");
    }
    for (auto &entry : consumers) {
      sawError |= verifyParticipantSet(entry.first, entry.second,
                                       bindSites.lookup(entry.first),
                                       "consumer", "waited on");
    }

    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
