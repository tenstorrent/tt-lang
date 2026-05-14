// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify DFB SPSC
//===----------------------------------------------------------------------===//
//
// Rejects modules in which a dataflow buffer (identified by its `cb_index`)
// is reserved or waited-on in more than one `ttl.kernel_thread`-tagged
// `func.func`. tt-metal CBs are single-producer single-consumer at the
// API level; see `docs/development/DFBManagement.md` for the rationale.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYDFBSPSC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

// Per-DFB index, the first acquire op observed in each kernel thread. We only
// keep the first op per thread because the diagnostic just needs one site per
// participant.
struct DFBThreadSet {
  llvm::SmallMapVector<func::FuncOp, Operation *, 2> threads;
};

// Error at the first thread's site, notes at the rest and (when available) at
// the `ttl.bind_cb` that declares this index.
static void emitMultiThreadError(int64_t cbIndex, const DFBThreadSet &sites,
                                 Operation *bindSite, llvm::StringRef role,
                                 llvm::StringRef verbedHere) {
  auto it = sites.threads.begin();
  Operation *primary = it->second;
  InFlightDiagnostic diag =
      primary->emitError() << "dataflow buffer cb_index=" << cbIndex << " has "
                           << sites.threads.size() << " " << role << " threads";
  diag.attachNote() << "tt-metal CBs are single-producer single-consumer; "
                       "allocate one DFB per "
                    << role;
  for (++it; it != sites.threads.end(); ++it) {
    diag.attachNote(it->second->getLoc()) << "also " << verbedHere << " here";
  }
  if (bindSite) {
    diag.attachNote(bindSite->getLoc()) << "dataflow buffer declared here";
  }
}

struct TTLVerifyDFBSPSCPass
    : public impl::TTLVerifyDFBSPSCBase<TTLVerifyDFBSPSCPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::MapVector<int64_t, DFBThreadSet> producers;
    llvm::MapVector<int64_t, DFBThreadSet> consumers;
    llvm::DenseMap<int64_t, Operation *> bindSites;

    auto record = [&](llvm::MapVector<int64_t, DFBThreadSet> &perCB,
                      Operation *op, Value cb) {
      func::FuncOp thread = getEnclosingKernelThread(op);
      if (!thread) {
        return;
      }
      std::optional<int64_t> idx = getCBIndex(cb);
      assert(idx.has_value() &&
             "ttl-verify-dfb-spsc requires finalized cb_index; run "
             "ttl-finalize-dfb-indices first");
      perCB[*idx].threads.insert({thread, op});
    };

    module.walk([&](Operation *op) {
      if (auto reserveOp = dyn_cast<CBReserveOp>(op)) {
        record(producers, op, reserveOp.getCb());
      } else if (auto waitOp = dyn_cast<CBWaitOp>(op)) {
        record(consumers, op, waitOp.getCb());
      } else if (auto bindOp = dyn_cast<BindCBOp>(op)) {
        std::optional<int64_t> idx = getCBIndex(bindOp.getResult());
        if (idx.has_value()) {
          bindSites.try_emplace(*idx, op);
        }
      }
    });

    bool sawError = false;
    for (auto &entry : producers) {
      if (entry.second.threads.size() > 1) {
        emitMultiThreadError(entry.first, entry.second,
                             bindSites.lookup(entry.first), "producer",
                             "reserved");
        sawError = true;
      }
    }
    for (auto &entry : consumers) {
      if (entry.second.threads.size() > 1) {
        emitMultiThreadError(entry.first, entry.second,
                             bindSites.lookup(entry.first), "consumer",
                             "waited on");
        sawError = true;
      }
    }

    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
