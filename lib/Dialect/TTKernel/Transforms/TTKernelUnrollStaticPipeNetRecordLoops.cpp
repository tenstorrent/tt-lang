// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELUNROLLSTATICPIPENETRECORDLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTKernelUnrollStaticPipeNetRecordLoopsPass
    : impl::TTKernelUnrollStaticPipeNetRecordLoopsBase<
          TTKernelUnrollStaticPipeNetRecordLoopsPass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp> recordLoops;
    // Unroll inner loops first because unrolling a parent invalidates its
    // nested operation handles and clones any unprocessed loop markers.
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr(kPipeNetLocalRecordLoopAttrName)) {
        recordLoops.push_back(loop);
      }
    });

    for (scf::ForOp recordLoop : recordLoops) {
      assert(recordLoop.getInitArgs().empty() &&
             recordLoop.getNumResults() == 0 &&
             "local PipeNet record loops cannot carry values");
      std::optional<APInt> tripCount = recordLoop.getStaticTripCount();
      if (!tripCount) {
        recordLoop->removeAttr(kPipeNetLocalRecordLoopAttrName);
        continue;
      }
      if (tripCount->isZero()) {
        recordLoop.erase();
        continue;
      }
      if (failed(loopUnrollFull(recordLoop))) {
        recordLoop.emitOpError(
            "failed to fully unroll local PipeNet record loop");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
