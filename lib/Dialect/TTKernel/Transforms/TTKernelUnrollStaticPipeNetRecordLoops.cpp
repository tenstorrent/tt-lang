// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELUNROLLSTATICPIPENETRECORDLOOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static void inlineSingleIterationLoop(scf::ForOp loop) {
  assert(loop.getInitArgs().empty() && loop.getNumResults() == 0 &&
         "local PipeNet record loops cannot carry values");

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), loop.getLowerBound());
  OpBuilder builder(loop);
  for (Operation &operation : loop.getBody()->without_terminator()) {
    builder.clone(operation, mapping);
  }
  loop.erase();
}

struct TTKernelUnrollStaticPipeNetRecordLoopsPass
    : impl::TTKernelUnrollStaticPipeNetRecordLoopsBase<
          TTKernelUnrollStaticPipeNetRecordLoopsPass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr(kPipeNetLocalRecordLoopAttrName)) {
        loops.push_back(loop);
      }
    });

    for (scf::ForOp loop : loops) {
      assert(loop.getInitArgs().empty() && loop.getNumResults() == 0 &&
             "local PipeNet record loops cannot carry values");
      std::optional<APInt> maybeTripCount = loop.getStaticTripCount();
      if (!maybeTripCount) {
        continue;
      }
      uint64_t tripCount = maybeTripCount->getZExtValue();
      if (tripCount == 0) {
        loop.erase();
        continue;
      }
      if (tripCount == 1) {
        inlineSingleIterationLoop(loop);
        continue;
      }
      if (failed(loopUnrollByFactor(loop, tripCount))) {
        loop.emitOpError("failed to unroll static local PipeNet record loop");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
