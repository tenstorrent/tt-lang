// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELBATCHSTATICPIPENETRECEIVES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct ReceiveLoopMatch {
  ttk::CBReserveBackOp reserve;
  ttk::NocSemaphoreIncOp readySignal;
  ttk::SemaphoreWaitMinOp completionWait;
  ttk::CBPushBackOp push;
};

static bool isDefinedInside(Operation *ancestor, Value value) {
  if (Operation *definition = value.getDefiningOp()) {
    return ancestor->isProperAncestor(definition);
  }
  auto blockArgument = dyn_cast<BlockArgument>(value);
  return blockArgument &&
         ancestor->isProperAncestor(blockArgument.getOwner()->getParentOp());
}

/// Match the computed-address receiver protocol. Receiver-published addresses
/// contain a NoC write before the ready signal and are intentionally rejected:
/// each post would otherwise publish the same coalesced write pointer.
static std::optional<ReceiveLoopMatch> matchReceiveLoop(scf::ForOp loop) {
  ReceiveLoopMatch match;
  unsigned reserveCount = 0;
  unsigned readySignalCount = 0;
  unsigned completionWaitCount = 0;
  unsigned pushCount = 0;

  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0) {
      return std::nullopt;
    }
    if (auto reserve = dyn_cast<ttk::CBReserveBackOp>(operation)) {
      match.reserve = reserve;
      ++reserveCount;
    } else if (auto readySignal =
                   dyn_cast<ttk::NocSemaphoreIncOp>(operation)) {
      match.readySignal = readySignal;
      ++readySignalCount;
    } else if (auto completionWait =
                   dyn_cast<ttk::SemaphoreWaitMinOp>(operation)) {
      match.completionWait = completionWait;
      ++completionWaitCount;
    } else if (auto push = dyn_cast<ttk::CBPushBackOp>(operation)) {
      match.push = push;
      ++pushCount;
    }
  }

  if (reserveCount != 1 || readySignalCount != 1 ||
      completionWaitCount != 1 || pushCount != 1 ||
      match.reserve.getCb() != match.push.getCb() ||
      match.reserve.getNumPages() != match.push.getNumPages() ||
      isDefinedInside(loop, match.reserve.getCb()) ||
      isDefinedInside(loop, match.reserve.getNumPages()) ||
      (match.readySignal.getNoc() &&
       isDefinedInside(loop, match.readySignal.getNoc())) ||
      !match.reserve->isBeforeInBlock(match.readySignal) ||
      !match.readySignal->isBeforeInBlock(match.completionWait) ||
      !match.completionWait->isBeforeInBlock(match.push) ||
      match.push->getNextNode() != loop.getBody()->getTerminator()) {
    return std::nullopt;
  }

  // The only observable operation before the ready signal must be the DFB
  // reservation itself. This excludes address publication and any callback
  // work whose ordering cannot be changed.
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (&operation == match.readySignal.getOperation()) {
      break;
    }
    if (&operation != match.reserve.getOperation() && !isPure(&operation)) {
      return std::nullopt;
    }
  }
  return match;
}

static LogicalResult batchReceiveLoop(scf::ForOp loop,
                                      ReceiveLoopMatch match,
                                      uint64_t tripCount) {
  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !step || *step <= 0) {
    return failure();
  }

  SmallVector<Operation *> bodyOperations;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    bodyOperations.push_back(&operation);
  }
  auto reserveIt = llvm::find(bodyOperations, match.reserve.getOperation());
  auto readySignalIt =
      llvm::find(bodyOperations, match.readySignal.getOperation());
  if (reserveIt == bodyOperations.end() ||
      readySignalIt == bodyOperations.end()) {
    return failure();
  }

  OpBuilder builder(loop);
  SmallVector<std::unique_ptr<IRMapping>> iterationMappings;
  iterationMappings.reserve(tripCount);

  // Clone record-address calculations first. They are pure by the matcher and
  // their mappings are reused when cloning each ready signal below.
  for (uint64_t iteration = 0; iteration < tripCount; ++iteration) {
    auto mapping = std::make_unique<IRMapping>();
    int64_t inductionValue =
        *lowerBound + static_cast<int64_t>(iteration) * *step;
    Value inductionConstant = arith::ConstantIndexOp::create(
        builder, loop.getLoc(), inductionValue);
    mapping->map(loop.getInductionVar(), inductionConstant);
    for (Operation *operation : llvm::make_range(bodyOperations.begin(),
                                                 reserveIt)) {
      builder.clone(*operation, *mapping);
    }
    iterationMappings.push_back(std::move(mapping));
  }

  Value reserveCB = iterationMappings.front()->lookupOrDefault(match.reserve.getCb());
  Value pagesPerRecord =
      iterationMappings.front()->lookupOrDefault(match.reserve.getNumPages());
  Value recordCount = arith::ConstantIntOp::create(
      builder, loop.getLoc(), static_cast<int64_t>(tripCount), 32);
  Value totalPages = arith::MulIOp::create(builder, loop.getLoc(),
                                          pagesPerRecord, recordCount);
  ttk::CBReserveBackOp::create(builder, loop.getLoc(), reserveCB, totalPages);

  SmallVector<ttk::NocSemaphoreIncOp> clonedReadySignals;
  clonedReadySignals.reserve(tripCount);
  for (std::unique_ptr<IRMapping> &mapping : iterationMappings) {
    for (Operation *operation : llvm::make_range(std::next(reserveIt),
                                                 std::next(readySignalIt))) {
      Operation *clone = builder.clone(*operation, *mapping);
      if (auto readySignal = dyn_cast<ttk::NocSemaphoreIncOp>(clone)) {
        clonedReadySignals.push_back(readySignal);
      }
    }
  }
  assert(clonedReadySignals.size() == tripCount &&
         "matcher guarantees one ready signal per record");

  // One barrier flushes every non-posted ready increment before this receiver
  // blocks on completion, matching the grouped PipeNet protocol.
  ttk::NocAsyncAtomicBarrierOp::create(
      builder, loop.getLoc(), clonedReadySignals.front().getNoc());

  for (std::unique_ptr<IRMapping> &mapping : iterationMappings) {
    for (Operation *operation : llvm::make_range(std::next(readySignalIt),
                                                 bodyOperations.end())) {
      builder.clone(*operation, *mapping);
    }
  }

  loop.erase();
  return success();
}

struct TTKernelBatchStaticPipeNetReceivesPass
    : impl::TTKernelBatchStaticPipeNetReceivesBase<
          TTKernelBatchStaticPipeNetReceivesPass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr(kPipeNetReceiveRecordLoopAttrName)) {
        loops.push_back(loop);
      }
    });

    for (scf::ForOp loop : loops) {
      std::optional<APInt> maybeTripCount = loop.getStaticTripCount();
      if (!maybeTripCount || maybeTripCount->getZExtValue() < 2) {
        continue;
      }
      std::optional<ReceiveLoopMatch> match = matchReceiveLoop(loop);
      if (!match) {
        continue;
      }
      if (failed(batchReceiveLoop(loop, *match,
                                  maybeTripCount->getZExtValue()))) {
        loop.emitOpError("failed to batch static PipeNet receives");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
