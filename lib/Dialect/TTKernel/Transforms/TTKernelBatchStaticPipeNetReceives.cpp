// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Post independently addressed receives before waiting for their payloads.
// TTL planning proves initial storage availability; this pass checks that the
// lowered protocol contains no other effects whose ordering would change.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELBATCHSTATICPIPENETRECEIVES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttk = mlir::tt::ttkernel;

struct ReceiveBatchPlan {
  scf::ForOp loop;
  ttk::CBReserveBackOp reserve;
  ttk::NocSemaphoreIncOp ready;
  SmallVector<int64_t> inductionValues;
  int64_t totalPages;
};

// Accept accesses to a nonescaping stack allocation outside `loop`. Such
// counters are inaccessible to senders and all accesses retain their order.
static bool isPrivateCounterAccess(Operation *operation, scf::ForOp loop) {
  Value buffer;
  if (auto load = dyn_cast<memref::LoadOp>(operation)) {
    buffer = load.getMemRef();
  } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
    buffer = store.getMemRef();
  } else {
    return false;
  }
  return buffer.getDefiningOp<memref::AllocaOp>() &&
         loop.isDefinedOutsideOfLoop(buffer) &&
         llvm::all_of(buffer.getUsers(), [](Operation *user) {
           return isa<memref::LoadOp, memref::StoreOp>(user);
         });
}

// Plan only protocols whose non-readiness effects remain in their original
// order. The marker proves storage availability, not arbitrary code motion.
static FailureOr<ReceiveBatchPlan> planReceiveBatch(scf::ForOp loop) {
  auto capacity =
      loop->getAttrOfType<IntegerAttr>(kPipeNetInitialReceiveCapacityAttrName);
  auto tripCount = loop.getStaticTripCount();
  if (!capacity || capacity.getInt() <= 0 || !tripCount ||
      tripCount->getActiveBits() > 32 || tripCount->getZExtValue() < 2 ||
      !loop.getInitArgs().empty() || loop.getNumResults() != 0) {
    return failure();
  }
  ttk::CBReserveBackOp reserve;
  ttk::NocSemaphoreIncOp ready;
  ttk::SemaphoreWaitMinOp wait;
  ttk::CBPushBackOp push;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0) {
      return failure();
    }
    if (auto candidate = dyn_cast<ttk::CBReserveBackOp>(operation)) {
      if (reserve) {
        return failure();
      }
      reserve = candidate;
    } else if (auto candidate = dyn_cast<ttk::NocSemaphoreIncOp>(operation)) {
      if (ready) {
        return failure();
      }
      ready = candidate;
    } else if (auto candidate = dyn_cast<ttk::SemaphoreWaitMinOp>(operation)) {
      if (wait) {
        return failure();
      }
      wait = candidate;
    } else if (auto candidate = dyn_cast<ttk::CBPushBackOp>(operation)) {
      if (push) {
        return failure();
      }
      push = candidate;
    } else if (!isPure(&operation) &&
               !(ready && isPrivateCounterAccess(&operation, loop))) {
      return failure();
    }
  }
  if (!reserve || !ready || !wait || !push || reserve.getCb() != push.getCb() ||
      reserve.getNumPages() != push.getNumPages() ||
      !loop.isDefinedOutsideOfLoop(reserve.getCb()) ||
      !reserve->isBeforeInBlock(ready) || !ready->isBeforeInBlock(wait) ||
      !wait->isBeforeInBlock(push) ||
      push->getNextNode() != loop.getBody()->getTerminator() ||
      (ready.getNoc() && !loop.isDefinedOutsideOfLoop(ready.getNoc()))) {
    return failure();
  }
  auto pages = getConstantIntValue(reserve.getNumPages());
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!pages || *pages <= 0 || !lower || !step || *step <= 0) {
    return failure();
  }
  auto totalPages =
      llvm::checkedMul(*pages, static_cast<int64_t>(tripCount->getZExtValue()));
  if (!totalPages || *totalPages > capacity.getInt() ||
      !llvm::isUInt<32>(*totalPages)) {
    return failure();
  }
  ReceiveBatchPlan plan{loop, reserve, ready, {}, *totalPages};
  int64_t inductionValue = *lower;
  for (uint64_t iteration = 0; iteration < tripCount->getZExtValue();
       ++iteration) {
    plan.inductionValues.push_back(inductionValue);
    if (iteration + 1 != tripCount->getZExtValue()) {
      auto next = llvm::checkedAdd(inductionValue, *step);
      if (!next) {
        return failure();
      }
      inductionValue = *next;
    }
  }
  return plan;
}

// Reserve the proven initial capacity, post every sender, then preserve the
// original wait/push order so consumers observe the same payload sequence.
static void applyReceiveBatch(ReceiveBatchPlan &plan) {
  OpBuilder builder(plan.loop);
  Location location = plan.loop.getLoc();
  Value pages =
      arith::ConstantIntOp::create(builder, location, plan.totalPages, 32);
  ttk::CBReserveBackOp::create(builder, location, plan.reserve.getCb(), pages);
  SmallVector<IRMapping> mappings(plan.inductionValues.size());
  for (auto [inductionValue, mapping] :
       llvm::zip_equal(plan.inductionValues, mappings)) {
    Value induction =
        arith::ConstantIndexOp::create(builder, location, inductionValue);
    mapping.map(plan.loop.getInductionVar(), induction);
    for (Operation &operation : plan.loop.getBody()->without_terminator()) {
      if (&operation != plan.reserve) {
        builder.clone(operation, mapping);
      }
      if (&operation == plan.ready) {
        break;
      }
    }
  }
  // Flush all readiness increments before blocking on a sender's completion.
  ttk::NocAsyncAtomicBarrierOp::create(builder, location, plan.ready.getNoc());
  for (IRMapping &mapping : mappings) {
    for (Operation *operation = plan.ready->getNextNode();
         operation != plan.loop.getBody()->getTerminator();
         operation = operation->getNextNode()) {
      builder.clone(*operation, mapping);
    }
  }
  plan.loop.erase();
}

struct TTKernelBatchStaticPipeNetReceivesPass
    : impl::TTKernelBatchStaticPipeNetReceivesBase<
          TTKernelBatchStaticPipeNetReceivesPass> {
  void runOnOperation() override {
    SmallVector<ReceiveBatchPlan> plans;
    SmallVector<scf::ForOp> markedLoops;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kPipeNetInitialReceiveCapacityAttrName)) {
        return;
      }
      markedLoops.push_back(loop);
      FailureOr<ReceiveBatchPlan> plan = planReceiveBatch(loop);
      if (succeeded(plan)) {
        plans.push_back(std::move(*plan));
      }
    });
    for (scf::ForOp loop : markedLoops) {
      loop->removeAttr(kPipeNetInitialReceiveCapacityAttrName);
    }
    for (ReceiveBatchPlan &plan : plans) {
      applyReceiveBatch(plan);
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
