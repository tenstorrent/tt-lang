// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Copy Wait
//===----------------------------------------------------------------------===//
//
// Inserts missing ttl.wait for ttl.copy ops whose result has no synchronization
// user. The wait is placed immediately after the copy.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Analysis/ValueOriginAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "ttl-insert-copy-wait"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTCOPYWAIT
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertCopyWaitPass
    : public impl::TTLInsertCopyWaitBase<TTLInsertCopyWaitPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());
    ValueOriginAnalysis valueOrigins(func);
    llvm::SmallPtrSet<Operation *, 8> synchronizedCopies;

    auto recordSynchronizedOrigins = [&](Value handle) {
      for (Value origin : valueOrigins.getOrigins(handle)) {
        if (auto copy = origin.getDefiningOp<CopyOp>()) {
          synchronizedCopies.insert(copy.getOperation());
        }
      }
    };
    func.walk([&](Operation *operation) {
      if (auto wait = dyn_cast<WaitOp>(operation)) {
        recordSynchronizedOrigins(wait.getXf());
        return;
      }
      if (auto waitAny = dyn_cast<WaitAnyOp>(operation)) {
        for (Value request : waitAny.getRequests()) {
          recordSynchronizedOrigins(request);
        }
      }
    });

    func.walk([&](CopyOp copy) {
      if (synchronizedCopies.contains(copy.getOperation())) {
        return;
      }

      builder.setInsertionPointAfter(copy);
      WaitOp::create(builder, copy.getLoc(), copy.getXf());
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
