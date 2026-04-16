// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Copy Wait
//===----------------------------------------------------------------------===//
//
// Inserts missing ttl.wait for ttl.copy ops whose transfer handle has no
// wait user. The wait is placed immediately after the copy.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

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

    func.walk([&](CopyOp copy) {
      Value handle = copy.getXf();
      bool hasWait = llvm::any_of(
          handle.getUsers(), [](Operation *user) { return isa<WaitOp>(user); });
      if (hasWait) {
        return;
      }

      builder.setInsertionPointAfter(copy);
      WaitOp::create(builder, copy.getLoc(), handle);
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
