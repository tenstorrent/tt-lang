// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLERASEDEADOPS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Iteratively erase dead ops until fixpoint.
/// Returns true if any ops were erased.
static bool eraseDeadOpsIteration(ModuleOp mod) {
  bool changed = false;
  SmallVector<Operation *, 8> toErase;

  mod.walk([&](Operation *op) {
    // Erase dead ttcore.get_global ops.
    if (llvm::isa<ttcore::GetGlobalOp>(op) && op->use_empty()) {
      toErase.push_back(op);
      return;
    }

    // Erase dead unrealized_conversion_cast ops.
    if (llvm::isa<UnrealizedConversionCastOp>(op) && op->use_empty()) {
      toErase.push_back(op);
      return;
    }
  });

  for (auto *op : toErase) {
    op->erase();
    changed = true;
  }

  return changed;
}

struct TTLEraseDeadOpsPass
    : impl::TTLEraseDeadOpsBase<TTLEraseDeadOpsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Iterate until fixpoint.
    while (eraseDeadOpsIteration(mod)) {
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
