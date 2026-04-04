// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Encode signpost scope information into op locations via FusedLoc with
// SignpostScopeAttr metadata.  Locations survive all MLIR transformations.
// Compiler-inserted ops that inherit FusedLoc via cloning will have plain
// FusedLoc (no SignpostScopeAttr), so the emit pass ignores them.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLATTACHSIGNPOSTSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLAttachSignpostScopesPass
    : impl::TTLAttachSignpostScopesBase<TTLAttachSignpostScopesPass> {
  using Base::Base;

  void processBlock(Block &block) {
    MLIRContext *ctx = &getContext();
    SmallVector<std::string> scopeStack;
    SmallVector<SignpostOp> toErase;

    for (auto &op : block) {
      if (auto sp = dyn_cast<SignpostOp>(&op)) {
        if (!sp.getIsEnd()) {
          scopeStack.push_back(sp.getName().str());
        } else {
          for (int idx = scopeStack.size() - 1; idx >= 0; --idx) {
            if (scopeStack[idx] == sp.getName()) {
              scopeStack.erase(scopeStack.begin() + idx);
              break;
            }
          }
        }
        toErase.push_back(sp);
        continue;
      }

      if (!scopeStack.empty()) {
        SmallVector<StringAttr> scopeAttrs;
        for (auto &name : scopeStack) {
          scopeAttrs.push_back(StringAttr::get(ctx, name));
        }
        auto metadata = SignpostScopeAttr::get(ctx, scopeAttrs);
        op.setLoc(FusedLoc::get(ctx, {op.getLoc()}, metadata));
      }
    }

    for (auto sp : toErase) {
      sp.erase();
    }
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([this](Block *block) { processBlock(*block); });
  }
};

} // namespace
} // namespace mlir::tt::ttl
