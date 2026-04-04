// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Reconstruct signpost ops from SignpostScopeAttr metadata in FusedLoc.
// Only ops explicitly annotated by TTLAttachSignpostScopes carry the
// attribute; compiler-inserted ops that inherited FusedLoc via cloning
// are ignored.  Unannotated ops between annotated ops with compatible
// scopes remain inside the enclosing scope (no artificial fragmentation).
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLEMITSIGNPOSTSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static SignpostScopeAttr getSignpostScope(Operation *op) {
  auto fusedLoc = dyn_cast<FusedLoc>(op->getLoc());
  if (!fusedLoc) {
    return nullptr;
  }
  return dyn_cast_or_null<SignpostScopeAttr>(fusedLoc.getMetadata());
}

static void stripSignpostScope(Operation *op) {
  auto fusedLoc = dyn_cast<FusedLoc>(op->getLoc());
  if (!fusedLoc) {
    return;
  }
  if (!isa_and_nonnull<SignpostScopeAttr>(fusedLoc.getMetadata())) {
    return;
  }
  auto locs = fusedLoc.getLocations();
  if (!locs.empty()) {
    op->setLoc(locs.front());
  }
}

static void emitScopeTransition(OpBuilder &builder, Location loc,
                                SmallVector<std::string> &current,
                                ArrayRef<std::string> target) {
  size_t common = 0;
  while (common < current.size() && common < target.size() &&
         current[common] == target[common]) {
    ++common;
  }

  for (size_t idx = current.size(); idx > common; --idx) {
    SignpostOp::create(builder, loc, builder.getStringAttr(current[idx - 1]),
                       builder.getUnitAttr());
  }

  for (size_t idx = common; idx < target.size(); ++idx) {
    SignpostOp::create(builder, loc, builder.getStringAttr(target[idx]),
                       UnitAttr());
  }

  current.assign(target.begin(), target.end());
}

/// Find the next annotated op after `start` in the same block.
/// Returns nullptr if none found.
static Operation *findNextAnnotated(Operation *start) {
  for (auto *op = start->getNextNode(); op; op = op->getNextNode()) {
    if (getSignpostScope(op)) {
      return op;
    }
  }
  return nullptr;
}

/// Check if `target` scope is a prefix-compatible extension of `current`.
/// That is, `current` is a prefix of `target` (the next annotated op
/// opens deeper scopes without closing any).
static bool isScopeExtension(ArrayRef<std::string> current,
                             ArrayRef<std::string> target) {
  if (target.size() < current.size()) {
    return false;
  }
  for (size_t idx = 0; idx < current.size(); ++idx) {
    if (current[idx] != target[idx]) {
      return false;
    }
  }
  return true;
}

static void processBlock(Block &block, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<std::string> currentScope;
  SmallVector<std::string> empty;

  for (auto &op : llvm::make_early_inc_range(block)) {
    // Recurse into nested regions first.
    for (auto &region : op.getRegions()) {
      for (auto &nestedBlock : region) {
        processBlock(nestedBlock, ctx);
      }
    }

    auto scopeAttr = getSignpostScope(&op);
    if (!scopeAttr) {
      // Unannotated op (compiler-inserted).  Check if the next annotated
      // op has a compatible scope.  If so, keep current scope open (the
      // unannotated op is part of the enclosing scope's implementation).
      // If not (or no next annotated op), close all scopes.
      if (!currentScope.empty()) {
        Operation *nextAnnotated = findNextAnnotated(&op);
        if (nextAnnotated) {
          auto nextAttr = getSignpostScope(nextAnnotated);
          SmallVector<std::string> nextTarget;
          for (auto strAttr : nextAttr.getScopes()) {
            nextTarget.push_back(strAttr.getValue().str());
          }
          if (isScopeExtension(currentScope, nextTarget)) {
            continue; // Keep scopes open.
          }
        }
        // Incompatible or no next annotated op: close all scopes.
        builder.setInsertionPoint(&op);
        emitScopeTransition(builder, op.getLoc(), currentScope, empty);
      }
      continue;
    }

    SmallVector<std::string> target;
    for (auto strAttr : scopeAttr.getScopes()) {
      target.push_back(strAttr.getValue().str());
    }

    builder.setInsertionPoint(&op);
    emitScopeTransition(builder, op.getLoc(), currentScope, target);
    stripSignpostScope(&op);
  }

  // Close remaining scopes before the block terminator.
  if (!currentScope.empty() && !block.empty()) {
    Operation *terminator = block.getTerminator();
    if (terminator) {
      builder.setInsertionPoint(terminator);
    } else {
      builder.setInsertionPointAfter(&block.back());
    }
    emitScopeTransition(builder, block.back().getLoc(), currentScope, empty);
  }
}

struct TTLEmitSignpostScopesPass
    : impl::TTLEmitSignpostScopesBase<TTLEmitSignpostScopesPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    for (auto &region : funcOp->getRegions()) {
      for (auto &block : region) {
        processBlock(block, &getContext());
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
