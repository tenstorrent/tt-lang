// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Inline and erase `ttl.pipenet_scope` markers after the verifier has
// consumed them. Pure transform; runs after `ttl-verify-pipenet-guards`.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLERASEPIPENETSCOPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLErasePipeNetScopesPass
    : impl::TTLErasePipeNetScopesBase<TTLErasePipeNetScopesPass> {
  void runOnOperation() override {
    SmallVector<PipeNetScopeOp> scopes;
    getOperation().walk(
        [&](PipeNetScopeOp scopeOp) { scopes.push_back(scopeOp); });
    for (PipeNetScopeOp scopeOp : scopes) {
      Block &body = scopeOp.getBody().front();
      // The body terminates with a `ttl.yield` (auto-inserted via
      // SingleBlockImplicitTerminator). Drop it so it isn't spliced into
      // the parent block.
      if (Operation *terminator = body.getTerminator();
          terminator && isa<YieldOp>(terminator)) {
        terminator->erase();
      }
      Operation *scopeOperation = scopeOp.getOperation();
      scopeOperation->getBlock()->getOperations().splice(
          scopeOperation->getIterator(), body.getOperations(), body.begin(),
          body.end());
      scopeOperation->erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
