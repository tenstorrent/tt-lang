// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLRESOLVESTATICDISPATCH
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static LogicalResult validateDispatcher(func::FuncOp dispatcher) {
  if (!isa_and_nonnull<UnitAttr>(dispatcher->getAttr(kDispatcherAttrName))) {
    return dispatcher.emitOpError()
           << "requires '" << kDispatcherAttrName << "' to be a unit attribute";
  }
  if (dispatcher.isExternal()) {
    return dispatcher.emitOpError("dispatch controller must have a body");
  }
  if (dispatcher.getFunctionType().getNumResults() != 0) {
    return dispatcher.emitOpError(
        "must not return values; target outputs are tensor operands");
  }
  if (!dispatcher.getBody().hasOneBlock()) {
    return dispatcher.emitOpError(
        "static dispatch controller must have one block after canonicalization");
  }

  bool hasInvocation = false;
  for (Operation &operation : dispatcher.getBody().front()) {
    if (isa<DispatchInvokeOp>(operation)) {
      hasInvocation = true;
      continue;
    }
    if (auto returnOp = dyn_cast<func::ReturnOp>(operation)) {
      if (!returnOp.getOperands().empty()) {
        return returnOp.emitOpError(
            "must not return values from a static dispatcher");
      }
      continue;
    }
    return operation.emitOpError(
        "remains in a static dispatch controller; run constant propagation "
        "and canonicalization before ttl-resolve-static-dispatch");
  }
  if (!hasInvocation) {
    return dispatcher.emitOpError(
        "static dispatch controller must invoke at least one target");
  }
  return success();
}

struct TTLResolveStaticDispatchPass
    : impl::TTLResolveStaticDispatchBase<TTLResolveStaticDispatchPass> {
  using TTLResolveStaticDispatchBase::TTLResolveStaticDispatchBase;

  void runOnOperation() override {
    SmallVector<func::FuncOp> dispatchers;
    getOperation().walk([&](func::FuncOp function) {
      if (function->hasAttr(kDispatcherAttrName)) {
        dispatchers.push_back(function);
      }
    });

    for (func::FuncOp dispatcher : dispatchers) {
      if (failed(validateDispatcher(dispatcher))) {
        signalPassFailure();
        return;
      }
    }
    for (func::FuncOp dispatcher : dispatchers) {
      dispatcher->setAttr(kDispatchResolvedAttrName,
                          UnitAttr::get(dispatcher.getContext()));
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
