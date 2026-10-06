// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLRESOLVESTATICDISPATCH
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct ArgumentEvent {
  DispatchInvokeOp invocation;
  unsigned invocationIndex;
  bool reads;
  bool writes;
  DispatchStorage storage;
  StringAttr state;
};

static bool readsArgument(DispatchAccess access) {
  return access == DispatchAccess::Read || access == DispatchAccess::ReadWrite;
}

static bool writesArgument(DispatchAccess access) {
  return access == DispatchAccess::Write || access == DispatchAccess::ReadWrite;
}

static LogicalResult validateArgumentContracts(func::FuncOp dispatcher) {
  llvm::DenseMap<Value, SmallVector<ArgumentEvent>> eventsByArgument;
  unsigned invocationIndex = 0;
  for (DispatchInvokeOp invocation :
       dispatcher.getBody().front().getOps<DispatchInvokeOp>()) {
    auto target = SymbolTable::lookupNearestSymbolFrom<DispatchTargetOp>(
        invocation, invocation.getTargetAttr());
    if (!target) {
      return invocation.emitOpError()
             << "cannot resolve target '" << invocation.getTarget() << "'";
    }

    for (auto [argument, contract] :
         llvm::zip_equal(invocation.getArguments(),
                         target.getArgumentContracts().getValue())) {
      auto dispatchContract = cast<DispatchArgumentAttr>(contract);
      auto &events = eventsByArgument[argument];
      bool reads = readsArgument(dispatchContract.getAccess());
      bool writes = writesArgument(dispatchContract.getAccess());
      if (!events.empty() && events.back().invocationIndex == invocationIndex) {
        ArgumentEvent &event = events.back();
        if (event.storage != dispatchContract.getStorage() ||
            event.state != dispatchContract.getState()) {
          return invocation.emitOpError(
              "aliases one dispatcher argument with incompatible contracts");
        }
        event.reads |= reads;
        event.writes |= writes;
        continue;
      }
      events.push_back({invocation, invocationIndex, reads, writes,
                        dispatchContract.getStorage(),
                        dispatchContract.getState()});
    }
    ++invocationIndex;
  }

  for (auto &entry : eventsByArgument) {
    auto &events = entry.second;
    DispatchStorage storage = events.front().storage;
    StringAttr state = events.front().state;
    for (ArgumentEvent &event : events) {
      if (event.storage != storage || event.state != state) {
        return event.invocation.emitOpError(
            "binds one dispatcher argument with incompatible cross-image "
            "storage contracts");
      }
    }
    if (storage != DispatchStorage::Ordinary) {
      continue;
    }

    std::optional<unsigned> priorWrite;
    for (ArgumentEvent &event : events) {
      if (event.reads && priorWrite && *priorWrite < event.invocationIndex) {
        return event.invocation.emitOpError()
               << "reads an ordinary dispatcher argument written by invocation "
               << *priorWrite
               << "; declare handoff or persistent_state storage";
      }
      if (event.writes) {
        priorWrite = event.invocationIndex;
      }
    }
  }
  return success();
}

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
    return dispatcher.emitOpError("static dispatch controller must have one "
                                  "block after canonicalization");
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
  return validateArgumentContracts(dispatcher);
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
