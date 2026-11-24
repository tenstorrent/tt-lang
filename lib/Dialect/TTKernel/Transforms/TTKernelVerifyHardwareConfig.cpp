// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ConfigFlow.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelConfigEffects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

#define DEBUG_TYPE "ttkernel-verify-hardware-config"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELVERIFYHARDWARECONFIG
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// MathInit state on entry to an operation. `writer` and `otherWriter` are the
/// earliest writes that explain the state, kept so a conflicting merge can
/// still name both sides.
struct MathInitState {
  std::optional<ttk::MathInitDescriptor> descriptor;
  Operation *writer = nullptr;
  Operation *otherWriter = nullptr;
};

int writerOrder(Operation *op, const llvm::DenseMap<Operation *, int> &order) {
  if (!op) {
    return std::numeric_limits<int>::max();
  }
  return order.lookup(op);
}

MathInitState withWriters(std::optional<ttk::MathInitDescriptor> descriptor,
                          ArrayRef<Operation *> writers,
                          const llvm::DenseMap<Operation *, int> &order) {
  SmallVector<Operation *, 4> unique;
  for (Operation *writer : writers) {
    if (writer && !llvm::is_contained(unique, writer)) {
      unique.push_back(writer);
    }
  }
  llvm::sort(unique, [&](Operation *lhs, Operation *rhs) {
    return writerOrder(lhs, order) < writerOrder(rhs, order);
  });
  MathInitState state;
  state.descriptor = std::move(descriptor);
  if (!unique.empty()) {
    state.writer = unique[0];
  }
  if (unique.size() > 1) {
    state.otherWriter = unique[1];
  }
  return state;
}

struct MathInitSlot {
  using State = MathInitState;

  explicit MathInitSlot(const llvm::DenseMap<Operation *, int> &order)
      : order(order) {}

  static State unknown() { return {}; }

  State join(const State &lhs, const State &rhs) const {
    bool agree = lhs.descriptor.has_value() == rhs.descriptor.has_value() &&
                 (!lhs.descriptor || *lhs.descriptor == *rhs.descriptor);
    return withWriters(
        agree ? lhs.descriptor : std::nullopt,
        {lhs.writer, lhs.otherWriter, rhs.writer, rhs.otherWriter}, order);
  }

  std::optional<State> getWrite(Operation *op) const {
    ttk::MathInitEffects effects = ttk::getMathInitEffects(op);
    if (!effects.write) {
      return std::nullopt;
    }
    return withWriters(effects.write->descriptor, {op}, order);
  }

  const llvm::DenseMap<Operation *, int> &order;
};

} // namespace

static void attachWriterNotes(InFlightDiagnostic &diagnostic,
                              const MathInitState &incoming) {
  for (Operation *writer : {incoming.writer, incoming.otherWriter}) {
    if (!writer) {
      continue;
    }
    ttk::MathInitEffects effects = ttk::getMathInitEffects(writer);
    bool reset = !effects.write || !effects.write->descriptor;
    diagnostic.attachNote(writer->getLoc())
        << (reset ? "MATH configuration reset here" : "configured here");
  }
}

static LogicalResult verifyRead(Operation *op, const MathInitState &incoming) {
  ttk::MathInitEffects effects = ttk::getMathInitEffects(op);
  if (!effects.read) {
    return success();
  }
  if (!effects.read->descriptor) {
    return op->emitOpError(
        "reads the MATH configuration without declaring the required init");
  }
  const ttk::MathInitDescriptor &required = *effects.read->descriptor;
  if (incoming.descriptor && *incoming.descriptor == required) {
    return success();
  }

  InFlightDiagnostic diagnostic = op->emitOpError()
                                  << "requires MATH configuration from '"
                                  << required.init.getValue() << "'";
  if (!incoming.descriptor) {
    diagnostic << " but it is not established on every incoming path";
  } else if (incoming.descriptor->init != required.init) {
    diagnostic << " but '" << incoming.descriptor->init.getValue()
               << "' is configured";
  } else {
    diagnostic << " but it is configured with different operands or "
                  "attributes";
  }
  attachWriterNotes(diagnostic, incoming);
  return diagnostic;
}

namespace {

struct TTKernelVerifyHardwareConfigPass
    : public impl::TTKernelVerifyHardwareConfigBase<
          TTKernelVerifyHardwareConfigPass> {
  void runOnOperation() override {
    bool hadError = false;
    getOperation()->walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal()) {
        return;
      }
      llvm::DenseMap<Operation *, int> order;
      int nextIndex = 0;
      funcOp.walk([&](Operation *op) { order[op] = nextIndex++; });
      MathInitSlot slot(order);
      ConfigFlow<MathInitSlot> flow(slot);
      flow.run(funcOp.getBody(), MathInitSlot::unknown(),
               [&](Operation *op, const MathInitState &incoming) {
                 if (failed(verifyRead(op, incoming))) {
                   hadError = true;
                 }
               });
    });
    if (hadError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
