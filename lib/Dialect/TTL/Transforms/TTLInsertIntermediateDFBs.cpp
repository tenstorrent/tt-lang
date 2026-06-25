// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers at fusion split
// points. Tensor-level ops whose tile-level lowerings require DFB inputs may
// receive operands from ttl.compute results that are not DFB-attached. This
// pass materializes those compute results as extra compute outputs so the
// remaining consumers can be converted after attachment.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-intermediate-dfbs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERMEDIATEDFBS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  using TTLInsertIntermediateDFBsBase::TTLInsertIntermediateDFBsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    // When compiler DFBs are disabled, verify that no operations require
    // them and emit an actionable error if any do.
    if (!enable) {
      for (DFBInputOpInterface dfbInputOp : candidates) {
        Operation *op = dfbInputOp.getOperation();
        auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

        for (unsigned idx : requiredIndices) {
          Value operand = op->getOperand(idx);
          if (getAttachedCB(operand)) {
            continue;
          }
          op->emitOpError("operand #")
              << idx
              << " requires a DFB-attached value but compiler-allocated DFBs "
                 "are disabled (--no-ttl-compiler-dfbs); either enable "
                 "compiler DFBs or store the intermediate to a user-declared "
                 "DFB before this operation";
          signalPassFailure();
          return;
        }
      }
      return;
    }

    OpBuilder builder(funcOp.getContext());
    llvm::DenseMap<Value, Value> materialized;

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {
        Value operand = op->getOperand(idx);

        if (getAttachedCB(operand)) {
          continue;
        }

        if (auto iter = materialized.find(operand);
            iter != materialized.end()) {
          op->setOperand(idx, iter->second);
          continue;
        }

        FailureOr<DFBMaterializedValue> materialization =
            materializeToDFB(operand, op, moduleOp, builder);
        if (failed(materialization)) {
          signalPassFailure();
          return;
        }

        // Replace only this specific operand. Elementwise consumers of
        // the same value retain the original SSA value and fuse with
        // the producer in a single compute block.
        op->setOperand(idx, materialization->materialized);

        materialized[materialization->source] = materialization->materialized;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
