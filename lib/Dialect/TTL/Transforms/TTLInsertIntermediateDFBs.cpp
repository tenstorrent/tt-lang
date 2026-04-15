// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers at fusion split
// points. Tensor-level ops whose tile-level lowerings require DFB inputs
// may receive operands from fused expression chains that are not
// DFB-attached. This pass materializes those intermediates to L1 via DFBs
// so that convert-ttl-to-compute sees all required operands as CB-attached.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

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

/// Materialize a value to a compiler-allocated DFB. Inserts bind_cb,
/// cb_reserve, store, cb_wait, attach_cb. Returns the CB-attached result,
/// or failure if the maximum CB count would be exceeded.
FailureOr<Value> materializeToDFB(Value intermediate, ModuleOp moduleOp,
                                  OpBuilder &builder) {
  auto tensorType = mlir::cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();
  MLIRContext *ctx = builder.getContext();

  // Intra-thread push/wait requires double-buffering so the packer and
  // unpacker can operate on different buffer halves simultaneously.
  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 2;
  auto cbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  int32_t dfbIndex = getNextAvailableDFBIndex(moduleOp);
  // TODO: Move this check to TTLFinalizeDFBIndices after DFB index reuse
  // is implemented (see docs/development/DFB_Index_Reuse.md). Reuse will
  // reduce the physical DFB count, so checking here over-reports.
  if (dfbIndex >= kMaxCircularBuffers) {
    return intermediate.getDefiningOp()->emitError()
           << "compiler-allocated DFB would exceed the maximum of "
           << kMaxCircularBuffers << " circular buffers (need index "
           << dfbIndex << ")";
  }

  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");
  builder.setInsertionPointAfter(defOp);

  auto indexAttr = builder.getIndexAttr(dfbIndex);
  auto blockCountAttr = builder.getI64IntegerAttr(blockCount);
  auto bindCB =
      BindCBOp::create(builder, loc, cbType, indexAttr, blockCountAttr);
  bindCB->setAttr(kCompilerAllocatedAttrName, builder.getUnitAttr());

  auto reserve =
      CBReserveOp::create(builder, loc, tensorType, bindCB.getResult());

  StoreOp::create(builder, loc, intermediate, reserve.getResult(),
                  /*accumulate=*/nullptr);

  // cb_push is inserted by ttl-insert-cb-sync which runs after this pass.

  auto wait = CBWaitOp::create(builder, loc, tensorType, bindCB.getResult());

  auto attachWait = AttachCBOp::create(builder, loc, tensorType,
                                       wait.getResult(), bindCB.getResult());

  return attachWait.getResult();
}

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return;
    }

    // Track values already materialized to avoid duplicate DFBs when
    // multiple DFBInputOpInterface ops consume the same intermediate.
    llvm::DenseMap<Value, Value> materialized;
    OpBuilder builder(funcOp.getContext());

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {
        Value operand = op->getOperand(idx);

        if (getAttachedCB(operand)) {
          continue;
        }

        // Reuse an existing materialization for a different consumer.
        if (auto iter = materialized.find(operand);
            iter != materialized.end()) {
          op->setOperand(idx, iter->second);
          continue;
        }

        auto replacement = materializeToDFB(operand, moduleOp, builder);
        if (failed(replacement)) {
          signalPassFailure();
          return;
        }

        // Replace only this specific operand. Elementwise consumers of
        // the same value retain the original SSA value and fuse with
        // the producer in a single compute block.
        op->setOperand(idx, *replacement);

        materialized[operand] = *replacement;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
