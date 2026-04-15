// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated scratch dataflow buffers at fusion split points.
// Tensor-level ops whose tile-level lowerings require CB inputs (reduce,
// bcast, matmul, transpose) may receive operands from fused expression
// chains that are not CB-attached. This pass materializes those
// intermediates to L1 via scratch DFBs so that convert-ttl-to-compute
// sees all operands as CB-attached.
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

/// Materialize a value to a DFB. Inserts bind_cb, cb_reserve,
/// store, cb_wait, attach_cb. Returns the CB-attached cb_wait result.
/// The storeOp output parameter receives the inserted store so the caller
/// can exclude it from replaceAllUsesWith.
static Value materializeToDFB(Value intermediate, ModuleOp moduleOp,
                              OpBuilder &builder, StoreOp &insertedStore) {
  auto tensorType = mlir::cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();
  MLIRContext *ctx = builder.getContext();

  // Build the CB type: shape from tensor, block_count = 1.
  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 1;
  auto cbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  // Allocate the next available DFB index.
  int32_t dfbIndex = getNextAvailableDFBIndex(moduleOp);

  // Insert after the defining op of the intermediate value.
  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");
  builder.setInsertionPointAfter(defOp);

  // bind_cb with compiler_allocated marker.
  auto indexAttr = builder.getIndexAttr(dfbIndex);
  auto blockCountAttr = builder.getI64IntegerAttr(blockCount);
  auto bindCB = BindCBOp::create(builder, loc, cbType, indexAttr,
                                 blockCountAttr);
  bindCB->setAttr(kCompilerAllocatedAttrName, builder.getUnitAttr());

  // cb_reserve -> tensor view.
  auto reserve =
      CBReserveOp::create(builder, loc, tensorType, bindCB.getResult());

  // store the intermediate to the reserved view.
  // The store verifier requires the view to come directly from cb_reserve.
  insertedStore = StoreOp::create(builder, loc, intermediate,
                                  reserve.getResult(),
                                  /*accumulate=*/nullptr);

  // cb_wait -> tensor view (consumer side).
  auto wait =
      CBWaitOp::create(builder, loc, tensorType, bindCB.getResult());

  // attach_cb on the wait result.
  auto attachWait = AttachCBOp::create(builder, loc, tensorType,
                                       wait.getResult(), bindCB.getResult());

  // Register the scratch DFB in the module attribute.
  int32_t numTiles = 1;
  for (int64_t dim : shape) {
    numTiles *= static_cast<int32_t>(dim);
  }
  registerScratchDFB(moduleOp, dfbIndex, numTiles, elementType, blockCount);

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

    // Track values already materialized to avoid duplicate DFBs.
    llvm::DenseMap<Value, Value> materialized;
    OpBuilder builder(funcOp.getContext());

    // Collect ops that implement DFBInputOpInterface.
    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) {
      candidates.push_back(op);
    });

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {

        Value operand = op->getOperand(idx);

        // Already CB-attached (user-declared DFB or prior materialization).
        if (getAttachedCB(operand)) {
          continue;
        }

        // Already materialized by this pass for a different consumer.
        if (auto iter = materialized.find(operand);
            iter != materialized.end()) {
          op->setOperand(idx, iter->second);
          continue;
        }

        // Materialize: insert scratch DFB.
        StoreOp insertedStore;
        Value replacement =
            materializeToDFB(operand, moduleOp, builder, insertedStore);

        // Replace all uses of the original value with the CB-attached
        // result, EXCEPT the store we just inserted (to avoid a cycle).
        SmallPtrSet<Operation *, 1> excludeSet;
        excludeSet.insert(insertedStore);
        operand.replaceAllUsesExcept(replacement, excludeSet);

        materialized[operand] = replacement;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
