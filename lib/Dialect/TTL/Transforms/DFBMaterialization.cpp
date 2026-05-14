// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

namespace mlir::tt::ttl {

BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp funcOp, ModuleOp moduleOp,
                                    OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();

  // Smallest size that lets the producer and consumer use different
  // slots concurrently; one slot would serialize them. Larger counts
  // work but waste L1, and no current caller has more than one in-flight
  // value (intermediate DFB at a fusion split; loop state with a pre-loop
  // store and per-iter consume/produce).
  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 2;
  auto dfbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  int32_t dfbIndex = getNextAvailableDFBIndex(moduleOp);

  // BindCBOp lives at function entry: cb_index is function-scoped and
  // finalize-dfb-indices requires that placement. Reserve/store/wait/attach
  // stay at the def site to preserve per-invocation accounting inside
  // loops and conditional branches.
  Block &body = funcOp.getBody().front();
  Operation *insertAfter = nullptr;
  for (Operation &op : body) {
    if (isa<BindCBOp>(&op)) {
      insertAfter = &op;
    } else if (insertAfter) {
      break;
    }
  }
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  } else {
    builder.setInsertionPointToStart(&body);
  }

  auto indexAttr = builder.getIndexAttr(dfbIndex);
  auto blockCountAttr = builder.getI64IntegerAttr(blockCount);
  auto bindDFB =
      BindCBOp::create(builder, loc, dfbType, indexAttr, blockCountAttr);
  bindDFB->setAttr(kCompilerAllocatedAttrName, builder.getUnitAttr());
  return bindDFB;
}

StoreOp createDFBStore(Value tensor, Value dfb, OpBuilder &builder,
                       UnitAttr accumulate) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  Location loc = tensor.getLoc();

  auto reserve = CBReserveOp::create(builder, loc, tensorType, dfb);
  return StoreOp::create(builder, loc, tensor, reserve.getResult(), accumulate);
}

AttachCBOp createDFBWaitAndAttach(Value dfb, RankedTensorType tensorType,
                                  Location loc, OpBuilder &builder) {
  auto wait = CBWaitOp::create(builder, loc, tensorType, dfb);
  return AttachCBOp::create(builder, loc, tensorType, wait.getResult(), dfb);
}

Value materializeToDFB(Value intermediate, ModuleOp moduleOp,
                       OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();

  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");

  auto funcOp = defOp->getParentOfType<func::FuncOp>();
  assert(funcOp && "intermediate must be inside a func::FuncOp");

  BindCBOp bindDFB =
      createCompilerAllocatedDFB(tensorType, loc, funcOp, moduleOp, builder);

  builder.setInsertionPointAfter(defOp);
  createDFBStore(intermediate, bindDFB.getResult(), builder);

  auto attach =
      createDFBWaitAndAttach(bindDFB.getResult(), tensorType, loc, builder);
  return attach.getResult();
}

} // namespace mlir::tt::ttl
