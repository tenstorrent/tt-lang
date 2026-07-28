// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

namespace mlir::tt::ttl {

namespace {

FailureOr<DFBMaterializedValue>
materializeTensorValueToDFB(Value intermediate, func::FuncOp funcOp,
                            OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();

  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");
  assert(funcOp && "intermediate must be inside a func::FuncOp");

  BindCBOp bindDFB =
      createCompilerAllocatedDFB(tensorType, loc, funcOp, builder);

  builder.setInsertionPointAfter(defOp);
  createDFBStore(intermediate, bindDFB.getResult(), builder);

  auto attach =
      createDFBWaitAndAttach(bindDFB.getResult(), tensorType, loc, builder);
  return DFBMaterializedValue{attach.getResult(), intermediate};
}

} // namespace

BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp funcOp, OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();

  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 1;
  auto dfbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  // Function-local allocation preserves pass isolation. The module finalizer
  // replaces this provisional index before index annotations are emitted.
  int32_t dfbIndex = getNextAvailableDFBIndex(funcOp.getOperation());

  // BindCBOp lives at function entry because finalize-dfb-indices requires that
  // placement. Reserve/store/wait/attach stay at the def site to preserve
  // per-invocation accounting inside loops and conditional branches.
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
  auto bindDFB = BindCBOp::create(builder, loc, dfbType, indexAttr,
                                  blockCountAttr, /*dfbId=*/nullptr);
  bindDFB->setAttr(kCompilerAllocatedAttrName, builder.getUnitAttr());
  return bindDFB;
}

StoreOp createDFBStore(Value tensor, Value dfb, OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  Location loc = tensor.getLoc();

  auto reserve = CBReserveOp::create(builder, loc, tensorType, dfb);
  return StoreOp::create(builder, loc, tensor, reserve.getResult(),
                         /*accumulate=*/nullptr);
}

AttachCBOp createDFBWaitAndAttach(Value dfb, RankedTensorType tensorType,
                                  Location loc, OpBuilder &builder) {
  auto wait = CBWaitOp::create(builder, loc, tensorType, dfb);
  return AttachCBOp::create(builder, loc, tensorType, wait.getResult(), dfb);
}

FailureOr<DFBMaterializedValue>
materializeToDFB(Value intermediate, func::FuncOp funcOp, OpBuilder &builder) {
  auto result = dyn_cast<OpResult>(intermediate);
  assert((!result || !isa<ComputeOp>(result.getOwner())) &&
         "compute results are materialized atomically by "
         "TTLInsertIntermediateDFBs");
  return materializeTensorValueToDFB(intermediate, funcOp, builder);
}

} // namespace mlir::tt::ttl
