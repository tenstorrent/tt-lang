// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/IR/Dominance.h"

#include <algorithm>

namespace mlir::tt::ttl {

/// Return a provisional index unique within `kernel`.
///
/// Different kernels may reuse provisional indices because module finalization
/// replaces them with disjoint physical ranges before indices are copied.
static int32_t getNextAvailableKernelDFBIndex(func::FuncOp kernel) {
  int32_t maxIndex = -1;
  kernel.walk([&](BindCBOp declaration) {
    maxIndex =
        std::max(maxIndex,
                 static_cast<int32_t>(declaration.getCbIndex().getSExtValue()));
  });
  return maxIndex + 1;
}

BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp kernel, OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();

  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 1;
  auto dfbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  // Kernel-local allocation preserves pass isolation. The module finalizer
  // replaces this provisional index before index annotations are emitted.
  int32_t dfbIndex = getNextAvailableKernelDFBIndex(kernel);

  // BindCBOp lives at kernel entry because finalize-dfb-indices requires that
  // placement. Reserve/store/wait/attach stay at the def site to preserve
  // per-invocation accounting inside loops and conditional branches.
  Block &body = kernel.getBody().front();
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
                                  blockCountAttr, /*dfbId=*/nullptr,
                                  /*tensorBacking=*/nullptr);
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

Value materializeToDFB(Value intermediate, Operation *insertionAnchor,
                       func::FuncOp kernel, OpBuilder &builder) {
  auto result = dyn_cast<OpResult>(intermediate);
  assert((!result || !isa<ComputeOp>(result.getOwner())) &&
         "compute results are materialized atomically by "
         "TTLInsertIntermediateDFBs");
  auto tensorType = cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();

  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");
  assert(insertionAnchor && "materialization requires an insertion anchor");
  assert(kernel && "intermediate must be inside a kernel");
  assert(DominanceInfo(kernel).dominates(defOp, insertionAnchor) &&
         "intermediate definition must dominate its insertion anchor");

  BindCBOp bindDFB =
      createCompilerAllocatedDFB(tensorType, loc, kernel, builder);

  builder.setInsertionPointAfter(insertionAnchor);
  createDFBStore(intermediate, bindDFB.getResult(), builder);

  return createDFBWaitAndAttach(bindDFB.getResult(), tensorType, loc, builder)
      .getResult();
}

} // namespace mlir::tt::ttl
