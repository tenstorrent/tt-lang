// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::ttl {

namespace {

FailureOr<DFBMaterializedValue>
materializeComputeResultToDFB(OpResult intermediate, Operation *consumerOp,
                              ModuleOp moduleOp, OpBuilder &builder) {
  auto computeOp = cast<ComputeOp>(intermediate.getOwner());
  auto tensorType = cast<RankedTensorType>(intermediate.getType());
  unsigned resultIndex = intermediate.getResultNumber();
  Location loc = intermediate.getLoc();

  // ComputeOp::verify requires static operand shapes, so the materialized
  // result and its compiler-allocated DFB are always statically sized.
  assert(tensorType.hasStaticShape() &&
         "ComputeOp::verify requires static operand shapes");

  if (resultIndex >= computeOp.getNumOutputs()) {
    return computeOp.emitOpError()
           << "result " << resultIndex
           << " has no matching formal output for DFB materialization";
  }

  Value selectedOutput = computeOp.getOutputs()[resultIndex];
  Value selectedCB = getAttachedCB(selectedOutput);
  if (!selectedCB) {
    return computeOp.emitOpError()
           << "result " << resultIndex
           << " has no DFB-attached formal output for materialization";
  }

  auto funcOp = computeOp->getParentOfType<func::FuncOp>();
  assert(funcOp && "compute materialization requires a func::FuncOp");

  BindCBOp bindDFB;
  {
    OpBuilder::InsertionGuard guard(builder);
    bindDFB =
        createCompilerAllocatedDFB(tensorType, loc, funcOp, moduleOp, builder);
  }

  builder.setInsertionPoint(computeOp);
  auto reserve =
      CBReserveOp::create(builder, loc, tensorType, bindDFB.getResult());
  Value init =
      tensor::EmptyOp::create(builder, loc, tensorType.getShape(),
                              tensorType.getElementType(), ValueRange{});
  auto initAttached =
      AttachCBOp::create(builder, loc, tensorType, init, bindDFB.getResult());

  SmallVector<Type> resultTypes(computeOp.getResultTypes().begin(),
                                computeOp.getResultTypes().end());
  resultTypes.push_back(tensorType);

  SmallVector<Value> outputs(computeOp.getOutputs().begin(),
                             computeOp.getOutputs().end());
  outputs.push_back(initAttached);

  SmallVector<Attribute> indexingMaps(computeOp.getIndexingMaps().begin(),
                                      computeOp.getIndexingMaps().end());
  indexingMaps.push_back(
      computeOp.getIndexingMaps()[computeOp.getNumInputs() + resultIndex]);

  auto newCompute = ComputeOp::create(
      builder, computeOp.getLoc(), TypeRange(resultTypes),
      computeOp.getInputs(), ValueRange(outputs),
      builder.getArrayAttr(indexingMaps), computeOp.getIteratorTypesAttr());

  Block &oldBody = computeOp.getBody().front();
  Block *newBody = builder.createBlock(&newCompute.getBody());
  IRMapping mapping;
  for (BlockArgument oldArg : oldBody.getArguments()) {
    BlockArgument newArg = newBody->addArgument(oldArg.getType(), loc);
    mapping.map(oldArg, newArg);
  }
  newBody->addArgument(tensorType.getElementType(), loc);

  builder.setInsertionPointToStart(newBody);
  unsigned materializedStoreCount = 0;
  for (Operation &bodyOp : oldBody.without_terminator()) {
    Operation *cloned = builder.clone(bodyOp, mapping);
    auto oldStore = dyn_cast<TileStoreOp>(bodyOp);
    if (!oldStore || getAttachedCB(oldStore.getView()) != selectedCB) {
      continue;
    }

    auto clonedStore = cast<TileStoreOp>(cloned);
    SmallVector<Value> indices(clonedStore.getIndices().begin(),
                               clonedStore.getIndices().end());
    TileStoreOp::create(builder, clonedStore.getLoc(), clonedStore.getTile(),
                        reserve.getResult(), indices,
                        clonedStore.getDstIndex());
    ++materializedStoreCount;
  }
  YieldOp::create(builder, oldBody.getTerminator()->getLoc());

  assert(materializedStoreCount > 0 &&
         "verified compute output must have a tile_store");

  Value remappedSource = newCompute.getResult(resultIndex);
  for (auto [index, result] : llvm::enumerate(computeOp.getResults())) {
    result.replaceAllUsesWith(newCompute.getResult(index));
  }
  computeOp.erase();

  // Keep the DFB push and acquisition in the same block; otherwise branch-local
  // consumers can leave an unconditional push without a matching pop.
  // TODO(#724): Relax this construction rule when trace-balance analysis can
  // prove balanced DFB occupancy across structured control flow.
  builder.setInsertionPointAfter(newCompute);
  auto push = CBPushOp::create(builder, loc, bindDFB.getResult(),
                               /*num_tiles=*/IntegerAttr{});

  builder.setInsertionPointAfter(push);
  auto attach =
      createDFBWaitAndAttach(bindDFB.getResult(), tensorType, loc, builder);
  return DFBMaterializedValue{attach.getResult(), remappedSource};
}

FailureOr<DFBMaterializedValue>
materializeTensorValueToDFB(Value intermediate, ModuleOp moduleOp,
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
  return DFBMaterializedValue{attach.getResult(), intermediate};
}

} // namespace

BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp funcOp, ModuleOp moduleOp,
                                    OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();

  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 2;
  auto dfbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  int32_t dfbIndex = getNextAvailableDFBIndex(moduleOp);

  // BindCBOp lives at function entry: cb_index is function-scoped and
  // finalize-dfb-indices requires that placement. Reserve/store/wait/attach
  // stay at the def site to preserve per-invocation accounting inside loops
  // and conditional branches.
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

FailureOr<DFBMaterializedValue> materializeToDFB(Value intermediate,
                                                 Operation *consumerOp,
                                                 ModuleOp moduleOp,
                                                 OpBuilder &builder) {
  auto result = dyn_cast<OpResult>(intermediate);
  if (result && isa<ComputeOp>(result.getOwner())) {
    return materializeComputeResultToDFB(result, consumerOp, moduleOp, builder);
  }
  return materializeTensorValueToDFB(intermediate, moduleOp, builder);
}

} // namespace mlir::tt::ttl
