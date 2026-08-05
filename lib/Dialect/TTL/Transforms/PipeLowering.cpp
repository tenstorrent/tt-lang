// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

static constexpr int64_t kPipeAddressWordBytes = 4;
static constexpr int64_t kPipeSramScratchAlignmentBytes = 32;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static CircularBufferType getTTLCBType(Value cb) {
  if (auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(cb.getType())) {
    return ttlCbTy;
  }
  if (auto castOp = cb.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      if (auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(
              castOp.getInputs()[0].getType())) {
        return ttlCbTy;
      }
    }
  }
  return nullptr;
}

static Value makeZeroI32(Location loc, ConversionPatternRewriter &rewriter) {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

struct PipeSourceKey {
  int64_t srcX;
  int64_t srcY;

  bool operator==(const PipeSourceKey &other) const {
    return srcX == other.srcX && srcY == other.srcY;
  }
};

} // namespace mlir::tt::ttl

namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttl::PipeSourceKey> {
  using Key = mlir::tt::ttl::PipeSourceKey;
  static unsigned getHashValue(const Key &sourceKey) {
    return hash_combine(sourceKey.srcX, sourceKey.srcY);
  }
  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

} // namespace llvm

namespace mlir::tt::ttl {

static PipeSourceKey getPipeSourceKey(PipeType pipeType) {
  return {pipeType.getSrcX(), pipeType.getSrcY()};
}

static FailureOr<PipeTransferCreateOp>
getPipeTransferCreate(Operation *op, Value transfer,
                      ValueOriginAnalysis &analysis) {
  FailureOr<PipeTransferCreateOp> maybeCreateOp =
      findPipeTransferCreateForTransfer(analysis, transfer);
  if (failed(maybeCreateOp)) {
    return op->emitError() << op->getName()
                           << " requires every possible transfer value to "
                              "derive from the same ttl.pipe_transfer.create";
  }
  return *maybeCreateOp;
}

static const PipeResourceInfo &
lookupPipeResourceInfo(Operation *protocolOp,
                       const PipeResourcePlan &pipeResourcePlan) {
  auto it = pipeResourcePlan.resources.find(protocolOp);
  assert(it != pipeResourcePlan.resources.end() &&
         "active pipe transfer must have a resource allocation");
  return it->second;
}

static int64_t alignTo(int64_t value, int64_t alignment) {
  assert(alignment > 0 && "alignment must be positive");
  return ((value + alignment - 1) / alignment) * alignment;
}

/// Count tensor arguments because TTKernel common runtime args list tensor
/// buffer addresses before computed DFB bases and compiler-managed resources.
static int64_t getNumTensorFunctionArgs(FuncOp func) {
  int64_t numTensorArgs = 0;
  for (BlockArgument argument : func.getArguments()) {
    if (llvm::isa<RankedTensorType>(argument.getType())) {
      ++numTensorArgs;
    }
  }
  return numTensorArgs;
}

static int64_t getNumComputedAddressRuntimeArgs(FuncOp func) {
  // Resource planning records the sorted receiver DFB list before lowering
  // computes common runtime argument indices.
  auto dfbIndices = func->getAttrOfType<DenseI32ArrayAttr>(
      kPipeComputedAddressDFBIndicesAttrName);
  return dfbIndices ? static_cast<int64_t>(dfbIndices.size()) : 0;
}

/// Pipe kernels receive tensor buffer addresses, computed receiver DFB bases,
/// and then compiler-managed pipe resources as common runtime arguments.
/// [Device 2.0] Keep this as a resource-plan lookup so the final device API
/// lowering can replace common-arg plumbing without changing pipe semantics.
static int64_t getPipeRuntimeCommonArgIndex(FuncOp func,
                                            int64_t pipeRuntimeArgIndex) {
  return getNumTensorFunctionArgs(func) +
         getNumComputedAddressRuntimeArgs(func) + pipeRuntimeArgIndex;
}

static int64_t getPipeRuntimeCommonArgIndex(Operation *op,
                                            int64_t pipeRuntimeArgIndex) {
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe op is not inside a function");
  return getPipeRuntimeCommonArgIndex(func, pipeRuntimeArgIndex);
}

static Value buildPipeRuntimeCommonArg(Location loc, OpBuilder &builder,
                                       int64_t commonArgIndex) {
  auto argIndex = arith::ConstantIndexOp::create(builder, loc, commonArgIndex);
  return ttk::GetCommonArgValOp::create(builder, loc, builder.getI32Type(),
                                        argIndex)
      .getResult();
}

static Value buildLocalSemaphoreAddress(Location loc, OpBuilder &builder,
                                        int64_t semaphoreIndex) {
  Value semaphoreIndexValue =
      arith::ConstantIndexOp::create(builder, loc, semaphoreIndex);
  return ttk::GetSemaphoreOp::create(builder, loc, semaphoreIndexValue)
      .getResult();
}

/// Return the first pipe-resource runtime arg index used for GlobalSemaphore
/// counter addresses.
static int64_t
getFirstPipeGlobalSemaphoreArgOffset(const PipeResourcePlan &info) {
  // GlobalSemaphore addresses follow the optional SRAM scratch base in the
  // common runtime args built by python/ttl/kernel_runner.py.
  return info.sramScratch.bytes > 0 ? 1 : 0;
}

/// Build the L1 address for any compiler-managed PipeNet counter.
static Value buildPipeCounterAddress(Location loc, FuncOp func,
                                     PipeCounterInfo counter,
                                     const PipeResourcePlan &pipeResourcePlan,
                                     OpBuilder &builder) {
  // [Device 2.0] This should become a typed semaphore-object lookup when the
  // device API exposes Semaphore/GlobalSemaphore objects directly.
  switch (counter.getStorage()) {
  case PipeCounterStorage::LocalSemaphore:
    return buildLocalSemaphoreAddress(loc, builder, counter.getIndex());
  case PipeCounterStorage::GlobalSemaphore: {
    int64_t argIndex = getPipeRuntimeCommonArgIndex(
        func, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
                  counter.getIndex());
    return buildPipeRuntimeCommonArg(loc, builder, argIndex);
  }
  }
  llvm_unreachable("unknown pipe counter storage");
}

static Value buildPipeCounterPtr(Location loc, FuncOp func,
                                 PipeCounterInfo counter,
                                 const PipeResourcePlan &pipeResourcePlan,
                                 OpBuilder &builder) {
  auto l1PtrTy = ttk::L1AddrPtrType::get(builder.getContext(), 32);
  Value address =
      buildPipeCounterAddress(loc, func, counter, pipeResourcePlan, builder);
  return ttk::CastToL1PtrOp::create(builder, loc, l1PtrTy, address).getResult();
}

/// Add a static byte offset to an L1 address without changing the address
/// representation.
static Value addByteOffset(Location loc, Value baseAddress, int64_t byteOffset,
                           ConversionPatternRewriter &rewriter) {
  if (byteOffset == 0) {
    return baseAddress;
  }
  auto offsetValue =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                rewriter.getI32IntegerAttr(byteOffset));
  return arith::AddIOp::create(rewriter, loc, baseAddress, offsetValue)
      .getResult();
}

/// Source-core address-table entry selected for one transfer allocation unit.
/// The common arg contains the host-allocated SRAM scratch buffer address;
/// byteOffset selects this transfer's 32-bit receiver-published address slot.
struct AddressTableInfo {
  int64_t scratchRuntimeCommonArgIndex;
  int64_t byteOffset = 0;
};

/// Record the scratch common-arg index with the per-transfer SRAM offset from
/// the resource plan.
static AddressTableInfo
getAddressTableInfo(Operation *op, const PipeResourceInfo &pipeResource) {
  assert(pipeResource.addressStorage.mode ==
             PipeAddressMode::ReceiverPublishedAddressTable &&
         "address-table info requested for computed-address pipe");
  assert(pipeResource.addressStorage.sramAddressTable.has_value() &&
         "receiver-published-address pipe missing address-table storage");
  int64_t scratchArgIndex = getPipeRuntimeCommonArgIndex(op, 0);
  return AddressTableInfo{
      scratchArgIndex,
      pipeResource.addressStorage.sramAddressTable->byteOffset};
}

/// Build the L1 address of this transfer's source-core address-table slot.
static Value buildAddressTableAddress(Location loc,
                                      const AddressTableInfo &info,
                                      ConversionPatternRewriter &rewriter) {
  Value scratchBase = buildPipeRuntimeCommonArg(
      loc, rewriter, info.scratchRuntimeCommonArgIndex);
  return addByteOffset(loc, scratchBase, info.byteOffset, rewriter);
}

/// Load the receiver-published destination DFB address from this pipe's
/// source-core SRAM address-table entry.
static Value
buildAddressTableDestinationAddress(Location loc, const AddressTableInfo &info,
                                    ConversionPatternRewriter &rewriter) {
  Value tableAddress = buildAddressTableAddress(loc, info, rewriter);
  // [Device 2.0] Address tables are compiler-managed SRAM state; only this
  // final load should depend on raw L1 pointer operations.
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  auto tablePtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, tableAddress);
  auto zeroI32 = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
  return ttk::LoadFromL1Op::create(rewriter, loc, rewriter.getI32Type(),
                                   tablePtr, zeroI32)
      .getResult();
}

/// Find the slot counter allocated during resource planning. Missing state is
/// a pass-ordering bug because computed-address sends are planned before
/// conversion patterns mutate the IR.
static Value lookupComputedAddressCounter(
    PipeTransferSendOp op, int64_t counterIndex,
    const PipeComputedAddressCounterMap &computedAddressCounters) {
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  auto funcIt = computedAddressCounters.find(senderFunc);
  assert(funcIt != computedAddressCounters.end() &&
         "sender function missing computed-address counters");
  auto counterIt = funcIt->second.find(counterIndex);
  assert(counterIt != funcIt->second.end() &&
         "computed-address counter missing from sender function");
  return counterIt->second;
}

/// Compute the receiver DFB destination address selected for this send. A
/// transfer that executes at most once uses `initialSlot` directly. A transfer
/// that can repeat with a nonzero stride uses a sender-local counter for
/// `slot(i)`.
static Value buildComputedReceiverDFBDestinationAddress(
    PipeTransferSendOp op, Location loc, const PipeComputedAddressInfo &info,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter) {
  Value baseAddress =
      buildPipeRuntimeCommonArg(loc, rewriter, info.baseRuntimeCommonArgIndex);
  if (!info.usesDynamicSlotCounter()) {
    int64_t byteOffset =
        info.initialSlot * info.blockStrideBytes + info.staticTileByteOffset;
    return addByteOffset(loc, baseAddress, byteOffset, rewriter);
  }

  Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value slotCounter = lookupComputedAddressCounter(
      op, *info.dynamicSlotCounterIndex, computedAddressCounters);
  Value currentSlot =
      memref::LoadOp::create(rewriter, loc, slotCounter, ValueRange{zeroIdx});
  Value blockStrideBytes =
      arith::ConstantIntOp::create(rewriter, loc, info.blockStrideBytes, 32);
  Value blockByteOffset =
      arith::MulIOp::create(rewriter, loc, currentSlot, blockStrideBytes);
  Value receiverAddress =
      arith::AddIOp::create(rewriter, loc, baseAddress, blockByteOffset);
  receiverAddress =
      addByteOffset(loc, receiverAddress, info.staticTileByteOffset, rewriter);

  Value repeatStride =
      arith::ConstantIntOp::create(rewriter, loc, info.repeatStride, 32);
  Value blockCount =
      arith::ConstantIntOp::create(rewriter, loc, info.blockCount, 32);
  Value nextSlotUnwrapped =
      arith::AddIOp::create(rewriter, loc, currentSlot, repeatStride);
  Value nextSlot =
      arith::RemUIOp::create(rewriter, loc, nextSlotUnwrapped, blockCount);
  memref::StoreOp::create(rewriter, loc, nextSlot, slotCounter,
                          ValueRange{zeroIdx});
  return receiverAddress;
}

struct ReceiverPublishedAddressInfo {
  Value receiverDFB;
  ttcore::TileType tileType;
};

static FailureOr<ReceiverPublishedAddressInfo>
getReceiverPublishedAddressInfo(Operation *op, Value dst,
                                ConversionPatternRewriter &rewriter) {
  Value receiverDFB = getAttachedCB(dst);
  if (!receiverDFB) {
    return rewriter.notifyMatchFailure(
        op, "pipe receive destination is not attached to a DFB");
  }

  auto receiverDFBType = getTTLCBType(receiverDFB);
  if (!receiverDFBType) {
    return rewriter.notifyMatchFailure(op, "failed to get receiver DFB type");
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>(receiverDFBType.getElementType());
  if (!tileType) {
    return rewriter.notifyMatchFailure(
        op, "receiver DFB element type must be tile");
  }

  return ReceiverPublishedAddressInfo{receiverDFB, tileType};
}

/// Compute the exact DFB address selected by ttl.copy(pipe, dst). Receivers
/// publish this address so senders do not have to infer receiver DFB state.
static Value
buildReceiverPublishedAddress(Value dst, Location loc,
                              const ReceiverPublishedAddressInfo &info,
                              ConversionPatternRewriter &rewriter) {
  auto receiverCBConverted =
      utils::convertTTLCBToTTKernel(info.receiverDFB, rewriter, loc);
  assert(succeeded(receiverCBConverted) &&
         "getTTLCBType guarantees a convertible receiver DFB");

  auto receiverWritePtr =
      ttk::GetWritePtrOp::create(rewriter, loc, *receiverCBConverted);
  Value publishedAddress = receiverWritePtr;
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value localTileIndex = zeroIdx;
  Value globalTileIndex =
      utils::addSliceOffset(dst, localTileIndex, rewriter, loc);
  if (globalTileIndex == localTileIndex) {
    return publishedAddress;
  }

  auto tileOffsetI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), globalTileIndex);
  auto pageSizeBytes = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(info.tileType.getSizeBytes()));
  auto byteOffset =
      arith::MulIOp::create(rewriter, loc, tileOffsetI32, pageSizeBytes);
  return arith::AddIOp::create(rewriter, loc, receiverWritePtr, byteOffset)
      .getResult();
}

static void
emitLocalReceiverAddressPublish(Location loc, Value tableAddress,
                                Value publishedAddress,
                                ConversionPatternRewriter &rewriter) {
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  Value tablePtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, tableAddress);
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  ttk::StoreToL1Op::create(rewriter, loc, publishedAddress, tablePtr, zero);
}

static void emitRemoteReceiverAddressPublish(
    Location loc, Value sourceX, Value sourceY, Value tableAddress,
    Value publishedAddress, Value nocVal, ConversionPatternRewriter &rewriter) {
  auto byteEnableAll = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
  ttk::NocInlineDwWriteOp::create(rewriter, loc, sourceX, sourceY, tableAddress,
                                  publishedAddress, byteEnableAll, nocVal);
}

/// Publish locally when the receiver is the source because inline NoC writes do
/// not update their issuing core's SRAM.
static void emitReceiverAddressPublish(Location loc, PipeType pipeType,
                                       Value sourceXTranslated,
                                       Value sourceYTranslated,
                                       Value tableAddress,
                                       Value publishedAddress, Value nocVal,
                                       ConversionPatternRewriter &rewriter) {
  if (!pipeType.srcInDstRange()) {
    emitRemoteReceiverAddressPublish(loc, sourceXTranslated, sourceYTranslated,
                                     tableAddress, publishedAddress, nocVal,
                                     rewriter);
    return;
  }
  if (pipeType.hasSingleReceiver()) {
    emitLocalReceiverAddressPublish(loc, tableAddress, publishedAddress,
                                    rewriter);
    return;
  }

  Value currentX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value currentY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value sourceX =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
  Value sourceY =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
  Value xMatches = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, currentX, sourceX);
  Value yMatches = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, currentY, sourceY);
  Value receiverIsSource =
      arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
  auto localPublish = scf::IfOp::create(rewriter, loc, receiverIsSource,
                                        /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&localPublish.getThenRegion().front());
    emitLocalReceiverAddressPublish(loc, tableAddress, publishedAddress,
                                    rewriter);
    rewriter.setInsertionPointToStart(&localPublish.getElseRegion().front());
    emitRemoteReceiverAddressPublish(loc, sourceXTranslated, sourceYTranslated,
                                     tableAddress, publishedAddress, nocVal,
                                     rewriter);
  }
  rewriter.setInsertionPointAfter(localPublish);
}

//===----------------------------------------------------------------------===//
// Receiver post sequence counter initialization
//===----------------------------------------------------------------------===//

void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &postSequenceCounters) {
  llvm::MapVector<FuncOp, SmallVector<PipeCounterInfo>> countersByFunc;
  for (const auto &[protocolOp, resource] : pipeResourcePlan.resources) {
    auto postOp = dyn_cast<PipeTransferPostOp>(protocolOp);
    if (!postOp) {
      continue;
    }
    FuncOp func = postOp->getParentOfType<FuncOp>();
    assert(func && "pipe transfer post must be inside a function");
    SmallVector<PipeCounterInfo> &counters = countersByFunc[func];
    if (!llvm::is_contained(counters, resource.completion.counter)) {
      counters.push_back(resource.completion.counter);
    }
  }

  for (auto &[func, counters] : countersByFunc) {
    // These entry-block values dominate posts nested in receiver control flow.
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto memrefTy = MemRefType::get({1}, builder.getI32Type());
    auto i32Ty = builder.getI32Type();
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantOp::create(builder, loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
    auto &countersForFunc = postSequenceCounters[func];
    llvm::sort(counters, [](PipeCounterInfo lhs, PipeCounterInfo rhs) {
      return std::make_pair(lhs.getStorage(), lhs.getIndex()) <
             std::make_pair(rhs.getStorage(), rhs.getIndex());
    });
    for (PipeCounterInfo counterInfo : counters) {
      auto counter = memref::AllocaOp::create(builder, loc, memrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, counter,
                              ValueRange{zeroIdx});
      countersForFunc.push_back(
          PipeCounterProgress{counterInfo, counter.getResult()});
    }
  }
}

void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters) {
  for (const auto &initializationEntry :
       pipeResourcePlan.computedAddressCounterInitializations) {
    func::FuncOp func = initializationEntry.first;
    const SmallVector<PipeComputedAddressCounterInitInfo> &initializations =
        initializationEntry.second;
    SmallVector<PipeComputedAddressCounterInitInfo> sortedInitializations(
        initializations);
    llvm::sort(sortedInitializations,
               [](const PipeComputedAddressCounterInitInfo &lhs,
                  const PipeComputedAddressCounterInitInfo &rhs) {
                 return lhs.counterIndex < rhs.counterIndex;
               });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefTy = MemRefType::get({1}, builder.getI32Type());
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    auto &perFuncCounters = computedAddressCounters[func];
    for (const PipeComputedAddressCounterInitInfo &init :
         sortedInitializations) {
      // Entry-block allocation dominates loop-carried sends while keeping the
      // slot state private to the sender kernel.
      auto counter = memref::AllocaOp::create(builder, loc, counterMemrefTy);
      Value initialSlot =
          arith::ConstantIntOp::create(builder, loc, init.initialSlot, 32);
      memref::StoreOp::create(builder, loc, initialSlot, counter,
                              ValueRange{zeroIdx});
      perFuncCounters[init.counterIndex] = counter.getResult();
    }
  }
}

static FailureOr<Value>
lookupPipeCounterProgress(const PipeCounterProgressMap *progress, FuncOp func,
                          PipeCounterInfo counter) {
  if (!progress) {
    return failure();
  }
  auto funcIt = progress->find(func);
  if (funcIt == progress->end()) {
    return failure();
  }
  auto progressIt =
      llvm::find_if(funcIt->second, [&](const PipeCounterProgress &entry) {
        return entry.counter == counter;
      });
  if (progressIt == funcIt->second.end()) {
    return failure();
  }
  return progressIt->value;
}

/// Lower CB -> Pipe copy: write source DFB data to the selected destination
/// address, then signal arrival.
LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, bool isConsumerCB,
    ValueOriginAnalysis &analysis, const PipeResourcePlan &pipeResourcePlan,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
    rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
    return success();
  }
  FailureOr<PipeTransferCreateOp> maybeCreateOp =
      getPipeTransferCreate(op.getOperation(), op.getTransfer(), analysis);
  if (failed(maybeCreateOp)) {
    return failure();
  }
  PipeTransferCreateOp createOp = *maybeCreateOp;
  auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
  const PipeResourceInfo &pipeResource =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan);
  PipeCompletionInfo completionInfo = pipeResource.completion;
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  assert(senderFunc && "pipe transfer send must be inside a function");
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto cbType = getTTLCBType(srcCB);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto cbShape = cbType.getShape();

  auto elementType = cbType.getElementType();
  auto tileType = llvm::dyn_cast<ttcore::TileType>(elementType);
  if (!tileType) {
    return rewriter.notifyMatchFailure(op, "CB element type must be tile");
  }
  int64_t pageSizeBytes = tileType.getSizeBytes();

  int64_t dstStartX = pipeType.getDstStartX();
  int64_t dstStartY = pipeType.getDstStartY();
  int64_t dstEndX = pipeType.getDstEndX();
  int64_t dstEndY = pipeType.getDstEndY();
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) &&
         "getTTLCBType guarantees a convertible source DFB");

  int64_t nocIdx = getNocIndex(op);
  Value nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                           rewriter.getI8IntegerAttr(nocIdx));

  int64_t expectedReceiverPosts =
      isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
  Value senderReadyCounterAddr = buildPipeCounterAddress(
      loc, senderFunc, pipeResource.readyCounter, pipeResourcePlan, rewriter);
  auto senderReadyCounterPtr = ttk::CastToL1PtrOp::create(
      rewriter, loc, l1PtrTy, senderReadyCounterAddr);
  auto expectedReadyCount = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(expectedReceiverPosts));
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderReadyCounterPtr,
                               expectedReadyCount);
  auto readyCounterResetValue =
      arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderReadyCounterPtr,
                                 readyCounterResetValue);

  SmallVector<int64_t> cbBounds(cbShape.begin(), cbShape.end());
  int64_t cbNumTiles = 1;
  for (int64_t dimension : cbBounds) {
    cbNumTiles *= dimension;
  }
  // Producer source address is at the source DFB's write_ptr (data is staged
  // there before push_back); consumer source address is at its read_ptr.
  Value srcPtrIdx;
  if (isConsumerCB) {
    auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbReadPtr);
  } else {
    auto srcWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, srcWritePtr);
  }

  // Hardware multicast destination coordinates use translated NOC coords.
  auto dstStartXLogical =
      arith::ConstantIndexOp::create(rewriter, loc, dstStartX);
  auto dstStartYLogical =
      arith::ConstantIndexOp::create(rewriter, loc, dstStartY);
  auto dstEndXLogical = arith::ConstantIndexOp::create(rewriter, loc, dstEndX);
  auto dstEndYLogical = arith::ConstantIndexOp::create(rewriter, loc, dstEndY);

  // NoC operations require translated coordinates.
  auto dstStartXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, dstStartXLogical);
  auto dstStartYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, dstStartYLogical);
  auto dstEndXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, dstEndXLogical);
  auto dstEndYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, dstEndYLogical);
  Value mcastStartXVal = dstStartXVal;
  Value mcastStartYVal = dstStartYVal;
  Value mcastEndXVal = dstEndXVal;
  Value mcastEndYVal = dstEndYVal;
  // TTKernel multicast ops follow tt-metal's NOC1 convention: callers pass
  // the rectangle with start/end reversed after coordinate translation.
  if (nocIdx == 1) {
    std::swap(mcastStartXVal, mcastEndXVal);
    std::swap(mcastStartYVal, mcastEndYVal);
  }

  auto numDestsVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numDests));

  // Transfer the entire block in one NoC write. Tiles are contiguous in the
  // DFB, and destination DFB layout is uniform across nodes, so lowering sends
  // all tiles at once instead of one per tile.
  int64_t totalSizeBytes = cbNumTiles * pageSizeBytes;
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(totalSizeBytes));

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, srcPtrIdx);

  Value dstAddr;
  if (pipeResource.addressStorage.usesComputedReceiverDFB()) {
    assert(pipeResource.addressStorage.computedAddress.has_value() &&
           "computed pipe missing computed-address info");
    dstAddr = buildComputedReceiverDFBDestinationAddress(
        op, loc, *pipeResource.addressStorage.computedAddress,
        computedAddressCounters, rewriter);
  } else {
    AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
    dstAddr =
        buildAddressTableDestinationAddress(loc, addressTableInfo, rewriter);
  }

  // TODO(ttl): Select unicast or multicast from a compiler optimization over
  // the transfer plan instead of directly preserving the user's tt-lang syntax.
  if (pipeType.hasSingleReceiver()) {
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr,
                                 ValueRange{dstStartXVal, dstStartYVal},
                                 ValueRange{}, dstAddr, totalSizeVal, nocVal);
  } else {
    if (pipeType.srcInDstRange()) {
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeVal, numDestsVal, mcastStartXVal,
          mcastStartYVal, mcastEndXVal, mcastEndYVal, dstAddr, nocVal,
          /*linked=*/nullptr);
    } else {
      ttk::NocAsyncWriteMulticastOp::create(
          rewriter, loc, srcAddr, totalSizeVal, numDestsVal, mcastStartXVal,
          mcastStartYVal, mcastEndXVal, mcastEndYVal, dstAddr, nocVal,
          /*linked=*/nullptr);
    }
  }

  // Wait for payload writes to complete before signaling receiver completion.
  // Without this barrier, the receiver may wake up before all data arrives.
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  Value receiverCompletionCounterAddr = buildPipeCounterAddress(
      loc, senderFunc, completionInfo.counter, pipeResourcePlan, rewriter);

  if (pipeType.hasSingleReceiver()) {
    // Point-to-point increments the destination receiver-completion counter.
    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto receiverCompletionNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal, dstStartYVal,
                                  receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, receiverCompletionNocAddr.getResult(),
        completionIncrement, nocVal, /*posted=*/BoolAttr());
  } else {
    // Collective increments every receiver-completion counter. The receiver
    // pairs this with a cumulative wait_min threshold.
    // Hardware multicast excludes the sender, so num_dests counts only remote
    // receivers. TT-Metal has no multicast loopback increment, so a source that
    // is also a receiver requires a separate local increment.
    int64_t numRemoteDests = pipeType.srcInDstRange() ? numDests - 1 : numDests;
    auto remoteReceiverCount = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numRemoteDests));

    auto remoteReceiverCompletionMcastNocAddr =
        ttk::GetNocMulticastAddrOp::create(
            rewriter, loc, mcastStartXVal, mcastStartYVal, mcastEndXVal,
            mcastEndYVal, receiverCompletionCounterAddr, nocVal);

    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, remoteReceiverCompletionMcastNocAddr.getResult(),
        completionIncrement, remoteReceiverCount, nocVal,
        /*posted=*/BoolAttr());

    if (pipeType.srcInDstRange()) {
      auto srcXLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
      auto srcYLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
      auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
          rewriter, loc, indexTy, srcXLogical);
      auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
          rewriter, loc, indexTy, srcYLogical);
      auto localReceiverCompletionNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, srcXTranslated, srcYTranslated,
          receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
  }

  // Point-to-point and collective completion signals use non-posted atomics.
  // The send ttl.wait lowers to a no-op, so this barrier is the only flush
  // before kernel exit; without it receivers can observe stale counts.
  ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    ValueOriginAnalysis &analysis,
                                    const PipeCounterProgressMap &counters,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
    auto token = UnrealizedConversionCastOp::create(
        rewriter, loc, op.getToken().getType(), ValueRange{});
    rewriter.replaceOp(op, token.getResult(0));
    return success();
  }
  FailureOr<PipeTransferCreateOp> maybeCreateOp =
      getPipeTransferCreate(op.getOperation(), op.getTransfer(), analysis);
  if (failed(maybeCreateOp)) {
    return failure();
  }
  PipeTransferCreateOp createOp = *maybeCreateOp;
  auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
  const PipeResourceInfo &pipeResource =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan);
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer post must be inside a function");
  FailureOr<Value> maybeSequenceCounter = lookupPipeCounterProgress(
      &counters, func, pipeResource.completion.counter);
  if (failed(maybeSequenceCounter)) {
    op.emitError("pipe receive post has no sequence counter for its completion "
                 "counter");
    return failure();
  }
  Value sequenceCounter = *maybeSequenceCounter;

  int64_t nocIdx = getNocIndex(op);
  auto indexTy = rewriter.getIndexType();

  // Preflight the only fallible validation before emitting any ops, so a match
  // failure leaves no partially-built IR for the conversion driver to roll
  // back.
  bool usesComputedReceiverDFB =
      pipeResource.addressStorage.usesComputedReceiverDFB();
  std::optional<ReceiverPublishedAddressInfo> maybePublishedAddressInfo;
  if (!usesComputedReceiverDFB) {
    FailureOr<ReceiverPublishedAddressInfo> info =
        getReceiverPublishedAddressInfo(op, dst, rewriter);
    if (failed(info)) {
      return failure();
    }
    maybePublishedAddressInfo = *info;
  }

  Value nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                           rewriter.getI8IntegerAttr(nocIdx));

  auto srcXLogical =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
  auto srcYLogical =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
  auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, srcXLogical);
  auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, srcYLogical);

  if (!usesComputedReceiverDFB) {
    AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
    Value publishedAddress = buildReceiverPublishedAddress(
        dst, loc, *maybePublishedAddressInfo, rewriter);
    Value tableAddress =
        buildAddressTableAddress(loc, addressTableInfo, rewriter);
    emitReceiverAddressPublish(loc, pipeType, srcXTranslated, srcYTranslated,
                               tableAddress, publishedAddress, nocVal,
                               rewriter);
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  Value senderReadyCounterAddr = buildPipeCounterAddress(
      loc, func, pipeResource.readyCounter, pipeResourcePlan, rewriter);
  auto senderReadyCounterNocAddr =
      ttk::GetNocAddrOp::create(rewriter, loc, srcXTranslated, srcYTranslated,
                                senderReadyCounterAddr, nocVal);
  auto readyCounterIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);
  ttk::NocSemaphoreIncOp::create(rewriter, loc,
                                 senderReadyCounterNocAddr.getResult(),
                                 readyCounterIncrement, nocVal,
                                 /*posted=*/BoolAttr());

  auto zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto previousSequence = memref::LoadOp::create(rewriter, loc, sequenceCounter,
                                                 ValueRange{zeroIndex});
  auto oneI32 = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(1));
  auto tokenSequence =
      arith::AddIOp::create(rewriter, loc, previousSequence, oneI32);
  memref::StoreOp::create(rewriter, loc, tokenSequence, sequenceCounter,
                          ValueRange{zeroIndex});
  rewriter.replaceOp(op, tokenSequence.getResult());
  return success();
}

/// Lower the receiver completion wait using the posted token's sequence.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
    rewriter.eraseOp(op);
    return success();
  }
  auto loc = op.getLoc();
  PipeCompletionInfo completionInfo =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan).completion;

  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer wait must be inside a function");
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  Value receiverCompletionCounterPtr = buildPipeCounterPtr(
      loc, func, completionInfo.counter, pipeResourcePlan, rewriter);

  ttk::SemaphoreWaitMinOp::create(rewriter, loc, receiverCompletionCounterPtr,
                                  tokenSequence);

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pipe conditional operation lowering patterns
//===----------------------------------------------------------------------===//

namespace {

// Replace `op` with an `scf.if(cond)` whose then-region is the original
// body. The body's `ttl.yield` terminator is dropped — `scf.if`'s own
// yield closes the region.
template <typename Op>
static void lowerToScfIf(Op op, Value cond,
                         ConversionPatternRewriter &rewriter) {
  auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), cond,
                                /*withElseRegion=*/false);
  Block &srcBlock = op.getBody().front();
  Block &thenBlock = ifOp.getThenRegion().front();
  if (Operation *terminator = srcBlock.getTerminator();
      terminator && mlir::isa<YieldOp>(terminator)) {
    rewriter.eraseOp(terminator);
  }
  rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());
  rewriter.eraseOp(op);
}

static Value buildSrcMatch(OpBuilder &builder, Location loc, Value coreX,
                           Value coreY, PipeType pipeType) {
  auto sourceX =
      arith::ConstantIndexOp::create(builder, loc, pipeType.getSrcX());
  auto sourceY =
      arith::ConstantIndexOp::create(builder, loc, pipeType.getSrcY());
  auto matchX = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      coreX, sourceX);
  auto matchY = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      coreY, sourceY);
  return arith::AndIOp::create(builder, loc, matchX, matchY);
}

static Value buildDstMatch(OpBuilder &builder, Location loc, Value coreX,
                           Value coreY, PipeType pipeType) {
  int64_t minX = std::min(pipeType.getDstStartX(), pipeType.getDstEndX());
  int64_t maxX = std::max(pipeType.getDstStartX(), pipeType.getDstEndX());
  int64_t minY = std::min(pipeType.getDstStartY(), pipeType.getDstEndY());
  int64_t maxY = std::max(pipeType.getDstStartY(), pipeType.getDstEndY());
  auto minXConst = arith::ConstantIndexOp::create(builder, loc, minX);
  auto maxXConst = arith::ConstantIndexOp::create(builder, loc, maxX);
  auto minYConst = arith::ConstantIndexOp::create(builder, loc, minY);
  auto maxYConst = arith::ConstantIndexOp::create(builder, loc, maxY);
  auto geMinX = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                      coreX, minXConst);
  auto leMaxX = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                      coreX, maxXConst);
  auto geMinY = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                      coreY, minYConst);
  auto leMaxY = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                      coreY, maxYConst);
  auto inRangeX = arith::AndIOp::create(builder, loc, geMinX, leMaxX);
  auto inRangeY = arith::AndIOp::create(builder, loc, geMinY, leMaxY);
  return arith::AndIOp::create(builder, loc, inRangeX, inRangeY);
}

struct IfSrcLowering : OpConversionPattern<IfSrcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfSrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    Value isSrc = buildSrcMatch(rewriter, loc, coreX, coreY, pipeType);
    lowerToScfIf(op, isSrc, rewriter);
    return success();
  }
};

struct IfDstLowering : OpConversionPattern<IfDstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfDstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    Value isDst = buildDstMatch(rewriter, loc, coreX, coreY, pipeType);
    lowerToScfIf(op, isDst, rewriter);
    return success();
  }
};

// Lower a per-pipe-role predicate op to the OR of per-pipe matches in the
// named PipeNet. `roleBuilder` produces the i1 match for one pipe.
template <typename Op>
static LogicalResult lowerRolePredicate(
    Op op, ConversionPatternRewriter &rewriter,
    const PipeNetIndex &pipeNetIndex,
    llvm::function_ref<Value(OpBuilder &, Location, Value, Value, PipeType)>
        roleBuilder) {
  auto loc = op.getLoc();
  int64_t netId = op.getPipeNetId();
  auto it = pipeNetIndex.find(netId);
  if (it == pipeNetIndex.end() || it->second.empty()) {
    return op->emitError() << op->getName() << " references unknown PipeNet "
                           << netId;
  }
  auto coreX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  auto coreY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value result;
  for (const PipeInfo &pipeInfo : it->second) {
    Value match = roleBuilder(rewriter, loc, coreX, coreY, pipeInfo.pipeType);
    result = result ? Value(arith::OrIOp::create(rewriter, loc, result, match))
                    : match;
  }
  rewriter.replaceOp(op, result);
  return success();
}

template <typename Op>
struct IsRoleLoweringBase : OpConversionPattern<Op> {
  IsRoleLoweringBase(const TypeConverter &tc, MLIRContext *ctx,
                     const PipeNetIndex *index)
      : OpConversionPattern<Op>(tc, ctx), pipeNetIndex(index) {}
  const PipeNetIndex *pipeNetIndex;
};

struct IsSrcLowering : IsRoleLoweringBase<IsSrcOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsSrcOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, *pipeNetIndex, buildSrcMatch);
  }
};

struct IsDstLowering : IsRoleLoweringBase<IsDstOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsDstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, *pipeNetIndex, buildDstMatch);
  }
};

struct IsActiveLowering : IsRoleLoweringBase<IsActiveOp> {
  using IsRoleLoweringBase::IsRoleLoweringBase;
  LogicalResult
  matchAndRewrite(IsActiveOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(
        op, rewriter, *pipeNetIndex,
        [](OpBuilder &builder, Location loc, Value coreX, Value coreY,
           PipeType pipeType) {
          Value isSrc = buildSrcMatch(builder, loc, coreX, coreY, pipeType);
          Value isDst = buildDstMatch(builder, loc, coreX, coreY, pipeType);
          return Value(arith::OrIOp::create(builder, loc, isSrc, isDst));
        });
  }
};

struct CreatePipeLowering : OpConversionPattern<CreatePipeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreatePipeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // CreatePipeOp produces a pipe type whose parameters carry the coordinate
    // info; coordinates are encoded into generated code by if_src/if_dst.
    // Replace with an unrealized cast so uses in nested regions (if_src /
    // if_dst bodies) that may be processed in a different order still resolve.
    // The unrealized cast preserves the type for downstream patterns.
    auto cast = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), ValueRange{});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

} // namespace

void buildPipeNetIndex(ModuleOp mod, PipeNetIndex &index) {
  using PipeKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<int64_t, llvm::SmallSetVector<PipeKey, 4>> seenPerNet;
  // Role-predicate lowering needs all pipes for a PipeNet to build the
  // `is_src`, `is_dst`, and `is_active` predicates. Walk create ops after Pipe
  // Transfer IR expansion so duplicate static pipes from cloned regions merge
  // into one predicate entry.
  mod.walk([&](PipeTransferCreateOp op) {
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());
    int64_t netId = pipeType.getPipeNetId();
    PipeKey key{pipeType.getSrcX(),      pipeType.getSrcY(),
                pipeType.getDstStartX(), pipeType.getDstStartY(),
                pipeType.getDstEndX(),   pipeType.getDstEndY()};
    PipeTransferContract contract = getPipeTransferContract(op);
    if (seenPerNet[netId].insert(key)) {
      index[netId].push_back(PipeInfo{pipeType, contract});
      return;
    }
    if (!isCollectiveTransfer(contract)) {
      return;
    }
    for (PipeInfo &pipeInfo : index[netId]) {
      PipeType existingType = pipeInfo.pipeType;
      PipeKey existingKey{
          existingType.getSrcX(),      existingType.getSrcY(),
          existingType.getDstStartX(), existingType.getDstStartY(),
          existingType.getDstEndX(),   existingType.getDstEndY()};
      if (existingKey == key) {
        pipeInfo.transferContract = PipeTransferContract::Collective;
        break;
      }
    }
  });
}

namespace {

/// Allocation unit for all resources owned by one transfer definition.
///
/// One send and its corresponding receiver posts share an address mechanism
/// and sender-ready counter. Each receiver wait uses the completion semaphore
/// assigned to the same unit.
struct PipeTransferAllocationUnit {
  PipeTransferNodeId transferNodeId = 0;
  Operation *sendOp = nullptr;
  /// Send, receiver-post, and receiver-wait operations for this transfer.
  SmallVector<Operation *> protocolOps;

  /// Logical pipe whose source node owns this unit's rendezvous resources.
  PipeKey pipe;

  PipeType pipeType;

  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;

  /// Stable tie-breaker for deterministic allocation.
  int64_t ordinal = 0;

  /// Conservative post-to-send lifetime for source-node rendezvous resources.
  OperationLiveInterval interval;

  /// Assigned first-fit color within the source node's allocation group.
  std::size_t resourceColor = 0;

  /// Completion-counter color; disjoint receiver sets may share one color.
  std::optional<int64_t> maybeCompletionCounterColor;

  /// Deterministic order used by first-fit interval coloring.
  bool operator<(const PipeTransferAllocationUnit &rhs) const {
    return std::make_tuple(interval.startOrdinal, pipe.srcX, pipe.srcY,
                           pipe.pipeNetId, pipe.dstStartX, pipe.dstStartY,
                           pipe.dstEndX, pipe.dstEndY, ordinal) <
           std::make_tuple(rhs.interval.startOrdinal, rhs.pipe.srcX,
                           rhs.pipe.srcY, rhs.pipe.pipeNetId,
                           rhs.pipe.dstStartX, rhs.pipe.dstStartY,
                           rhs.pipe.dstEndX, rhs.pipe.dstEndY, rhs.ordinal);
  }
};

} // namespace

static bool pipeTransferIntervalsOverlap(const PipeTransferAllocationUnit &lhs,
                                         const PipeTransferAllocationUnit &rhs,
                                         const DominanceInfo &dominanceInfo) {
  return intervalsOverlap(lhs.interval, rhs.interval, dominanceInfo);
}

static FailureOr<SmallVector<PipeTransferAllocationUnit>>
collectPipeTransferAllocationUnits(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, const DominanceInfo &dominanceInfo,
    const PostDominanceInfo &postDominanceInfo,
    llvm::SmallPtrSetImpl<Operation *> &staticallyInactiveOps) {
  SmallVector<PipeTransferAllocationUnit> units;
  llvm::DenseMap<Operation *, int64_t> operationOrdinals;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> waitOpsByPost;
  int64_t nextOperationOrdinal = 0;
  WalkResult provenanceWalkResult = mod.walk([&](Operation *op) {
    if (isa<PipeTransferPostOp, PipeTransferSendOp>(op)) {
      operationOrdinals[op] = nextOperationOrdinal++;
      if (!pipeGraph.getPipeTransferNodeForProtocolOp(op)) {
        staticallyInactiveOps.insert(op);
      }
      return WalkResult::advance();
    }
    if (auto waitOp = dyn_cast<PipeTransferWaitOp>(op)) {
      ArrayRef<Operation *> possiblePosts =
          transferIndex.getPossibleReceivePosts(waitOp);
      if (possiblePosts.size() != 1) {
        waitOp.emitError() << "requires exactly one possible receiver post; "
                              "found "
                           << possiblePosts.size();
        return WalkResult::interrupt();
      }
      Operation *postOp = possiblePosts.front();
      if (!pipeGraph.getPipeTransferNodeForProtocolOp(postOp)) {
        staticallyInactiveOps.insert(op);
        return WalkResult::advance();
      }
      waitOpsByPost[postOp].push_back(op);
    }
    return WalkResult::advance();
  });
  if (provenanceWalkResult.wasInterrupted()) {
    return failure();
  }

  units.reserve(pipeGraph.getPipeTransferNodes().size());
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    assert(transferNode.sendOp &&
           "pipe transfer graph node must have a send operation");
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    PipeTransferCreateOp createOp =
        transferIndex.getTransferCreate(sendOp.getOperation());

    PipeTransferAllocationUnit unit;
    unit.transferNodeId = transferNode.id;
    unit.sendOp = sendOp.getOperation();
    unit.pipe = transferNode.pipe;
    unit.pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
    unit.transferContract = transferNode.transferContract;
    unit.ordinal = static_cast<int64_t>(transferNode.id);
    unit.protocolOps.push_back(sendOp.getOperation());
    updateIntervalEnd(unit.interval, sendOp.getOperation(), dominanceInfo);
    for (Operation *postOp : transferNode.receiverPostOps) {
      auto ordinalIt = operationOrdinals.find(postOp);
      assert(ordinalIt != operationOrdinals.end() &&
             "receiver post is missing an operation ordinal");
      unit.protocolOps.push_back(postOp);
      auto waitIt = waitOpsByPost.find(postOp);
      if (waitIt != waitOpsByPost.end()) {
        unit.protocolOps.append(waitIt->second.begin(), waitIt->second.end());
      }
      updateIntervalStart(unit.interval, postOp, ordinalIt->second,
                          dominanceInfo);
    }
    finalizeInterval(unit.interval, dominanceInfo, postDominanceInfo);
    units.push_back(std::move(unit));
  }
  return units;
}

using SourceColorMap =
    llvm::MapVector<PipeSourceKey, SmallVector<SmallVector<std::size_t>>>;

static SourceColorMap
assignLiveIntervalColors(MutableArrayRef<PipeTransferAllocationUnit> units,
                         const DominanceInfo &dominanceInfo) {
  llvm::MapVector<PipeSourceKey, SmallVector<std::size_t>> unitIndicesBySource;
  for (std::size_t index = 0, size = units.size(); index < size; ++index) {
    unitIndicesBySource[getPipeSourceKey(units[index].pipeType)].push_back(
        index);
  }

  SourceColorMap colorUsersBySource;
  for (auto &entry : unitIndicesBySource) {
    SmallVector<SmallVector<std::size_t>> colorUsers =
        assignGreedyIntervalColors<std::size_t>(
            entry.second,
            [&](std::size_t lhsIndex, std::size_t rhsIndex) {
              return std::less<PipeTransferAllocationUnit>()(units[lhsIndex],
                                                             units[rhsIndex]);
            },
            [&](std::size_t lhsIndex, std::size_t rhsIndex) {
              return pipeTransferIntervalsOverlap(
                  units[lhsIndex], units[rhsIndex], dominanceInfo);
            });

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      for (std::size_t unitIndex : indexedColor.value()) {
        units[unitIndex].resourceColor = indexedColor.index();
      }
    }

    colorUsersBySource.insert({entry.first, std::move(colorUsers)});
  }

  return colorUsersBySource;
}

/// Return whether two closed destination rectangles share a receiver.
static bool receiverSetsOverlap(const PipeKey &lhs, const PipeKey &rhs) {
  return lhs.dstStartX <= rhs.dstEndX && rhs.dstStartX <= lhs.dstEndX &&
         lhs.dstStartY <= rhs.dstEndY && rhs.dstStartY <= lhs.dstEndY;
}

/// Reuse a counter color only across disjoint physical receiver sets.
/// Transfers sharing a receiver need distinct state because either send may
/// complete first.
static int64_t allocateCompletionCounterColor(
    const PipeKey &pipe,
    SmallVectorImpl<SmallVector<PipeKey>> &pipesByCounterColor) {
  for (auto indexedPipes : llvm::enumerate(pipesByCounterColor)) {
    bool overlapsAssignedReceiverSet =
        llvm::any_of(indexedPipes.value(), [&](const PipeKey &allocatedPipe) {
          return receiverSetsOverlap(pipe, allocatedPipe);
        });
    if (!overlapsAssignedReceiverSet) {
      indexedPipes.value().push_back(pipe);
      return static_cast<int64_t>(indexedPipes.index());
    }
  }
  pipesByCounterColor.push_back(SmallVector<PipeKey>{pipe});
  return static_cast<int64_t>(pipesByCounterColor.size() - 1);
}

static std::optional<FuncOp>
getSingleSenderFunc(const PipeTransferAllocationUnit &unit) {
  FuncOp senderFunc = unit.sendOp->getParentOfType<FuncOp>();
  return senderFunc ? std::optional<FuncOp>(senderFunc) : std::nullopt;
}

static int64_t getReceiverDFBBlockStrideBytes(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.dfbType.getElementsPerBlock() * tileType.getSizeBytes();
}

static int64_t getReceiverDFBStaticByteOffset(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.staticTileOffset * tileType.getSizeBytes();
}

/// Return metadata only when the sender can compute every receiver address.
/// The caller uses receiver-published addresses when this proof fails.
static std::optional<PipeComputedAddressInfo>
getComputedAddressInfo(const PipeReceiverEndpoint &receiverEndpoint) {
  const ReceiverDFBInfo &receiverInfo = receiverEndpoint.receiverDFBInfo;
  if (!receiverInfo.hasStaticTileOffset) {
    return std::nullopt;
  }
  if (!llvm::isa<ttcore::TileType>(receiverInfo.dfbType.getElementType())) {
    return std::nullopt;
  }
  // Static receiver addresses are derived from the pipe graph's physical slot
  // assignment. Non-pipe DFB traffic can advance the hardware ring without a
  // pipe post, so computed addressing requires the graph to prove that the
  // receiver stream contains only pipe-delivered blocks.
  const ReceiverAddressSequenceProof &sequence =
      receiverEndpoint.addressSequence;
  if (sequence.getKind() == ReceiverAddressSequenceProofKind::FullyDynamic) {
    return std::nullopt;
  }
  const ReceiverAddressRecurrence &recurrence = *sequence.recurrence;
  if (recurrence.blockCount <= 0 || recurrence.initialSlot < 0 ||
      recurrence.initialSlot >= recurrence.blockCount ||
      recurrence.repeatStride < 0 ||
      recurrence.repeatStride >= recurrence.blockCount ||
      recurrence.blockCount != receiverInfo.blockCount) {
    return std::nullopt;
  }
  int64_t blockStrideBytes = getReceiverDFBBlockStrideBytes(receiverInfo);
  int64_t staticTileByteOffset = getReceiverDFBStaticByteOffset(receiverInfo);
  if (blockStrideBytes <= 0 || !llvm::isInt<32>(blockStrideBytes) ||
      !llvm::isInt<32>(staticTileByteOffset) ||
      !llvm::isInt<32>(recurrence.initialSlot) ||
      !llvm::isInt<32>(recurrence.repeatStride) ||
      !llvm::isInt<32>(receiverInfo.blockCount)) {
    return std::nullopt;
  }
  int64_t maxBlockByteOffset =
      (receiverInfo.blockCount - 1) * blockStrideBytes + staticTileByteOffset;
  if (!llvm::isInt<32>(maxBlockByteOffset)) {
    return std::nullopt;
  }
  return PipeComputedAddressInfo{receiverInfo.dfbIndex,
                                 /*baseRuntimeCommonArgIndex=*/0,
                                 recurrence.initialSlot,
                                 recurrence.repeatStride,
                                 receiverInfo.blockCount,
                                 blockStrideBytes,
                                 staticTileByteOffset,
                                 std::nullopt};
}

/// Computed-address facts indexed by transfer allocation unit before resource
/// coloring builds the final plan.
struct ComputedAddressPlan {
  llvm::DenseMap<std::size_t, PipeComputedAddressInfo> infoByUnitIndex;
  llvm::MapVector<FuncOp, SmallVector<PipeComputedAddressCounterInitInfo>>
      counterInitializations;
};

static ComputedAddressPlan
buildComputedAddressPlan(ModuleOp mod,
                         MutableArrayRef<PipeTransferAllocationUnit> units,
                         const PipeGraph &pipeGraph) {
  ComputedAddressPlan plan;

  /// One transfer whose recurrence can be materialized by its sender.
  struct Candidate {
    std::size_t unitIndex = 0;
    FuncOp senderFunc;
    PipeComputedAddressInfo computedAddress;
  };
  SmallVector<Candidate> candidates;
  llvm::MapVector<FuncOp, llvm::SmallSetVector<int64_t, 4>> dfbIndicesByFunc;

  for (auto indexedUnit : llvm::enumerate(units)) {
    PipeTransferAllocationUnit &unit = indexedUnit.value();
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(unit.transferNodeId);
    const PipeReceiverEndpoint *receiverEndpoint =
        pipeGraph.getProvenReceiverAddressEndpoint(transferNode.id);
    if (!receiverEndpoint) {
      continue;
    }
    const ReceiverDFBInfo &receiverInfo = receiverEndpoint->receiverDFBInfo;
    std::optional<PipeComputedAddressInfo> maybeComputedAddress =
        getComputedAddressInfo(*receiverEndpoint);
    if (!maybeComputedAddress) {
      continue;
    }
    std::optional<FuncOp> maybeSenderFunc = getSingleSenderFunc(unit);
    if (!maybeSenderFunc) {
      continue;
    }
    candidates.push_back(Candidate{indexedUnit.index(), *maybeSenderFunc,
                                   *maybeComputedAddress});
    dfbIndicesByFunc[*maybeSenderFunc].insert(receiverInfo.dfbIndex);
  }

  if (candidates.empty()) {
    return plan;
  }

  OpBuilder builder(mod.getContext());
  llvm::DenseMap<FuncOp, SmallVector<int64_t>> sortedDFBIndicesByFunc;
  for (auto &[func, dfbSet] : dfbIndicesByFunc) {
    SmallVector<int64_t> sortedDFBIndices(dfbSet.begin(), dfbSet.end());
    llvm::sort(sortedDFBIndices);
    sortedDFBIndicesByFunc[func] = sortedDFBIndices;

    SmallVector<int32_t> dfbAttrs =
        llvm::map_to_vector(sortedDFBIndices, [](int64_t dfbIndex) {
          return static_cast<int32_t>(dfbIndex);
        });
    func->setAttr(kPipeComputedAddressDFBIndicesAttrName,
                  builder.getDenseI32ArrayAttr(dfbAttrs));
  }

  llvm::MapVector<FuncOp, int64_t> nextDynamicSlotCounterIndexByFunc;
  for (const Candidate &candidate : candidates) {
    FuncOp senderFunc = candidate.senderFunc;
    const SmallVector<int64_t> &dfbIndices = sortedDFBIndicesByFunc[senderFunc];
    PipeComputedAddressInfo computedAddress = candidate.computedAddress;
    auto dfbIt = llvm::find(dfbIndices, computedAddress.receiverDFBIndex);
    assert(dfbIt != dfbIndices.end() && "candidate DFB missing from func list");
    computedAddress.baseRuntimeCommonArgIndex =
        getNumTensorFunctionArgs(senderFunc) +
        std::distance(dfbIndices.begin(), dfbIt);

    const PipeTransferAllocationUnit &unit = units[candidate.unitIndex];
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(unit.transferNodeId);
    const PipeReceiverEndpoint *receiverEndpoint =
        pipeGraph.getProvenReceiverAddressEndpoint(transferNode.id);
    assert(receiverEndpoint &&
           "computed-address unit missing receiver address proof");
    const ReceiverAddressSequenceProof &sequence =
        receiverEndpoint->addressSequence;
    bool canRepeat =
        sequence.getKind() != ReceiverAddressSequenceProofKind::KnownCount ||
        *sequence.executionCount > 1;
    if (canRepeat && computedAddress.repeatStride != 0) {
      int64_t counterIndex = nextDynamicSlotCounterIndexByFunc[senderFunc]++;
      computedAddress.dynamicSlotCounterIndex = counterIndex;
      plan.counterInitializations[senderFunc].push_back(
          PipeComputedAddressCounterInitInfo{counterIndex,
                                             computedAddress.initialSlot});
    }
    plan.infoByUnitIndex[candidate.unitIndex] = computedAddress;
  }

  return plan;
}

// Compact the per-source colors whose units need a resource into a dense
// 0..N-1 index range, keyed by original color index. Returns the compacted map
// and the maximum compacted count across sources.
template <typename PredT>
static std::pair<
    llvm::MapVector<PipeSourceKey, llvm::DenseMap<std::size_t, int64_t>>,
    int64_t>
compactColors(const SourceColorMap &colorUsersBySource,
              PredT unitNeedsResource) {
  llvm::MapVector<PipeSourceKey, llvm::DenseMap<std::size_t, int64_t>>
      compactedBySource;
  int64_t maxPerSource = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    int64_t nextColor = 0;
    llvm::DenseMap<std::size_t, int64_t> &compacted =
        compactedBySource[sourceKey];
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      if (llvm::any_of(indexedColor.value(), unitNeedsResource)) {
        compacted[indexedColor.index()] = nextColor++;
      }
    }
    maxPerSource = std::max(maxPerSource, nextColor);
  }
  return {std::move(compactedBySource), maxPerSource};
}

LogicalResult buildPipeResourcePlan(ModuleOp mod,
                                    const PipeTransferIndex &transferIndex,
                                    const PipeGraph &pipeGraph,
                                    PipeResourcePlan &info,
                                    bool enableComputedAddresses) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, transferIndex, pipeGraph,
                                         dominanceInfo, postDominanceInfo,
                                         info.staticallyInactiveOps);
  if (failed(maybeUnits)) {
    return failure();
  }
  SmallVector<PipeTransferAllocationUnit> &units = *maybeUnits;
  SourceColorMap colorUsersBySource =
      assignLiveIntervalColors(units, dominanceInfo);
  ComputedAddressPlan computedAddressPlan;
  if (enableComputedAddresses) {
    computedAddressPlan = buildComputedAddressPlan(mod, units, pipeGraph);
  }
  info.computedAddressCounterInitializations =
      computedAddressPlan.counterInitializations;

  SmallVector<SmallVector<PipeKey>> pipesByCompletionCounterColor;
  for (PipeTransferAllocationUnit &unit : units) {
    unit.maybeCompletionCounterColor = allocateCompletionCounterColor(
        unit.pipe, pipesByCompletionCounterColor);
  }
  int64_t firstSourceLocalReadyCounterSemIdx =
      static_cast<int64_t>(pipesByCompletionCounterColor.size());

  auto [readyColorBySourceColor, maxReadyCountersPerSource] =
      compactColors(colorUsersBySource, [](std::size_t) { return true; });

  // Use one ready-counter kind per kernel so host allocation has one compact
  // descriptor layout.
  bool useGlobalReadyCounters =
      firstSourceLocalReadyCounterSemIdx + maxReadyCountersPerSource >
      kMaxHardwareSemaphoreIds;

  llvm::MapVector<PipeSourceKey, SmallVector<int64_t>> globalIndexBySourceColor;
  int64_t nextGlobalSemaphoreIndex = 0;
  if (useGlobalReadyCounters) {
    for (const auto &[sourceKey, readyColors] : readyColorBySourceColor) {
      SmallVector<int64_t> &indices = globalIndexBySourceColor[sourceKey];
      indices.reserve(readyColors.size());
      for (std::size_t color = 0, colorCount = readyColors.size();
           color < colorCount; ++color) {
        indices.push_back(nextGlobalSemaphoreIndex++);
      }
    }
  }

  auto [addressColorBySourceColor, maxAddressColorsPerSource] =
      compactColors(colorUsersBySource, [&](std::size_t unitIndex) {
        return computedAddressPlan.infoByUnitIndex.find(unitIndex) ==
               computedAddressPlan.infoByUnitIndex.end();
      });
  int64_t maxAddressTableBytes =
      maxAddressColorsPerSource * kPipeAddressWordBytes;

  for (auto indexedUnit : llvm::enumerate(units)) {
    const PipeTransferAllocationUnit &unit = indexedUnit.value();
    assert(unit.maybeCompletionCounterColor &&
           "pipe transfer is missing a completion counter color");
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    auto sourceIt = readyColorBySourceColor.find(sourceKey);
    assert(sourceIt != readyColorBySourceColor.end());
    auto colorIt = sourceIt->second.find(unit.resourceColor);
    assert(colorIt != sourceIt->second.end());
    int64_t readyColor = colorIt->second;
    PipeCounterInfo readyCounter = PipeCounterInfo::localSemaphore(
        firstSourceLocalReadyCounterSemIdx + readyColor);
    if (useGlobalReadyCounters) {
      auto globalIt = globalIndexBySourceColor.find(sourceKey);
      assert(globalIt != globalIndexBySourceColor.end());
      assert(readyColor < static_cast<int64_t>(globalIt->second.size()));
      readyCounter =
          PipeCounterInfo::globalSemaphore(globalIt->second[readyColor]);
    }

    auto computedIt =
        computedAddressPlan.infoByUnitIndex.find(indexedUnit.index());
    PipeAddressStorageInfo addressStorage;
    if (computedIt != computedAddressPlan.infoByUnitIndex.end()) {
      addressStorage =
          PipeAddressStorageInfo::computedReceiverDFB(computedIt->second);
    } else {
      auto sourceIt = addressColorBySourceColor.find(sourceKey);
      assert(sourceIt != addressColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      addressStorage = PipeAddressStorageInfo::receiverPublishedAddressTable(
          PipeSramAddressTableInfo{colorIt->second * kPipeAddressWordBytes});
    }
    PipeResourceInfo pipeResource{
        unit.pipe,
        unit.transferContract,
        PipeCompletionInfo{
            PipeCounterInfo::localSemaphore(*unit.maybeCompletionCounterColor)},
        readyCounter,
        addressStorage,
    };
    for (Operation *protocolOp : unit.protocolOps) {
      auto [resourceIt, inserted] =
          info.resources.insert({protocolOp, pipeResource});
      assert((inserted || resourceIt->second.pipe == pipeResource.pipe) &&
             "pipe protocol operation assigned to two transfers");
    }
  }

  info.sramScratch.bytes =
      maxAddressTableBytes == 0
          ? 0
          : alignTo(maxAddressTableBytes, kPipeSramScratchAlignmentBytes);
  return success();
}

PipeResourceRequirements
getPipeResourceRequirements(const PipeResourcePlan &info) {
  PipeCounterAllocationCounts counts;
  for (const PipeResourceInfo &resource :
       llvm::make_second_range(info.resources)) {
    counts.include(resource.completion.counter);
    counts.include(resource.readyCounter);
  }

  return PipeResourceRequirements{
      counts.localSemaphoreCount,
      counts.globalSemaphoreCount,
      info.sramScratch.bytes,
  };
}

/// Verify local semaphore ids before emitting ttkernel.get_semaphore. The
/// highest-id owner is tracked only to make over-limit diagnostics actionable.
LogicalResult
verifyPipeResourcePlanFitsHardware(ModuleOp mod, const PipeResourcePlan &info,
                                   const PipeResourceRequirements &reqs) {
  enum class PipeSemaphoreKind {
    ReceiverCompletion,
    SenderReady,
  };

  struct HighestSemaphore {
    int64_t index = -1;
    PipeSemaphoreKind kind = PipeSemaphoreKind::ReceiverCompletion;
    std::optional<PipeKey> pipe;
  };

  HighestSemaphore highest;
  for (const PipeResourceInfo &resource :
       llvm::make_second_range(info.resources)) {
    PipeCounterInfo completionCounter = resource.completion.counter;
    if (completionCounter.getStorage() == PipeCounterStorage::LocalSemaphore &&
        completionCounter.getIndex() > highest.index) {
      highest = HighestSemaphore{completionCounter.getIndex(),
                                 PipeSemaphoreKind::ReceiverCompletion,
                                 resource.pipe};
    }
    PipeCounterInfo readyCounter = resource.readyCounter;
    if (readyCounter.getStorage() == PipeCounterStorage::LocalSemaphore &&
        readyCounter.getIndex() > highest.index) {
      highest = HighestSemaphore{readyCounter.getIndex(),
                                 PipeSemaphoreKind::SenderReady, resource.pipe};
    }
  }

  int64_t requiredSemaphoreIds = reqs.syncSemaphoreCount;
  if (requiredSemaphoreIds <= kMaxHardwareSemaphoreIds) {
    return success();
  }

  auto diag = mod.emitError()
              << "pipe synchronization requires " << requiredSemaphoreIds
              << " hardware semaphore ids, exceeding TT hardware limit of "
              << kMaxHardwareSemaphoreIds
              << "; issue #619 tracks scalable pipe synchronization allocation";
  Diagnostic &note = diag.attachNote(mod.getLoc())
                     << "highest allocated semaphore id is " << highest.index
                     << " for ";
  auto appendPipe = [&](const PipeKey &pipe) {
    note << "pipe net " << pipe.pipeNetId << " src(" << pipe.srcX << ", "
         << pipe.srcY << ") dst(" << pipe.dstStartX << ", " << pipe.dstStartY
         << ") to(" << pipe.dstEndX << ", " << pipe.dstEndY << ")";
  };

  switch (highest.kind) {
  case PipeSemaphoreKind::ReceiverCompletion:
    note << "receiver-completion counter for ";
    assert(highest.pipe && "receiver-completion resource must have a pipe");
    appendPipe(*highest.pipe);
    break;
  case PipeSemaphoreKind::SenderReady:
    note << "sender-ready counter for ";
    assert(highest.pipe && "sender-ready resource must have a pipe");
    appendPipe(*highest.pipe);
    break;
  }

  return failure();
}

void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter,
                                  const PipeNetIndex &pipeNetIndex) {
  patterns.add<IfSrcLowering, IfDstLowering, CreatePipeLowering>(
      typeConverter, patterns.getContext());
  patterns.add<IsSrcLowering, IsDstLowering, IsActiveLowering>(
      typeConverter, patterns.getContext(), &pipeNetIndex);
}

} // namespace mlir::tt::ttl
