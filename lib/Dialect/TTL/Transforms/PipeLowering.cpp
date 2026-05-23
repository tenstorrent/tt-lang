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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <optional>
#include <tuple>

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

static constexpr int64_t kPipeAddressWordBytes = 4;
static constexpr int64_t kPipeSramScratchAlignmentBytes = 32;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// TODO: move getTTLCBType and makeZeroI32 to a shared location if more
// lowering files need them.

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

static int64_t getNocIndex(Operation *op) {
  auto parentFunc = op->getParentOfType<FuncOp>();
  if (!parentFunc) {
    return 0;
  }
  auto attr = parentFunc->getAttrOfType<IntegerAttr>("ttl.noc_index");
  if (!attr) {
    return 0;
  }
  return attr.getInt();
}

static PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
}

static PipeKey getPipeSourceKey(PipeType pipeType) {
  return {pipeType.getSrcX(), pipeType.getSrcY(), 0, 0, 0, 0, 0};
}

static FailureOr<PipeResourceInfo>
lookupPipeResourceInfo(Operation *op, PipeType pipeType,
                       const PipeResourcePlan *pipeResourcePlan) {
  if (!pipeResourcePlan) {
    return op->emitError("internal compiler error: missing pipe resource plan");
  }
  auto it = pipeResourcePlan->resources.find(getPipeKey(pipeType));
  if (it == pipeResourcePlan->resources.end()) {
    return op->emitError("internal compiler error: pipe missing from pipe "
                         "resource plan");
  }
  return it->second;
}

static FailureOr<PipeCompletionWaitInfo>
lookupPipeCompletionWaitInfo(Operation *op, PipeType pipeType,
                             const PipeResourcePlan *pipeResourcePlan) {
  if (!pipeResourcePlan) {
    return op->emitError("internal compiler error: missing pipe resource plan");
  }
  auto it = pipeResourcePlan->completionWaits.find(pipeType.getPipeNetId());
  if (it == pipeResourcePlan->completionWaits.end()) {
    return op->emitError("internal compiler error: pipe net missing from pipe "
                         "completion lowering info");
  }
  return it->second;
}

static int64_t alignTo(int64_t value, int64_t alignment) {
  assert(alignment > 0 && "alignment must be positive");
  return ((value + alignment - 1) / alignment) * alignment;
}

/// Count tensor arguments because TTKernel common runtime args list tensor
/// buffer addresses before compiler-managed pipe resources.
static int64_t getNumTensorFunctionArgs(FuncOp func) {
  int64_t numTensorArgs = 0;
  for (BlockArgument argument : func.getArguments()) {
    if (llvm::isa<RankedTensorType>(argument.getType())) {
      ++numTensorArgs;
    }
  }
  return numTensorArgs;
}

/// Pipe kernels receive common runtime args for tensor buffer addresses first,
/// followed by compiler-managed pipe resources. `pipeRuntimeArgIndex` indexes
/// that pipe-resource suffix.
/// [Device 2.0] Keep this as a resource-plan lookup so the final device API
/// lowering can replace common-arg plumbing without changing pipe semantics.
static FailureOr<Value>
getPipeRuntimeCommonArg(Operation *op, Location loc,
                        ConversionPatternRewriter &rewriter,
                        int64_t pipeRuntimeArgIndex) {
  FuncOp func = op->getParentOfType<FuncOp>();
  if (!func) {
    return op->emitError("internal compiler error: pipe op is not inside a "
                         "function");
  }
  auto argIndex = arith::ConstantIndexOp::create(
      rewriter, loc, getNumTensorFunctionArgs(func) + pipeRuntimeArgIndex);
  return ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                        argIndex)
      .getResult();
}

/// Return the L1 base address of the compiler-managed pipe SRAM table.
static FailureOr<Value>
getPipeSramScratchBase(Operation *op, Location loc,
                       ConversionPatternRewriter &rewriter) {
  return getPipeRuntimeCommonArg(op, loc, rewriter, 0);
}

/// Return the first pipe-resource runtime arg index used for GlobalSemaphore
/// ready-counter addresses.
static int64_t
getFirstPipeGlobalSemaphoreArgOffset(const PipeResourcePlan &info) {
  // GlobalSemaphore addresses follow the optional SRAM scratch base in the
  // common runtime args built by python/ttl/kernel_runner.py.
  return info.sramScratch.bytes > 0 ? 1 : 0;
}

static FailureOr<Value>
buildReadyCounterAddress(Operation *op, Location loc,
                         const PipeResourceInfo &pipeResource,
                         const PipeResourcePlan &pipeResourcePlan,
                         ConversionPatternRewriter &rewriter) {
  // Lowering consumes both local and GlobalSemaphore ready counters as L1
  // addresses; only address construction differs between the two kinds.
  // [Device 2.0] This should become a typed semaphore-object lookup when the
  // device API exposes Semaphore/GlobalSemaphore objects directly.
  switch (pipeResource.readyCounter.kind) {
  case PipeReadyCounterKind::LocalSemaphore: {
    auto senderSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, pipeResource.readyCounter.senderReadySemIdx);
    return ttk::GetSemaphoreOp::create(rewriter, loc, senderSemIdx).getResult();
  }
  case PipeReadyCounterKind::GlobalSemaphore:
    return getPipeRuntimeCommonArg(
        op, loc, rewriter,
        getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
            pipeResource.readyCounter.globalSemaphoreIndex);
  }
  llvm_unreachable("unknown pipe ready counter kind");
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

/// Load the receiver-published destination DFB address from this pipe's
/// source-core SRAM address-table entry.
static FailureOr<Value>
buildAddressTableDestinationAddress(Operation *op, Location loc,
                                    const PipeResourceInfo &pipeResource,
                                    ConversionPatternRewriter &rewriter) {
  if (!pipeResource.usesSramAddressTable()) {
    return op->emitError("internal compiler error: pipe has no SRAM address "
                         "table");
  }
  FailureOr<Value> scratchBase = getPipeSramScratchBase(op, loc, rewriter);
  if (failed(scratchBase)) {
    return failure();
  }
  Value tableAddress = addByteOffset(
      loc, *scratchBase, pipeResource.addressStorage.sramAddressTable.byteOffset,
      rewriter);
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

/// Compute the exact DFB address selected by ttl.copy(pipe, dst). Receivers
/// publish this address so senders do not infer receiver DFB state.
static FailureOr<Value>
buildReceiverPublishedAddress(Operation *op, Value dst, Location loc,
                              ConversionPatternRewriter &rewriter) {
  Value receiverCB = getAttachedCB(dst);
  if (!receiverCB) {
    return rewriter.notifyMatchFailure(
        op, "pipe receive destination is not attached to a DFB");
  }
  auto receiverCBConverted =
      utils::convertTTLCBToTTKernel(receiverCB, rewriter, loc);
  if (failed(receiverCBConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert receiver DFB");
  }

  auto receiverCBType = getTTLCBType(receiverCB);
  if (!receiverCBType) {
    return rewriter.notifyMatchFailure(op, "failed to get receiver DFB type");
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>(receiverCBType.getElementType());
  if (!tileType) {
    return rewriter.notifyMatchFailure(
        op, "receiver DFB element type must be tile");
  }

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
      rewriter.getI32IntegerAttr(tileType.getSizeBytes()));
  auto byteOffset =
      arith::MulIOp::create(rewriter, loc, tileOffsetI32, pageSizeBytes);
  return arith::AddIOp::create(rewriter, loc, receiverWritePtr, byteOffset)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Per-PipeNet receiver counter allocation
//===----------------------------------------------------------------------===//

void allocatePipeNetReceiveCounters(ModuleOp mod, PipeNetCounterMap &counters) {
  mod.walk([&](FuncOp func) {
    // Collect unique pipeNetIds that have at least one receive in this
    // function. A runtime counter is required because receive waits may be
    // dynamically re-executed inside loops.
    llvm::SmallSetVector<int64_t, 4> pipeNetIds;
    func.walk([&](Operation *op) {
      if (auto post = mlir::dyn_cast<PipeRecvPostOp>(op)) {
        auto pipeTy = mlir::cast<PipeType>(post.getPipe().getType());
        if (getAttachedCB(post.getDst())) {
          pipeNetIds.insert(pipeTy.getPipeNetId());
        }
      }
    });
    if (pipeNetIds.empty()) {
      return;
    }
    // Allocas + zero-stores at function entry dominate every receive post,
    // including posts inside scf.if from `if_dst`.
    OpBuilder b(func.getContext());
    b.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto memrefTy = MemRefType::get({1}, b.getI32Type());
    auto i32Ty = b.getI32Type();
    Value zeroIdx = arith::ConstantIndexOp::create(b, loc, 0);
    Value zeroI32 =
        arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
    auto &perFunc = counters[func];
    SmallVector<int64_t> sortedPipeNetIds(pipeNetIds.begin(), pipeNetIds.end());
    llvm::sort(sortedPipeNetIds);
    for (int64_t pipeNetId : sortedPipeNetIds) {
      auto alloca = memref::AllocaOp::create(b, loc, memrefTy);
      memref::StoreOp::create(b, loc, zeroI32, alloca, ValueRange{zeroIdx});
      perFunc[pipeNetId] = alloca.getResult();
    }
  });
}

/// Lower CB -> Pipe copy: write source DFB data to the receiver-published
/// destination address, then signal arrival.
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            bool isConsumerCB,
                            const PipeResourcePlan *pipeResourcePlan,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = mlir::cast<PipeType>(pipe.getType());
  FailureOr<PipeResourceInfo> pipeResource =
      lookupPipeResourceInfo(op, pipeType, pipeResourcePlan);
  if (failed(pipeResource)) {
    return failure();
  }
  FailureOr<PipeCompletionWaitInfo> completionInfo =
      lookupPipeCompletionWaitInfo(op, pipeType, pipeResourcePlan);
  if (failed(completionInfo)) {
    return failure();
  }
  assert(completionInfo->kind == PipeCompletionWaitKind::LocalSemaphore);
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }

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

  // Build optional NOC index value for ops that accept a noc parameter.
  int64_t nocIdx = getNocIndex(op);
  Value nocVal;
  if (nocIdx > 0) {
    nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                       rewriter.getI8IntegerAttr(nocIdx));
  }

  int64_t expectedSignals =
      isCollectiveTransfer(pipeResource->transferContract) ? numDests : 1;
  FailureOr<Value> senderSemAddr = buildReadyCounterAddress(
      op, loc, *pipeResource, *pipeResourcePlan, rewriter);
  if (failed(senderSemAddr)) {
    return failure();
  }
  auto senderSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, *senderSemAddr);
  auto expectedVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(expectedSignals));
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedVal);
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);

  SmallVector<int64_t> cbBounds(cbShape.begin(), cbShape.end());
  int64_t cbNumTiles = 1;
  for (int64_t d : cbBounds) {
    cbNumTiles *= d;
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

  // NOC operations require virtual/translated coordinates
  auto dstStartXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, dstStartXLogical);
  auto dstStartYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, dstStartYLogical);
  auto dstEndXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, dstEndXLogical);
  auto dstEndYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, dstEndYLogical);

  auto numDestsVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numDests));

  // Transfer the entire block in a single NOC write. Tiles are contiguous in
  // the CB, and destination CB layout is uniform across cores, so we can send
  // all tiles at once instead of one per tile.
  int64_t totalSizeBytes = cbNumTiles * pageSizeBytes;
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(totalSizeBytes));

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, srcPtrIdx);

  FailureOr<Value> dstAddr =
      buildAddressTableDestinationAddress(op, loc, *pipeResource, rewriter);
  if (failed(dstAddr)) {
    return failure();
  }

  if (pipeType.isUnicast()) {
    auto nocAddr = ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal,
                                             dstStartYVal, *dstAddr);
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr, nocAddr.getResult(),
                                 totalSizeVal);
  } else {
    auto mcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
        *dstAddr, nocVal);
    if (pipeType.srcInDstRange()) {
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, mcastAddr.getResult(), totalSizeVal,
          numDestsVal, /*linked=*/nullptr,
          /*multicast_path_reserve=*/nullptr, nocVal);
    } else {
      ttk::NocAsyncWriteMulticastOp::create(
          rewriter, loc, srcAddr, mcastAddr.getResult(), totalSizeVal,
          numDestsVal, /*linked=*/nullptr,
          /*multicast_path_reserve=*/nullptr, nocVal);
    }
  }

  // Wait for all async writes to complete before signaling the semaphore.
  // Without this barrier, the receiver may wake up before all data arrives.
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);

  // Signal that data has arrived.
  if (pipeType.isUnicast()) {
    // Point-to-point: atomically increment destination's semaphore.
    auto semIdx = arith::ConstantIndexOp::create(
        rewriter, loc, completionInfo->receiverSemIdx);
    auto semAddr = ttk::GetSemaphoreOp::create(rewriter, loc, semIdx);
    auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto dstSemNocAddr = ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal,
                                                   dstStartYVal, semAddr);
    ttk::NocSemaphoreIncOp::create(rewriter, loc, dstSemNocAddr.getResult(),
                                   incrVal, /*noc_id=*/Value(),
                                   /*posted=*/BoolAttr());
  } else {
    // Multicast: atomic inc on every receiver's recvSem. Receiver pairs
    // with cumulative wait_min via the per-PipeNet runtime counter.
    auto recvSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, completionInfo->receiverSemIdx);
    auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);

    // HW multicast auto-excludes the sender; num_dests counts only remote
    // receivers. No inc_multicast_loopback in tt-metal — sender's own
    // recvSem is incremented locally below.
    int64_t numRemoteDests = pipeType.srcInDstRange() ? numDests - 1 : numDests;
    auto numRemoteDestsVal = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numRemoteDests));

    auto recvSemMcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
        recvSemAddr, nocVal);

    auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, recvSemMcastAddr.getResult(), incrVal, numRemoteDestsVal,
        /*noc_id=*/Value(), /*posted=*/BoolAttr());

    if (pipeType.srcInDstRange()) {
      // Local self-inc: when sender is also a receiver of overlapping
      // pipes, its own cumulative count must include this pipe.
      auto srcXLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
      auto srcYLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
      auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
          rewriter, loc, indexTy, srcXLogical);
      auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
          rewriter, loc, indexTy, srcYLogical);
      auto selfRecvSemNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, srcXTranslated, srcYTranslated, recvSemAddr);
      ttk::NocSemaphoreIncOp::create(rewriter, loc,
                                     selfRecvSemNocAddr.getResult(), incrVal,
                                     /*noc_id=*/Value(), /*posted=*/BoolAttr());
    }

    // Flush the (non-posted) atomic increments before the kernel can move
    // on. Without this barrier, receivers race with the sender on recvSem.
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, /*noc_id=*/Value());
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeRecvPost(PipeRecvPostOp op, Value pipe, Value dst,
                                const PipeResourcePlan *pipeResourcePlan,
                                ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = mlir::cast<PipeType>(pipe.getType());
  FailureOr<PipeResourceInfo> pipeResource =
      lookupPipeResourceInfo(op, pipeType, pipeResourcePlan);
  if (failed(pipeResource)) {
    return failure();
  }
  int64_t nocIdx = getNocIndex(op);
  auto indexTy = rewriter.getIndexType();

  Value nocVal;
  Value inlineNocId = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));
  if (nocIdx > 0) {
    nocVal = inlineNocId;
  }

  auto srcXLogical =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
  auto srcYLogical =
      arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
  auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, srcXLogical);
  auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, srcYLogical);

  FailureOr<Value> publishedAddress =
      buildReceiverPublishedAddress(op, dst, loc, rewriter);
  if (failed(publishedAddress)) {
    return failure();
  }
  FailureOr<Value> scratchBase = getPipeSramScratchBase(op, loc, rewriter);
  if (failed(scratchBase)) {
    return failure();
  }
  Value tableAddress = addByteOffset(
      loc, *scratchBase,
      pipeResource->addressStorage.sramAddressTable.byteOffset, rewriter);
  // [Device 2.0] This is a receiver-authored write to a typed address table;
  // only this lowering should select the current inline NoC write primitive.
  auto senderTableNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, tableAddress);
  auto byteEnableAll = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
  ttk::NocInlineDwWriteOp::create(rewriter, loc, senderTableNocAddr.getResult(),
                                  *publishedAddress, byteEnableAll,
                                  inlineNocId);
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);

  FailureOr<Value> senderSemAddr = buildReadyCounterAddress(
      op, loc, *pipeResource, *pipeResourcePlan, rewriter);
  if (failed(senderSemAddr)) {
    return failure();
  }
  auto senderSemNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, *senderSemAddr);
  auto readyIncr = arith::ConstantIndexOp::create(rewriter, loc, 1);
  ttk::NocSemaphoreIncOp::create(rewriter, loc, senderSemNocAddr.getResult(),
                                 readyIncr, nocVal, /*posted=*/BoolAttr());

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Lower the receiver completion wait with a per-PipeNet runtime counter.
LogicalResult lowerPipeRecvWait(PipeRecvWaitOp op, Value pipe, Value dst,
                                const PipeNetCounterMap *counters,
                                const PipeResourcePlan *pipeResourcePlan,
                                ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = mlir::cast<PipeType>(pipe.getType());
  FailureOr<PipeCompletionWaitInfo> completionInfo =
      lookupPipeCompletionWaitInfo(op, pipeType, pipeResourcePlan);
  if (failed(completionInfo)) {
    return failure();
  }
  assert(completionInfo->kind == PipeCompletionWaitKind::LocalSemaphore);
  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  (void)dst;

  auto recvSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, completionInfo->receiverSemIdx);
  auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  auto recvSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, recvSemAddr);

  Value counter;
  if (counters) {
    auto func = op->getParentOfType<func::FuncOp>();
    auto fIt = counters->find(func);
    if (fIt != counters->end()) {
      auto pIt = fIt->second.find(pipeType.getPipeNetId());
      if (pIt != fIt->second.end()) {
        counter = pIt->second;
      }
    }
  }
  if (!counter) {
    // Counter pre-allocation is a hard precondition. Surfacing this as
    // notifyMatchFailure would let the partial-conversion driver report
    // a generic legalization failure instead of the actual pipeline-ordering
    // bug; emit a real error.
    op.emitError("pipe receive without per-PipeNet counter; "
                 "allocatePipeNetReceiveCounters must run before "
                 "convert-ttl-to-ttkernel");
    return failure();
  }

  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto loaded =
      memref::LoadOp::create(rewriter, loc, counter, ValueRange{zeroIdx});
  auto oneI32 = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
  auto newCounter = arith::AddIOp::create(rewriter, loc, loaded, oneI32);
  memref::StoreOp::create(rewriter, loc, newCounter, counter,
                          ValueRange{zeroIdx});
  ttk::SemaphoreWaitMinOp::create(rewriter, loc, recvSemPtr, newCounter);

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

struct IfSrcLowering : OpConversionPattern<IfSrcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfSrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    // Get current core coordinates.
    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    // Get source coordinates from pipe type.
    auto srcXConst =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
    auto srcYConst =
        arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());

    // Check if current core matches source coordinates.
    auto matchX = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        coreX, srcXConst);
    auto matchY = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        coreY, srcYConst);
    auto isSrc = arith::AndIOp::create(rewriter, loc, matchX, matchY);

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

    // Get current core coordinates.
    auto coreX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto coreY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());

    // Get destination range from pipe type.
    int64_t dstMinX = std::min(pipeType.getDstStartX(), pipeType.getDstEndX());
    int64_t dstMaxX = std::max(pipeType.getDstStartX(), pipeType.getDstEndX());
    int64_t dstMinY = std::min(pipeType.getDstStartY(), pipeType.getDstEndY());
    int64_t dstMaxY = std::max(pipeType.getDstStartY(), pipeType.getDstEndY());

    auto minXConst = arith::ConstantIndexOp::create(rewriter, loc, dstMinX);
    auto maxXConst = arith::ConstantIndexOp::create(rewriter, loc, dstMaxX);
    auto minYConst = arith::ConstantIndexOp::create(rewriter, loc, dstMinY);
    auto maxYConst = arith::ConstantIndexOp::create(rewriter, loc, dstMaxY);

    // Check if current core is within destination range.
    // coreX >= minX && coreX <= maxX && coreY >= minY && coreY <= maxY
    auto geMinX = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, coreX, minXConst);
    auto leMaxX = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, coreX, maxXConst);
    auto geMinY = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, coreY, minYConst);
    auto leMaxY = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, coreY, maxYConst);

    auto inRangeX = arith::AndIOp::create(rewriter, loc, geMinX, leMaxX);
    auto inRangeY = arith::AndIOp::create(rewriter, loc, geMinY, leMaxY);
    auto isDst = arith::AndIOp::create(rewriter, loc, inRangeX, inRangeY);

    lowerToScfIf(op, isDst, rewriter);
    return success();
  }
};

static Value buildSrcMatch(OpBuilder &b, Location loc, Value coreX, Value coreY,
                           PipeType pt) {
  auto sx = arith::ConstantIndexOp::create(b, loc, pt.getSrcX());
  auto sy = arith::ConstantIndexOp::create(b, loc, pt.getSrcY());
  auto eqX = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, coreX, sx);
  auto eqY = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, coreY, sy);
  return arith::AndIOp::create(b, loc, eqX, eqY);
}

static Value buildDstMatch(OpBuilder &b, Location loc, Value coreX, Value coreY,
                           PipeType pt) {
  int64_t minX = std::min(pt.getDstStartX(), pt.getDstEndX());
  int64_t maxX = std::max(pt.getDstStartX(), pt.getDstEndX());
  int64_t minY = std::min(pt.getDstStartY(), pt.getDstEndY());
  int64_t maxY = std::max(pt.getDstStartY(), pt.getDstEndY());
  auto cMinX = arith::ConstantIndexOp::create(b, loc, minX);
  auto cMaxX = arith::ConstantIndexOp::create(b, loc, maxX);
  auto cMinY = arith::ConstantIndexOp::create(b, loc, minY);
  auto cMaxY = arith::ConstantIndexOp::create(b, loc, maxY);
  auto geX =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, coreX, cMinX);
  auto leX =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, coreX, cMaxX);
  auto geY =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, coreY, cMinY);
  auto leY =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, coreY, cMaxY);
  auto inX = arith::AndIOp::create(b, loc, geX, leX);
  auto inY = arith::AndIOp::create(b, loc, geY, leY);
  return arith::AndIOp::create(b, loc, inX, inY);
}

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

// Base for IsSrc/IsDst/IsActive lowerings: holds the shared PipeNetIndex
// borrowed pointer so the per-pattern matchAndRewrite stays compact.
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
        [](OpBuilder &b, Location loc, Value cx, Value cy, PipeType pt) {
          Value src = buildSrcMatch(b, loc, cx, cy, pt);
          Value dst = buildDstMatch(b, loc, cx, cy, pt);
          return Value(arith::OrIOp::create(b, loc, src, dst));
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
  mod.walk([&](CreatePipeOp op) {
    auto pt = mlir::cast<PipeType>(op.getResult().getType());
    int64_t netId = pt.getPipeNetId();
    PipeKey key{pt.getSrcX(),      pt.getSrcY(),    pt.getDstStartX(),
                pt.getDstStartY(), pt.getDstEndX(), pt.getDstEndY()};
    PipeTransferContract contract = getPipeTransferContract(op);
    if (seenPerNet[netId].insert(key)) {
      index[netId].push_back(PipeInfo{pt, contract});
      return;
    }
    if (!isCollectiveTransfer(contract)) {
      return;
    }
    for (PipeInfo &pipeInfo : index[netId]) {
      PipeType existingType = pipeInfo.pipeType;
      PipeKey existingKey{existingType.getSrcX(), existingType.getSrcY(),
                          existingType.getDstStartX(),
                          existingType.getDstStartY(),
                          existingType.getDstEndX(), existingType.getDstEndY()};
      if (existingKey == key) {
        pipeInfo.transferContract = PipeTransferContract::Collective;
        break;
      }
    }
  });
}

void buildPipeResourcePlan(ModuleOp, const PipeNetIndex &index,
                           const PipeGraph &, PipeResourcePlan &info) {
  int64_t numPipeNets = 0;
  for (const auto &[pipeNetId, pipes] : index) {
    if (!pipes.empty()) {
      numPipeNets = std::max(numPipeNets, pipeNetId + 1);
    }
  }

  SmallVector<int64_t> sortedPipeNetIds;
  sortedPipeNetIds.reserve(index.size());
  for (const auto &[pipeNetId, pipes] : index) {
    if (!pipes.empty()) {
      sortedPipeNetIds.push_back(pipeNetId);
    }
  }
  llvm::sort(sortedPipeNetIds);

  for (int64_t pipeNetId : sortedPipeNetIds) {
    info.completionWaits[pipeNetId] = PipeCompletionWaitInfo{
        PipeCompletionWaitKind::LocalSemaphore, pipeNetId,
        getReceiverCompletionSemIdx(pipeNetId)};
  }
  int64_t firstSourceLocalSemIdx = numPipeNets;

  llvm::MapVector<PipeKey, int64_t> pipeCountBySource;
  for (int64_t pipeNetId : sortedPipeNetIds) {
    auto pipeNetIt = index.find(pipeNetId);
    assert(pipeNetIt != index.end());
    for (PipeInfo pipeInfo : pipeNetIt->second) {
      ++pipeCountBySource[getPipeSourceKey(pipeInfo.pipeType)];
    }
  }
  int64_t maxPipesPerSource = 0;
  for (const auto &[sourceKey, count] : pipeCountBySource) {
    (void)sourceKey;
    maxPipesPerSource = std::max(maxPipesPerSource, count);
  }
  bool useGlobalReadyCounters =
      firstSourceLocalSemIdx + maxPipesPerSource > kMaxHardwareSemaphoreIds;

  llvm::MapVector<PipeKey, int64_t> nextSemaphoreIdxBySource;
  int64_t nextGlobalSemaphoreIndex = 0;
  int64_t nextAddressTableByteOffset = 0;

  for (int64_t pipeNetId : sortedPipeNetIds) {
    auto pipeNetIt = index.find(pipeNetId);
    assert(pipeNetIt != index.end());
    SmallVector<PipeInfo> pipes = pipeNetIt->second;
    llvm::sort(pipes, [](PipeInfo lhs, PipeInfo rhs) {
      PipeType lhsType = lhs.pipeType;
      PipeType rhsType = rhs.pipeType;
      return std::make_tuple(lhsType.getSrcX(), lhsType.getSrcY(),
                             lhsType.getDstStartX(), lhsType.getDstStartY(),
                             lhsType.getDstEndX(), lhsType.getDstEndY()) <
             std::make_tuple(rhsType.getSrcX(), rhsType.getSrcY(),
                             rhsType.getDstStartX(), rhsType.getDstStartY(),
                             rhsType.getDstEndX(), rhsType.getDstEndY());
    });

    for (PipeInfo pipeInfo : pipes) {
      PipeType pipeType = pipeInfo.pipeType;
      PipeKey sourceKey = getPipeSourceKey(pipeType);
      auto emplaceResult = nextSemaphoreIdxBySource.try_emplace(
          sourceKey, firstSourceLocalSemIdx);
      int64_t &nextSemaphoreIdx = emplaceResult.first->second;
      int64_t senderReadySemIdx = nextSemaphoreIdx++;
      PipeResourceInfo pipeResource{};
      pipeResource.transferContract = pipeInfo.transferContract;
      if (useGlobalReadyCounters) {
        pipeResource.readyCounter.kind = PipeReadyCounterKind::GlobalSemaphore;
        pipeResource.readyCounter.globalSemaphoreIndex =
            nextGlobalSemaphoreIndex++;
      } else {
        pipeResource.readyCounter.kind = PipeReadyCounterKind::LocalSemaphore;
        pipeResource.readyCounter.senderReadySemIdx = senderReadySemIdx;
      }
      pipeResource.addressStorage.kind =
          PipeAddressStorageKind::SramAddressTable;
      pipeResource.addressStorage.sramAddressTable =
          PipeSramAddressTableInfo{nextAddressTableByteOffset};
      nextAddressTableByteOffset += kPipeAddressWordBytes;
      info.resources[getPipeKey(pipeType)] = pipeResource;
    }
  }
  info.sramScratch.bytes =
      info.resources.empty()
          ? 0
          : alignTo(nextAddressTableByteOffset, kPipeSramScratchAlignmentBytes);
}

int64_t getRequiredPipeSyncSemaphoreCount(const PipeResourcePlan &info) {
  int64_t highestSemaphoreIdx = -1;
  auto observe = [&](int64_t index) {
    highestSemaphoreIdx = std::max(highestSemaphoreIdx, index);
  };

  for (const auto &[pipeNetId, completion] : info.completionWaits) {
    (void)pipeNetId;
    assert(completion.kind == PipeCompletionWaitKind::LocalSemaphore);
    observe(completion.receiverSemIdx);
  }
  for (const auto &[pipe, resource] : info.resources) {
    (void)pipe;
    if (resource.readyCounter.kind == PipeReadyCounterKind::LocalSemaphore) {
      observe(resource.readyCounter.senderReadySemIdx);
    }
  }
  return highestSemaphoreIdx + 1;
}

int64_t
getRequiredPipeGlobalSemaphoreCount(const PipeResourcePlan &info) {
  int64_t highestGlobalSemaphoreIndex = -1;
  for (const auto &[pipe, resource] : info.resources) {
    (void)pipe;
    if (resource.readyCounter.kind == PipeReadyCounterKind::GlobalSemaphore) {
      highestGlobalSemaphoreIndex =
          std::max(highestGlobalSemaphoreIndex,
                   resource.readyCounter.globalSemaphoreIndex);
    }
  }
  return highestGlobalSemaphoreIndex + 1;
}

int64_t getRequiredPipeSramScratchBytes(const PipeResourcePlan &info) {
  return info.sramScratch.bytes;
}

LogicalResult
verifyPipeResourcePlanFitsHardware(ModuleOp mod,
                                   const PipeResourcePlan &info) {
  enum class ResourceKind {
    ReceiverCompletion,
    SenderReady,
  };

  struct HighestSemaphore {
    int64_t index = -1;
    ResourceKind resource = ResourceKind::ReceiverCompletion;
    std::optional<PipeKey> pipe;
  };

  HighestSemaphore highest;
  auto observe = [&](int64_t index, ResourceKind resource,
                     std::optional<PipeKey> pipe = std::nullopt) {
    if (index > highest.index) {
      highest = HighestSemaphore{index, resource, pipe};
    }
  };

  for (const auto &[pipeNetId, completion] : info.completionWaits) {
    (void)pipeNetId;
    assert(completion.kind == PipeCompletionWaitKind::LocalSemaphore);
    observe(completion.receiverSemIdx, ResourceKind::ReceiverCompletion);
  }
  for (const auto &[pipe, resource] : info.resources) {
    if (resource.readyCounter.kind == PipeReadyCounterKind::LocalSemaphore) {
      observe(resource.readyCounter.senderReadySemIdx, ResourceKind::SenderReady,
              pipe);
    }
  }

  int64_t requiredSemaphoreIds = getRequiredPipeSyncSemaphoreCount(info);
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

  switch (highest.resource) {
  case ResourceKind::ReceiverCompletion:
    note << "receiver-completion counter";
    break;
  case ResourceKind::SenderReady:
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
