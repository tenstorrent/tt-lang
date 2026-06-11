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
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>
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

static PipeResourceInfo
lookupPipeResourceInfo(PipeType pipeType,
                       const PipeResourcePlan *pipeResourcePlan) {
  assert(pipeResourcePlan && "missing pipe resource plan");
  auto it = pipeResourcePlan->resources.find(getPipeKey(pipeType));
  assert(it != pipeResourcePlan->resources.end() &&
         "pipe missing from pipe resource plan");
  return it->second;
}

static PipeCompletionWaitInfo
lookupPipeCompletionWaitInfo(PipeType pipeType,
                             const PipeResourcePlan *pipeResourcePlan) {
  assert(pipeResourcePlan && "missing pipe resource plan");
  auto it = pipeResourcePlan->completionWaits.find(pipeType.getPipeNetId());
  assert(it != pipeResourcePlan->completionWaits.end() &&
         "pipe net missing from pipe completion info");
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
/// followed by compiler-managed pipe resources.
/// [Device 2.0] Keep this as a resource-plan lookup so the final device API
/// lowering can replace common-arg plumbing without changing pipe semantics.
static int64_t getPipeRuntimeCommonArgIndex(Operation *op,
                                            int64_t pipeRuntimeArgIndex) {
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe op is not inside a function");
  return getNumTensorFunctionArgs(func) + pipeRuntimeArgIndex;
}

static Value buildPipeRuntimeCommonArg(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       int64_t commonArgIndex) {
  auto argIndex = arith::ConstantIndexOp::create(rewriter, loc, commonArgIndex);
  return ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                        argIndex)
      .getResult();
}

/// Return the first pipe-resource runtime arg index used for GlobalSemaphore
/// ready-counter addresses.
static int64_t
getFirstPipeGlobalSemaphoreArgOffset(const PipeResourcePlan &info) {
  // GlobalSemaphore addresses follow the optional SRAM scratch base in the
  // common runtime args built by python/ttl/kernel_runner.py.
  return info.sramScratch.bytes > 0 ? 1 : 0;
}

struct LocalReadyCounterAddressInfo {
  int64_t senderReadySemIdx;
};

struct GlobalReadyCounterAddressInfo {
  int64_t runtimeCommonArgIndex;
};

using ReadyCounterAddressInfo =
    std::variant<LocalReadyCounterAddressInfo, GlobalReadyCounterAddressInfo>;

static ReadyCounterAddressInfo
getReadyCounterAddressInfo(Operation *op, const PipeResourceInfo &pipeResource,
                           const PipeResourcePlan &pipeResourcePlan) {
  if (auto *globalCounter =
          std::get_if<PipeGlobalReadyCounterInfo>(&pipeResource.readyCounter)) {
    int64_t argIndex = getPipeRuntimeCommonArgIndex(
        op, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
                globalCounter->globalSemaphoreIndex);
    return ReadyCounterAddressInfo{GlobalReadyCounterAddressInfo{argIndex}};
  }

  auto *localCounter =
      std::get_if<PipeLocalReadyCounterInfo>(&pipeResource.readyCounter);
  assert(localCounter && "unknown ready counter info");
  return ReadyCounterAddressInfo{
      LocalReadyCounterAddressInfo{localCounter->senderReadySemIdx}};
}

static Value buildReadyCounterAddress(Location loc,
                                      const ReadyCounterAddressInfo &info,
                                      ConversionPatternRewriter &rewriter) {
  // Lowering consumes both local and GlobalSemaphore ready counters as L1
  // addresses; only address construction differs between the two kinds.
  // [Device 2.0] This should become a typed semaphore-object lookup when the
  // device API exposes Semaphore/GlobalSemaphore objects directly.
  if (auto *localInfo = std::get_if<LocalReadyCounterAddressInfo>(&info)) {
    auto senderSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, localInfo->senderReadySemIdx);
    return ttk::GetSemaphoreOp::create(rewriter, loc, senderSemIdx).getResult();
  }

  auto *globalInfo = std::get_if<GlobalReadyCounterAddressInfo>(&info);
  assert(globalInfo && "unknown ready counter address info");
  return buildPipeRuntimeCommonArg(loc, rewriter,
                                   globalInfo->runtimeCommonArgIndex);
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

struct AddressTableInfo {
  int64_t scratchRuntimeCommonArgIndex;
  int64_t byteOffset = 0;
};

static AddressTableInfo
getAddressTableInfo(Operation *op, const PipeResourceInfo &pipeResource) {
  int64_t scratchArgIndex = getPipeRuntimeCommonArgIndex(op, 0);
  return AddressTableInfo{
      scratchArgIndex, pipeResource.addressStorage.sramAddressTable.byteOffset};
}

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
/// publish this address so senders do not infer receiver DFB state.
static Value
buildReceiverPublishedAddress(Value dst, Location loc,
                              const ReceiverPublishedAddressInfo &info,
                              ConversionPatternRewriter &rewriter) {
  auto receiverCBConverted =
      utils::convertTTLCBToTTKernel(info.receiverDFB, rewriter, loc);
  assert(succeeded(receiverCBConverted) &&
         "preflight checked receiver DFB type");

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
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto memrefTy = MemRefType::get({1}, builder.getI32Type());
    auto i32Ty = builder.getI32Type();
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantOp::create(builder, loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
    auto &perFunc = counters[func];
    SmallVector<int64_t> sortedPipeNetIds(pipeNetIds.begin(), pipeNetIds.end());
    llvm::sort(sortedPipeNetIds);
    for (int64_t pipeNetId : sortedPipeNetIds) {
      auto alloca = memref::AllocaOp::create(builder, loc, memrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, alloca,
                              ValueRange{zeroIdx});
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
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(pipeType, pipeResourcePlan);
  PipeCompletionWaitInfo completionInfo =
      lookupPipeCompletionWaitInfo(pipeType, pipeResourcePlan);
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

  ReadyCounterAddressInfo readyCounterInfo =
      getReadyCounterAddressInfo(op, pipeResource, *pipeResourcePlan);
  AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);

  int64_t dstStartX = pipeType.getDstStartX();
  int64_t dstStartY = pipeType.getDstStartY();
  int64_t dstEndX = pipeType.getDstEndX();
  int64_t dstEndY = pipeType.getDstEndY();
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked source DFB type");

  int64_t nocIdx = getNocIndex(op);
  Value nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                           rewriter.getI8IntegerAttr(nocIdx));

  int64_t expectedSignals =
      isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
  Value senderSemAddr =
      buildReadyCounterAddress(loc, readyCounterInfo, rewriter);
  auto senderSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
  auto expectedVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(expectedSignals));
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedVal);
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);

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

  // NOC operations require virtual/translated coordinates
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

  // Transfer the entire block in a single NOC write. Tiles are contiguous in
  // the CB, and destination CB layout is uniform across cores, so we can send
  // all tiles at once instead of one per tile.
  int64_t totalSizeBytes = cbNumTiles * pageSizeBytes;
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(totalSizeBytes));

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, srcPtrIdx);

  Value dstAddr =
      buildAddressTableDestinationAddress(loc, addressTableInfo, rewriter);

  if (pipeType.hasSingleReceiver()) {
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr,
                                 ValueRange{dstStartXVal, dstStartYVal},
                                 ValueRange{}, dstAddr, totalSizeVal);
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

  // Wait for all async writes to complete before signaling the semaphore.
  // Without this barrier, the receiver may wake up before all data arrives.
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);

  // Signal that data has arrived.
  if (pipeType.hasSingleReceiver()) {
    // Point-to-point: atomically increment destination's semaphore.
    auto semIdx = arith::ConstantIndexOp::create(rewriter, loc,
                                                 completionInfo.receiverSemIdx);
    auto semAddr = ttk::GetSemaphoreOp::create(rewriter, loc, semIdx);
    auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto dstSemNocAddr = ttk::GetNocAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, semAddr, nocVal);
    ttk::NocSemaphoreIncOp::create(rewriter, loc, dstSemNocAddr.getResult(),
                                   incrVal, nocVal, /*posted=*/BoolAttr());
  } else {
    // Collective: atomic inc on every receiver's recvSem. Receiver pairs
    // with cumulative wait_min via the per-PipeNet runtime counter.
    auto recvSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, completionInfo.receiverSemIdx);
    auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);

    // HW multicast auto-excludes the sender; num_dests counts only remote
    // receivers. No inc_multicast_loopback in tt-metal — sender's own
    // recvSem is incremented locally below.
    int64_t numRemoteDests = pipeType.srcInDstRange() ? numDests - 1 : numDests;
    auto numRemoteDestsVal = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(numRemoteDests));

    auto recvSemMcastAddr = ttk::GetNocMulticastAddrOp::create(
        rewriter, loc, mcastStartXVal, mcastStartYVal, mcastEndXVal,
        mcastEndYVal, recvSemAddr, nocVal);

    auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, recvSemMcastAddr.getResult(), incrVal, numRemoteDestsVal,
        nocVal, /*posted=*/BoolAttr());

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
          rewriter, loc, srcXTranslated, srcYTranslated, recvSemAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(rewriter, loc,
                                     selfRecvSemNocAddr.getResult(), incrVal,
                                     nocVal, /*posted=*/BoolAttr());
    }

    // Flush the (non-posted) atomic increments before the kernel can move
    // on. Without this barrier, receivers race with the sender on recvSem.
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeRecvPost(PipeRecvPostOp op, Value pipe, Value dst,
                                const PipeResourcePlan *pipeResourcePlan,
                                ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = mlir::cast<PipeType>(pipe.getType());
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(pipeType, pipeResourcePlan);
  FailureOr<ReceiverPublishedAddressInfo> publishedAddressInfo =
      getReceiverPublishedAddressInfo(op, dst, rewriter);
  if (failed(publishedAddressInfo)) {
    return failure();
  }
  AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
  ReadyCounterAddressInfo readyCounterInfo =
      getReadyCounterAddressInfo(op, pipeResource, *pipeResourcePlan);

  int64_t nocIdx = getNocIndex(op);
  auto indexTy = rewriter.getIndexType();

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

  Value publishedAddress =
      buildReceiverPublishedAddress(dst, loc, *publishedAddressInfo, rewriter);
  Value tableAddress =
      buildAddressTableAddress(loc, addressTableInfo, rewriter);
  // [Device 2.0] This is a receiver-authored write to a typed address table;
  // only this lowering should select the current inline NoC write primitive.
  auto senderTableNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, tableAddress, nocVal);
  auto byteEnableAll = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
  ttk::NocInlineDwWriteOp::create(rewriter, loc, senderTableNocAddr.getResult(),
                                  publishedAddress, byteEnableAll, nocVal);
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);

  Value senderSemAddr =
      buildReadyCounterAddress(loc, readyCounterInfo, rewriter);
  auto senderSemNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, senderSemAddr, nocVal);
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
  PipeCompletionWaitInfo completionInfo =
      lookupPipeCompletionWaitInfo(pipeType, pipeResourcePlan);
  (void)dst;

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

  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto recvSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, completionInfo.receiverSemIdx);
  auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  auto recvSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, recvSemAddr);

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

    // Get current core coordinates.
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

    // Get current core coordinates.
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
  mod.walk([&](CreatePipeOp op) {
    auto pipeType = mlir::cast<PipeType>(op.getResult().getType());
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

void buildPipeResourcePlan(const PipeNetIndex &index, PipeResourcePlan &info) {
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
        pipeNetId, getReceiverCompletionSemIdx(pipeNetId)};
  }
  int64_t firstSourceLocalSemIdx = numPipeNets;

  llvm::MapVector<PipeSourceKey, int64_t> pipeCountBySource;
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
  // Use one ready-counter kind per kernel so host allocation has one compact
  // descriptor layout. Liveness allocation can make this per source later.
  bool useGlobalReadyCounters =
      firstSourceLocalSemIdx + maxPipesPerSource > kMaxHardwareSemaphoreIds;

  llvm::MapVector<PipeSourceKey, int64_t> nextSemaphoreIdxBySource;
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
      PipeResourceInfo pipeResource{};
      pipeResource.transferContract = pipeInfo.transferContract;
      if (useGlobalReadyCounters) {
        pipeResource.readyCounter =
            PipeGlobalReadyCounterInfo{nextGlobalSemaphoreIndex++};
      } else {
        PipeSourceKey sourceKey = getPipeSourceKey(pipeType);
        auto emplaceResult = nextSemaphoreIdxBySource.try_emplace(
            sourceKey, firstSourceLocalSemIdx);
        int64_t &nextSemaphoreIdx = emplaceResult.first->second;
        pipeResource.readyCounter =
            PipeLocalReadyCounterInfo{nextSemaphoreIdx++};
      }
      pipeResource.addressStorage.sramAddressTable =
          PipeSramAddressTableInfo{nextAddressTableByteOffset};
      // Offsets are global within the per-core pipe SRAM allocation; the same
      // layout is instantiated independently on each source core.
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
    observe(completion.receiverSemIdx);
  }
  for (const auto &[pipe, resource] : info.resources) {
    (void)pipe;
    if (auto *localCounter =
            std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
      observe(localCounter->senderReadySemIdx);
    }
  }
  return highestSemaphoreIdx + 1;
}

int64_t getRequiredPipeGlobalSemaphoreCount(const PipeResourcePlan &info) {
  int64_t highestGlobalSemaphoreIndex = -1;
  for (const auto &[pipe, resource] : info.resources) {
    (void)pipe;
    if (auto *globalCounter =
            std::get_if<PipeGlobalReadyCounterInfo>(&resource.readyCounter)) {
      highestGlobalSemaphoreIndex = std::max(
          highestGlobalSemaphoreIndex, globalCounter->globalSemaphoreIndex);
    }
  }
  return highestGlobalSemaphoreIndex + 1;
}

int64_t getRequiredPipeSramScratchBytes(const PipeResourcePlan &info) {
  return info.sramScratch.bytes;
}

LogicalResult verifyPipeResourcePlanFitsHardware(ModuleOp mod,
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
    observe(completion.receiverSemIdx, ResourceKind::ReceiverCompletion);
  }
  for (const auto &[pipe, resource] : info.resources) {
    if (auto *localCounter =
            std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
      observe(localCounter->senderReadySemIdx, ResourceKind::SenderReady, pipe);
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
