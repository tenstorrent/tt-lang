// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeLowering.h"

#include "PipeCapacityAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
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

static FailureOr<PipeTransferCreateOp> getPipeTransferCreate(Operation *op,
                                                             Value transfer) {
  auto createOp = findPipeTransferCreateForTransfer(transfer);
  if (!createOp) {
    return op->emitError() << op->getName()
                           << " must use a transfer derived from "
                              "ttl.pipe_transfer.create";
  }
  return createOp;
}

static PipeResourceInfo
lookupPipeResourceInfo(PipeTransferCreateOp createOp,
                       const PipeResourcePlan &pipeResourcePlan) {
  auto it = pipeResourcePlan.resources.find(createOp.getOperation());
  assert(it != pipeResourcePlan.resources.end() &&
         "pipe transfer missing from pipe resource plan");
  return it->second;
}

static PipeCompletionWaitInfo
lookupPipeCompletionWaitInfo(PipeType pipeType,
                             const PipeResourcePlan &pipeResourcePlan) {
  auto it = pipeResourcePlan.completionWaits.find(pipeType.getPipeNetId());
  assert(it != pipeResourcePlan.completionWaits.end() &&
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

static Value buildLocalSemaphoreAddress(Location loc, OpBuilder &builder,
                                        int64_t semaphoreIndex) {
  Value semaphoreIndexValue =
      arith::ConstantIndexOp::create(builder, loc, semaphoreIndex);
  return ttk::GetSemaphoreOp::create(builder, loc, semaphoreIndexValue)
      .getResult();
}

static Value buildLocalSemaphorePtr(Location loc, OpBuilder &builder,
                                    int64_t semaphoreIndex) {
  auto l1PtrTy = ttk::L1AddrPtrType::get(builder.getContext(), 32);
  Value semaphoreAddress =
      buildLocalSemaphoreAddress(loc, builder, semaphoreIndex);
  return ttk::CastToL1PtrOp::create(builder, loc, l1PtrTy, semaphoreAddress)
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

PipeReadyCounterInfo
PipeReadyCounterInfo::localSemaphore(int64_t senderReadyCounterSemIdx) {
  return PipeReadyCounterInfo(PipeReadyCounterStorage::LocalSemaphore,
                              senderReadyCounterSemIdx);
}

PipeReadyCounterInfo
PipeReadyCounterInfo::globalSemaphore(int64_t globalSemaphoreIndex) {
  return PipeReadyCounterInfo(PipeReadyCounterStorage::GlobalSemaphore,
                              globalSemaphoreIndex);
}

ReadyCounterAddressInfo PipeReadyCounterInfo::getAddressInfo(
    Operation *op, const PipeResourcePlan &pipeResourcePlan) const {
  switch (storage) {
  case PipeReadyCounterStorage::LocalSemaphore:
    return {ReadyCounterAddressStorage::LocalSemaphore, index};
  case PipeReadyCounterStorage::GlobalSemaphore: {
    int64_t argIndex = getPipeRuntimeCommonArgIndex(
        op, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) + index);
    return {ReadyCounterAddressStorage::GlobalSemaphoreRuntimeArg, argIndex};
  }
  }
  llvm_unreachable("unknown pipe ready-counter storage");
}

void PipeReadyCounterInfo::observe(PipeReadyCounterObserver &observer) const {
  switch (storage) {
  case PipeReadyCounterStorage::LocalSemaphore:
    observer.observeLocalSemaphore(index);
    return;
  case PipeReadyCounterStorage::GlobalSemaphore:
    observer.observeGlobalSemaphore(index);
    return;
  }
  llvm_unreachable("unknown pipe ready-counter storage");
}

/// Resolve the resource-plan ready-counter allocation to the addressing form
/// used by TTKernel lowering at this operation site.
static ReadyCounterAddressInfo
getReadyCounterAddressInfo(Operation *op, const PipeResourceInfo &pipeResource,
                           const PipeResourcePlan &pipeResourcePlan) {
  assert(pipeResource.readyCounter &&
         "sender-ready protocol selected without a sender-ready counter");
  return pipeResource.readyCounter->getAddressInfo(op, pipeResourcePlan);
}

/// Build the L1 address for the sender-ready counter for either storage kind.
static Value buildReadyCounterAddress(Location loc,
                                      const ReadyCounterAddressInfo &info,
                                      ConversionPatternRewriter &rewriter) {
  // Lowering consumes both local and GlobalSemaphore ready counters as L1
  // addresses; only address construction differs between the two kinds.
  // [Device 2.0] This should become a typed semaphore-object lookup when the
  // device API exposes Semaphore/GlobalSemaphore objects directly.
  switch (info.storage) {
  case ReadyCounterAddressStorage::LocalSemaphore: {
    auto senderReadyCounterSemIdx =
        arith::ConstantIndexOp::create(rewriter, loc, info.index);
    auto senderReadyCounterAddr =
        ttk::GetSemaphoreOp::create(rewriter, loc, senderReadyCounterSemIdx);
    return senderReadyCounterAddr.getResult();
  }
  case ReadyCounterAddressStorage::GlobalSemaphoreRuntimeArg:
    return buildPipeRuntimeCommonArg(loc, rewriter, info.index);
  }
  llvm_unreachable("unknown ready counter address storage");
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
         "fallback pipe missing address-table storage");
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

/// Compute the statically assigned receiver DFB address.
static Value buildComputedReceiverDFBDestinationAddress(
    Location loc, const PipeComputedAddressInfo &info,
    ConversionPatternRewriter &rewriter) {
  auto baseAddress = ttk::GetCompileArgValOp::create(
      rewriter, loc, rewriter.getI32Type(),
      static_cast<int32_t>(info.baseCompileTimeArgIndex));
  return addByteOffset(loc, baseAddress.getResult(), info.staticByteOffset,
                       rewriter);
}

static void lowerPipeCapacityRelease(Location loc,
                                     const PipeCapacityReleaseInfo &release,
                                     Value nocVal,
                                     ConversionPatternRewriter &rewriter) {
  const PipeCapacityReleaseTarget &target = release.target;
  auto indexTy = rewriter.getIndexType();
  Value semaphoreAddress =
      buildLocalSemaphoreAddress(loc, rewriter, release.semaphoreIndex);
  Value sourceXLogical =
      arith::ConstantIndexOp::create(rewriter, loc, target.logicalX);
  Value sourceYLogical =
      arith::ConstantIndexOp::create(rewriter, loc, target.logicalY);
  Value sourceXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, sourceXLogical);
  Value sourceYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, sourceYLogical);
  Value releaseCount =
      arith::ConstantIntOp::create(rewriter, loc, release.count, 32);
  Value remoteCapacityNocAddr =
      ttk::GetNocAddrOp::create(rewriter, loc, sourceXTranslated,
                                sourceYTranslated, semaphoreAddress, nocVal)
          .getResult();
  ttk::NocSemaphoreIncOp::create(rewriter, loc, remoteCapacityNocAddr,
                                 releaseCount, nocVal, /*posted=*/BoolAttr());
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
  // Each kernel function tracks its own receive-wait progress. Walk the
  // function bodies to find the PipeNets that may complete receives there.
  mod.walk([&](FuncOp func) {
    // Collect unique pipeNetIds that have at least one receive in this
    // function. A runtime counter is required because receive waits may be
    // dynamically re-executed inside loops.
    llvm::SmallSetVector<int64_t, 4> pipeNetIds;
    func.walk([&](Operation *op) {
      if (auto post = mlir::dyn_cast<PipeTransferPostOp>(op)) {
        auto createOp = findPipeTransferCreateForTransfer(post.getTransfer());
        assert(createOp && "pipe transfer post missing traced create op");
        auto pipeTy = mlir::cast<PipeType>(createOp.getPipe().getType());
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

void initializePipeCapacitySemaphores(
    const PipeCapacityPlan &pipeCapacityPlan,
    PipeNetCounterMap &senderCapacityCounters) {
  for (const auto &entry : pipeCapacityPlan.getInitializations()) {
    FuncOp func = entry.first;
    const SmallVector<PipeCapacityInitInfo> &initializations = entry.second;
    SmallVector<PipeCapacityInitInfo> sortedInitializations(initializations);
    llvm::sort(sortedInitializations, [](const PipeCapacityInitInfo &lhs,
                                         const PipeCapacityInitInfo &rhs) {
      return lhs.semaphoreIndex < rhs.semaphoreIndex;
    });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefTy = MemRefType::get({1}, builder.getI32Type());
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto &perFuncCounters = senderCapacityCounters[func];
    for (const PipeCapacityInitInfo &init : sortedInitializations) {
      Value capacitySemaphorePtr =
          buildLocalSemaphorePtr(loc, builder, init.semaphoreIndex);
      Value initialCapacity =
          arith::ConstantIntOp::create(builder, loc, init.initialCapacity, 32);
      ttk::NocSemaphoreSetOp::create(builder, loc, capacitySemaphorePtr,
                                     initialCapacity);
      // The sender tracks its cumulative acquired count in a kernel-local
      // counter and waits for the capacity semaphore to reach it, so the
      // receiver's remote increment stays the only writer of the shared word.
      auto counter = memref::AllocaOp::create(builder, loc, counterMemrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, counter,
                              ValueRange{zeroIdx});
      perFuncCounters[init.semaphoreIndex] = counter.getResult();
    }
  }
}

/// Lower CB -> Pipe copy: write source DFB data to the selected destination
/// address, then signal arrival.
LogicalResult
lowerPipeTransferSend(PipeTransferSendOp op, Value srcCB, bool isConsumerCB,
                      const PipeResourcePlan &pipeResourcePlan,
                      const PipeCapacityPlan *pipeCapacityPlan,
                      const PipeNetCounterMap *senderCapacityCounters,
                      ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  PipeTransferCreateOp createOp =
      findPipeTransferCreateForTransfer(op.getTransfer());
  assert(createOp &&
         "pipe resource plan analysis already validated transfer provenance");
  auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(createOp, pipeResourcePlan);
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

  ArrayRef<PipeCapacityAcquireInfo> capacityAcquires =
      pipeCapacityPlan ? pipeCapacityPlan->lookupAcquires(op)
                       : ArrayRef<PipeCapacityAcquireInfo>{};
  if (!capacityAcquires.empty()) {
    FuncOp senderFunc = op->getParentOfType<FuncOp>();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (const PipeCapacityAcquireInfo &capacityAcquire : capacityAcquires) {
      Value capacitySemaphorePtr =
          buildLocalSemaphorePtr(loc, rewriter, capacityAcquire.semaphoreIndex);
      Value senderCapacityCounter;
      if (senderCapacityCounters) {
        auto funcIt = senderCapacityCounters->find(senderFunc);
        if (funcIt != senderCapacityCounters->end()) {
          auto counterIt = funcIt->second.find(capacityAcquire.semaphoreIndex);
          if (counterIt != funcIt->second.end()) {
            senderCapacityCounter = counterIt->second;
          }
        }
      }
      if (!senderCapacityCounter) {
        // Counter pre-allocation in initializePipeCapacitySemaphores is a hard
        // precondition; a missing counter is a pipeline-ordering bug, not a
        // legalization miss.
        op.emitError("pipe capacity acquire without sender counter; "
                     "initializePipeCapacitySemaphores must run before "
                     "convert-ttl-to-ttkernel");
        return failure();
      }
      // Advance the sender's cumulative acquired count and block until the
      // capacity semaphore reaches it. The receiver's remote increment is the
      // only writer of the shared semaphore, so the acquire never writes it.
      Value previousAcquired = memref::LoadOp::create(
          rewriter, loc, senderCapacityCounter, ValueRange{zeroIdx});
      Value capacityCount = arith::ConstantIntOp::create(
          rewriter, loc, capacityAcquire.count, 32);
      Value nextAcquired =
          arith::AddIOp::create(rewriter, loc, previousAcquired, capacityCount);
      memref::StoreOp::create(rewriter, loc, nextAcquired,
                              senderCapacityCounter, ValueRange{zeroIdx});
      ttk::SemaphoreWaitMinOp::create(rewriter, loc, capacitySemaphorePtr,
                                      nextAcquired);
    }
  } else {
    ReadyCounterAddressInfo readyCounterInfo =
        getReadyCounterAddressInfo(op, pipeResource, pipeResourcePlan);
    int64_t expectedReceiverPosts =
        isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
    Value senderReadyCounterAddr =
        buildReadyCounterAddress(loc, readyCounterInfo, rewriter);
    auto senderReadyCounterPtr = ttk::CastToL1PtrOp::create(
        rewriter, loc, l1PtrTy, senderReadyCounterAddr);
    auto expectedReadyCount = arith::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(expectedReceiverPosts));
    ttk::SemaphoreWaitOp::create(rewriter, loc, senderReadyCounterPtr,
                                 expectedReadyCount);
    auto readyCounterResetValue =
        arith::ConstantIndexOp::create(rewriter, loc, 0);
    ttk::NocSemaphoreSetOp::create(rewriter, loc, senderReadyCounterPtr,
                                   readyCounterResetValue);
  }

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

  Value dstAddr;
  if (pipeResource.addressStorage.usesComputedReceiverDFB()) {
    assert(pipeResource.addressStorage.computedAddress.has_value() &&
           "computed pipe missing computed-address info");
    dstAddr = buildComputedReceiverDFBDestinationAddress(
        loc, *pipeResource.addressStorage.computedAddress, rewriter);
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

  // Signal receiver completion.
  if (pipeType.hasSingleReceiver()) {
    // Point-to-point increments the destination receiver-completion counter.
    auto receiverCompletionCounterSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, completionInfo.receiverCompletionSemIdx);
    auto receiverCompletionCounterAddr = ttk::GetSemaphoreOp::create(
        rewriter, loc, receiverCompletionCounterSemIdx);
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
    auto receiverCompletionCounterSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, completionInfo.receiverCompletionSemIdx);
    auto receiverCompletionCounterAddr = ttk::GetSemaphoreOp::create(
        rewriter, loc, receiverCompletionCounterSemIdx);

    // HW multicast auto-excludes the sender; num_dests counts only remote
    // receivers. tt-metal has no inc_multicast_loopback primitive, so the
    // source node's receiver-completion counter is incremented locally below.
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
      auto localReceiverCompletionNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, srcXTranslated, srcYTranslated,
          receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
  }

  // Both branches signal completion with non-posted atomics; the send ttl.wait
  // lowers to a no-op, so this barrier is the only flush before the kernel
  // exits. Without it receivers can observe stale completion counts.
  ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    const PipeCapacityPlan *pipeCapacityPlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  PipeTransferCreateOp createOp =
      findPipeTransferCreateForTransfer(op.getTransfer());
  assert(createOp &&
         "pipe resource plan analysis already validated transfer provenance");
  auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(createOp, pipeResourcePlan);

  int64_t nocIdx = getNocIndex(op);
  auto indexTy = rewriter.getIndexType();

  // Preflight the only fallible validation before emitting any ops, so a match
  // failure leaves no partially-built IR for the conversion driver to roll
  // back.
  bool usesComputedReceiverDFB =
      pipeResource.addressStorage.usesComputedReceiverDFB();
  std::optional<ReceiverPublishedAddressInfo> publishedAddressInfo;
  if (!usesComputedReceiverDFB) {
    FailureOr<ReceiverPublishedAddressInfo> info =
        getReceiverPublishedAddressInfo(op, dst, rewriter);
    if (failed(info)) {
      return failure();
    }
    publishedAddressInfo = *info;
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
        dst, loc, *publishedAddressInfo, rewriter);
    Value tableAddress =
        buildAddressTableAddress(loc, addressTableInfo, rewriter);
    // [Device 2.0] This is a receiver-authored write to a typed address table;
    // only this lowering should select the current inline NoC write primitive.
    auto byteEnableAll = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
    ttk::NocInlineDwWriteOp::create(rewriter, loc, srcXTranslated,
                                    srcYTranslated, tableAddress,
                                    publishedAddress, byteEnableAll, nocVal);
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  if (!pipeCapacityPlan || !pipeCapacityPlan->usesCapacityProtocol(createOp)) {
    ReadyCounterAddressInfo readyCounterInfo =
        getReadyCounterAddressInfo(op, pipeResource, pipeResourcePlan);
    Value senderReadyCounterAddr =
        buildReadyCounterAddress(loc, readyCounterInfo, rewriter);
    auto senderReadyCounterNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, srcXTranslated, srcYTranslated,
                                  senderReadyCounterAddr, nocVal);
    auto readyCounterIncrement =
        arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, senderReadyCounterNocAddr.getResult(),
        readyCounterIncrement, nocVal, /*posted=*/BoolAttr());
  }

  auto token = UnrealizedConversionCastOp::create(
      rewriter, loc, op.getToken().getType(), ValueRange{});
  rewriter.replaceOp(op, token.getResult(0));
  return success();
}

static Value computeDFBPopNumTiles(CBPopOp op, Value originalCb,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc) {
  if (auto attr = op.getNumTilesAttr()) {
    return arith::ConstantIntOp::create(rewriter, loc, attr.getInt(), 32);
  }
  auto ttlCbTy = getTTLCBType(originalCb);
  assert(ttlCbTy && "lowerCBPop already verified the DFB type");
  return arith::ConstantIntOp::create(rewriter, loc,
                                      ttlCbTy.getElementsPerBlock(), 32);
}

LogicalResult lowerCBPop(CBPopOp op, Value cb,
                         const PipeCapacityPlan *pipeCapacityPlan,
                         ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value originalCb = op.getCb();
  auto ttlCbTy = getTTLCBType(originalCb);
  if (!ttlCbTy) {
    return rewriter.notifyMatchFailure(op, "failed to get TTL DFB type");
  }

  auto convertedCb = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
  if (failed(convertedCb)) {
    return rewriter.notifyMatchFailure(op, "failed to convert DFB operand");
  }

  Value numTiles = computeDFBPopNumTiles(op, originalCb, rewriter, loc);
  ttk::CBPopFrontOp::create(rewriter, loc, *convertedCb, numTiles);

  ArrayRef<PipeCapacityReleaseInfo> releases =
      pipeCapacityPlan ? pipeCapacityPlan->lookupReleases(op)
                       : ArrayRef<PipeCapacityReleaseInfo>{};
  if (!releases.empty()) {
    int64_t nocIdx = getNocIndex(op);
    Value nocVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));
    for (const PipeCapacityReleaseInfo &release : releases) {
      lowerPipeCapacityRelease(loc, release, nocVal, rewriter);
    }
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

  rewriter.eraseOp(op);
  return success();
}

/// Lower the receiver completion wait with a per-PipeNet runtime counter.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op,
                                    const PipeNetCounterMap *counters,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto tokenType = mlir::cast<PipeTokenType>(op.getToken().getType());
  auto completionIt =
      pipeResourcePlan.completionWaits.find(tokenType.getPipeNetId());
  if (completionIt == pipeResourcePlan.completionWaits.end()) {
    op.emitError("pipe transfer wait references PipeNet ")
        << tokenType.getPipeNetId() << " with no completion resource";
    return failure();
  }
  PipeCompletionWaitInfo completionInfo = completionIt->second;

  Value waitProgressCounter;
  if (counters) {
    auto func = op->getParentOfType<func::FuncOp>();
    auto fIt = counters->find(func);
    if (fIt != counters->end()) {
      auto pIt = fIt->second.find(tokenType.getPipeNetId());
      if (pIt != fIt->second.end()) {
        waitProgressCounter = pIt->second;
      }
    }
  }
  if (!waitProgressCounter) {
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

  auto receiverCompletionCounterSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, completionInfo.receiverCompletionSemIdx);
  auto receiverCompletionCounterAddr = ttk::GetSemaphoreOp::create(
      rewriter, loc, receiverCompletionCounterSemIdx);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  auto receiverCompletionCounterPtr = ttk::CastToL1PtrOp::create(
      rewriter, loc, l1PtrTy, receiverCompletionCounterAddr);

  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto previousWaitCount = memref::LoadOp::create(
      rewriter, loc, waitProgressCounter, ValueRange{zeroIdx});
  auto oneI32 = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
  auto nextWaitCount =
      arith::AddIOp::create(rewriter, loc, previousWaitCount, oneI32);
  memref::StoreOp::create(rewriter, loc, nextWaitCount, waitProgressCounter,
                          ValueRange{zeroIdx});
  ttk::SemaphoreWaitMinOp::create(rewriter, loc, receiverCompletionCounterPtr,
                                  nextWaitCount);

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

/// Operation kind that changes source-node rendezvous state for a pipe
/// transfer.
///
/// Address-table slots and sender-ready counters are live from receive post
/// until send consumes the posted state. Waits use receiver-completion
/// resources, so they are intentionally not rendezvous events.
enum class PipeTransferRendezvousEventKind {
  Post,
  Send,
};

struct PipeTransferAllocationUnit;

/// One ordered post/send operation used to validate bounded rendezvous depth.
struct PipeTransferRendezvousEvent {
  static PipeTransferRendezvousEvent post(Operation *op) {
    return {op, PipeTransferRendezvousEventKind::Post};
  }

  static PipeTransferRendezvousEvent send(Operation *op) {
    return {op, PipeTransferRendezvousEventKind::Send};
  }

  /// Pipe transfer post or send operation.
  Operation *op;
  /// Whether the operation creates or consumes one posted rendezvous phase.
  PipeTransferRendezvousEventKind kind;

  /// Block-local operation order used for queue-depth validation.
  bool operator<(const PipeTransferRendezvousEvent &rhs) const {
    return op->isBeforeInBlock(rhs.op);
  }

  /// Update live post count for one event in block order.
  LogicalResult updateLivePosts(const PipeTransferAllocationUnit &unit,
                                int64_t &livePosts, int64_t maxLivePosts) const;
};

/// Allocation unit for source-node pipe rendezvous resources.
///
/// Repeated static transfer operations for the same logical pipe share one
/// unit so they preserve the existing per-pipe protocol state. The interval
/// bounds the lifetime of the unit's address-table slot and sender-ready
/// counter for deterministic coloring.
struct PipeTransferAllocationUnit {
  /// Pipe transfer create operations represented by this allocation unit.
  SmallVector<Operation *> transferCreateOps;

  /// Post/send events used to reject unsupported queue depth in linear blocks.
  SmallVector<PipeTransferRendezvousEvent> rendezvousEvents;

  /// Logical pipe whose source node owns this unit's rendezvous resources.
  PipeKey pipe;

  /// Pipe type cached from the first create op for resource-plan construction.
  PipeType pipeType;

  /// Collective takes precedence when cloned regions produce mixed contracts.
  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;

  /// Stable tie-breaker for deterministic allocation.
  int64_t ordinal = 0;

  /// Conservative post-to-send lifetime for source-node rendezvous resources.
  OperationLiveInterval interval;

  /// Assigned first-fit color within the source node's allocation group.
  int64_t resourceColor = 0;

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

static LogicalResult
emitUnsupportedQueueDepth(Operation *op,
                          const PipeTransferAllocationUnit &unit) {
  return op->emitError()
         << "pipe transfer for pipe net " << unit.pipe.pipeNetId << " src("
         << unit.pipe.srcX << ", " << unit.pipe.srcY << ") dst("
         << unit.pipe.dstStartX << ", " << unit.pipe.dstStartY << ") to("
         << unit.pipe.dstEndX << ", " << unit.pipe.dstEndY
         << ") requires queue depth greater than 1; current lowering supports "
            "one live receive post per pipe before each send";
}

LogicalResult PipeTransferRendezvousEvent::updateLivePosts(
    const PipeTransferAllocationUnit &unit, int64_t &livePosts,
    int64_t maxLivePosts) const {
  switch (kind) {
  case PipeTransferRendezvousEventKind::Post:
    ++livePosts;
    if (livePosts > maxLivePosts) {
      return emitUnsupportedQueueDepth(op, unit);
    }
    return success();
  case PipeTransferRendezvousEventKind::Send:
    if (livePosts > 0) {
      --livePosts;
    }
    return success();
  }
  llvm_unreachable("unknown pipe transfer rendezvous event kind");
}

static Region *findRegionOwnedByAncestor(Operation *op, Operation *ancestorOp) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parentOp = region->getParentOp();
    if (parentOp == ancestorOp) {
      return region;
    }
    region = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return nullptr;
}

static bool areInMutuallyExclusiveIfRegions(Operation *lhsOp,
                                            Operation *rhsOp) {
  for (Operation *ancestorOp = lhsOp->getParentOp(); ancestorOp;
       ancestorOp = ancestorOp->getParentOp()) {
    if (!isa<scf::IfOp>(ancestorOp)) {
      continue;
    }
    Region *lhsRegion = findRegionOwnedByAncestor(lhsOp, ancestorOp);
    Region *rhsRegion = findRegionOwnedByAncestor(rhsOp, ancestorOp);
    if (lhsRegion && rhsRegion && lhsRegion != rhsRegion) {
      return true;
    }
  }
  return false;
}

static LogicalResult
validateMaxLivePosts(const PipeTransferAllocationUnit &unit,
                     int64_t maxLivePosts) {
  llvm::MapVector<Block *, SmallVector<PipeTransferRendezvousEvent>>
      eventsByBlock;
  SmallVector<Operation *> postOps;
  for (const PipeTransferRendezvousEvent &event : unit.rendezvousEvents) {
    if (event.kind == PipeTransferRendezvousEventKind::Post) {
      postOps.push_back(event.op);
    }
    eventsByBlock[event.op->getBlock()].push_back(event);
  }

  if (postOps.size() <= static_cast<size_t>(maxLivePosts)) {
    return success();
  }

  for (size_t lhsIndex = 0; lhsIndex < postOps.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < postOps.size();
         ++rhsIndex) {
      Operation *lhsOp = postOps[lhsIndex];
      Operation *rhsOp = postOps[rhsIndex];
      if (lhsOp->getBlock() != rhsOp->getBlock() &&
          !areInMutuallyExclusiveIfRegions(lhsOp, rhsOp)) {
        return emitUnsupportedQueueDepth(rhsOp, unit);
      }
    }
  }

  for (auto &entry : eventsByBlock) {
    SmallVector<PipeTransferRendezvousEvent> &events = entry.second;
    if (events.size() <= static_cast<size_t>(maxLivePosts)) {
      continue;
    }

    llvm::sort(events, std::less<PipeTransferRendezvousEvent>());

    int64_t livePosts = 0;
    for (const PipeTransferRendezvousEvent &event : events) {
      if (failed(event.updateLivePosts(unit, livePosts, maxLivePosts))) {
        return failure();
      }
    }
  }

  return success();
}

static bool pipeTransferIntervalsOverlap(const PipeTransferAllocationUnit &lhs,
                                         const PipeTransferAllocationUnit &rhs,
                                         const DominanceInfo &dominanceInfo) {
  return intervalsOverlap(lhs.interval, rhs.interval, dominanceInfo);
}

static FailureOr<SmallVector<PipeTransferAllocationUnit>>
collectPipeTransferAllocationUnits(ModuleOp mod,
                                   const DominanceInfo &dominanceInfo,
                                   const PostDominanceInfo &postDominanceInfo) {
  SmallVector<PipeTransferAllocationUnit> units;
  llvm::MapVector<Operation *, unsigned> indexByTransferCreateOp;
  llvm::MapVector<PipeKey, unsigned> indexByPipe;
  int64_t nextOrdinal = 0;
  int64_t nextEventOrdinal = 0;

  auto getOrCreateUnit =
      [&](Operation *protocolOp,
          Value transfer) -> FailureOr<PipeTransferAllocationUnit *> {
    FailureOr<PipeTransferCreateOp> createOp =
        getPipeTransferCreate(protocolOp, transfer);
    if (failed(createOp)) {
      return failure();
    }

    Operation *transferCreateOp = (*createOp).getOperation();
    auto existing = indexByTransferCreateOp.find(transferCreateOp);
    if (existing != indexByTransferCreateOp.end()) {
      return &units[existing->second];
    }

    auto pipeType = mlir::cast<PipeType>((*createOp).getPipe().getType());
    PipeKey pipe = getPipeKey(pipeType);
    PipeTransferContract transferContract = getPipeTransferContract(*createOp);
    auto existingPipe = indexByPipe.find(pipe);
    if (existingPipe != indexByPipe.end()) {
      PipeTransferAllocationUnit &unit = units[existingPipe->second];
      unit.transferCreateOps.push_back(transferCreateOp);
      if (isCollectiveTransfer(transferContract)) {
        unit.transferContract = PipeTransferContract::Collective;
      }
      indexByTransferCreateOp.insert({transferCreateOp, existingPipe->second});
      return &unit;
    }

    PipeTransferAllocationUnit unit;
    unit.transferCreateOps.push_back(transferCreateOp);
    unit.pipe = pipe;
    unit.pipeType = pipeType;
    unit.transferContract = transferContract;
    unit.ordinal = nextOrdinal++;
    indexByTransferCreateOp.insert({transferCreateOp, units.size()});
    indexByPipe.insert({pipe, units.size()});
    units.push_back(unit);
    return &units.back();
  };

  // Resource allocation depends only on receive posts and sends. Walk the
  // module once in operation order to form per-pipe allocation units, record
  // rendezvous events for queue-depth validation, and build post-to-send live
  // intervals for coloring.
  WalkResult walkResult = mod.walk([&](Operation *op) {
    if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
      int64_t eventOrdinal = nextEventOrdinal++;
      FailureOr<PipeTransferAllocationUnit *> unit =
          getOrCreateUnit(op, postOp.getTransfer());
      if (failed(unit)) {
        return WalkResult::interrupt();
      }
      (*unit)->rendezvousEvents.push_back(
          PipeTransferRendezvousEvent::post(op));
      updateIntervalStart((*unit)->interval, op, eventOrdinal, dominanceInfo);
      return WalkResult::advance();
    }

    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      FailureOr<PipeTransferAllocationUnit *> unit =
          getOrCreateUnit(op, sendOp.getTransfer());
      if (failed(unit)) {
        return WalkResult::interrupt();
      }
      (*unit)->rendezvousEvents.push_back(
          PipeTransferRendezvousEvent::send(op));
      updateIntervalEnd((*unit)->interval, op, dominanceInfo);
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  for (PipeTransferAllocationUnit &unit : units) {
    if (failed(validateMaxLivePosts(unit, /*maxLivePosts=*/1))) {
      return failure();
    }
    finalizeInterval(unit.interval, dominanceInfo, postDominanceInfo);
  }

  return units;
}

using SourceColorMap =
    llvm::MapVector<PipeSourceKey, SmallVector<SmallVector<unsigned>>>;

static SourceColorMap
assignLiveIntervalColors(MutableArrayRef<PipeTransferAllocationUnit> units,
                         const DominanceInfo &dominanceInfo) {
  llvm::MapVector<PipeSourceKey, SmallVector<unsigned>> unitIndicesBySource;
  for (unsigned index = 0, size = units.size(); index < size; ++index) {
    unitIndicesBySource[getPipeSourceKey(units[index].pipeType)].push_back(
        index);
  }

  SourceColorMap colorUsersBySource;
  for (auto &entry : unitIndicesBySource) {
    SmallVector<SmallVector<unsigned>> colorUsers =
        assignGreedyIntervalColors<unsigned>(
            entry.second,
            [&](unsigned lhsIndex, unsigned rhsIndex) {
              return std::less<PipeTransferAllocationUnit>()(units[lhsIndex],
                                                             units[rhsIndex]);
            },
            [&](unsigned lhsIndex, unsigned rhsIndex) {
              return pipeTransferIntervalsOverlap(
                  units[lhsIndex], units[rhsIndex], dominanceInfo);
            });

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      for (unsigned unitIndex : indexedColor.value()) {
        units[unitIndex].resourceColor = indexedColor.index();
      }
    }

    colorUsersBySource.insert({entry.first, std::move(colorUsers)});
  }

  return colorUsersBySource;
}

static std::pair<int64_t, int64_t>
countPostAndSendEvents(const PipeTransferAllocationUnit &unit) {
  int64_t postCount = 0;
  int64_t sendCount = 0;
  for (const PipeTransferRendezvousEvent &event : unit.rendezvousEvents) {
    switch (event.kind) {
    case PipeTransferRendezvousEventKind::Post:
      ++postCount;
      break;
    case PipeTransferRendezvousEventKind::Send:
      ++sendCount;
      break;
    }
  }
  return {postCount, sendCount};
}

static bool usesSenderReadyCounter(const PipeTransferAllocationUnit &unit,
                                   const PipeCapacityPlan *pipeCapacityPlan) {
  if (!pipeCapacityPlan) {
    return true;
  }
  for (Operation *transferCreateOp : unit.transferCreateOps) {
    if (!pipeCapacityPlan->usesCapacityProtocol(
            llvm::cast<PipeTransferCreateOp>(transferCreateOp))) {
      return true;
    }
  }
  return false;
}

static std::optional<FuncOp>
getSingleSenderFunc(const PipeTransferAllocationUnit &unit) {
  std::optional<FuncOp> senderFunc;
  for (const PipeTransferRendezvousEvent &event : unit.rendezvousEvents) {
    if (event.kind != PipeTransferRendezvousEventKind::Send) {
      continue;
    }
    FuncOp func = event.op->getParentOfType<FuncOp>();
    if (!func) {
      return std::nullopt;
    }
    if (senderFunc && *senderFunc != func) {
      return std::nullopt;
    }
    senderFunc = func;
  }
  return senderFunc;
}

static int64_t getReceiverDFBBlockStrideBytes(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.dfbType.getElementsPerBlock() * tileType.getSizeBytes();
}

static int64_t getReceiverDFBStaticByteOffset(const ReceiverDFBInfo &info) {
  auto tileType = llvm::cast<ttcore::TileType>(info.dfbType.getElementType());
  return info.staticTileOffset * tileType.getSizeBytes();
}

static const PipeEdge *lookupPipeEdge(const PipeGraph &pipeGraph,
                                      const PipeKey &pipe) {
  for (const PipeEdge &pipeEdge : pipeGraph.getPipeEdges()) {
    if (pipeEdge.pipe == pipe) {
      return &pipeEdge;
    }
  }
  return nullptr;
}

static std::optional<int64_t>
getUniformReceiverBatchSize(const PipeGraph &pipeGraph,
                            const PipeEdge &pipeEdge) {
  std::optional<int64_t> receiverBatchSize;
  for (PipeReceiverEndpointId endpointId :
       pipeGraph.getPipeReceiverEndpoints(pipeEdge.id)) {
    const PipeReceiverEndpoint &endpoint =
        pipeGraph.getPipeReceiverEndpoint(endpointId);
    const PipeReceiverDFBNode &receiverDFBNode =
        pipeGraph.getReceiverDFBNode(endpoint.receiverDFBNode);
    if (receiverBatchSize &&
        *receiverBatchSize != receiverDFBNode.receiverBatchSize) {
      return std::nullopt;
    }
    receiverBatchSize = receiverDFBNode.receiverBatchSize;
  }
  return receiverBatchSize;
}

static bool hasProvenPipeOnlyReceiverStreams(const PipeGraph &pipeGraph,
                                             const PipeEdge &pipeEdge) {
  for (PipeReceiverEndpointId endpointId :
       pipeGraph.getPipeReceiverEndpoints(pipeEdge.id)) {
    const PipeReceiverEndpoint &endpoint =
        pipeGraph.getPipeReceiverEndpoint(endpointId);
    const PipeReceiverDFBNode &receiverDFBNode =
        pipeGraph.getReceiverDFBNode(endpoint.receiverDFBNode);
    if (!receiverDFBNode.hasProvenPipeOnlyStream) {
      return false;
    }
  }
  return true;
}

static std::optional<int64_t>
getStaticReceiverByteOffset(const ReceiverDFBInfo &receiverInfo,
                            int64_t receiverBatchSize) {
  // Static receiver addresses use one compile-time base per receiver DFB.
  // This is exact only when the physical DFB ring period equals the receiver
  // batch; otherwise the sender would need dynamic batch-iteration state to
  // distinguish repeated visits to the same physical slot.
  if (receiverInfo.blockCount != receiverBatchSize) {
    return std::nullopt;
  }
  int64_t blockOffsetBytes = *receiverInfo.receiverSlotIndex *
                             getReceiverDFBBlockStrideBytes(receiverInfo);
  return blockOffsetBytes + getReceiverDFBStaticByteOffset(receiverInfo);
}

static bool isComputedAddressCandidate(const PipeTransferAllocationUnit &unit,
                                       const ReceiverDFBInfo &receiverInfo,
                                       const PipeGraph &pipeGraph,
                                       int64_t &staticByteOffset) {
  if (!receiverInfo.hasStaticTileOffset) {
    return false;
  }
  if (!llvm::isa<ttcore::TileType>(receiverInfo.dfbType.getElementType())) {
    return false;
  }
  if (!receiverInfo.receiverSlotIndex.has_value()) {
    return false;
  }
  auto [postCount, sendCount] = countPostAndSendEvents(unit);
  if (postCount != 1 || sendCount != 1) {
    return false;
  }
  const PipeEdge *pipeEdge = lookupPipeEdge(pipeGraph, unit.pipe);
  if (!pipeEdge) {
    return false;
  }
  // Static receiver addresses are derived from the pipe graph's physical slot
  // assignment. Non-pipe DFB traffic can advance the hardware ring without a
  // pipe post, so computed addressing requires the graph to prove that the
  // receiver stream contains only pipe-delivered blocks.
  if (!hasProvenPipeOnlyReceiverStreams(pipeGraph, *pipeEdge)) {
    return false;
  }
  std::optional<int64_t> receiverBatchSize =
      getUniformReceiverBatchSize(pipeGraph, *pipeEdge);
  if (!receiverBatchSize) {
    return false;
  }
  std::optional<int64_t> offset =
      getStaticReceiverByteOffset(receiverInfo, *receiverBatchSize);
  if (!offset) {
    return false;
  }
  staticByteOffset = *offset;
  return true;
}

struct ComputedAddressPlan {
  llvm::DenseMap<unsigned, PipeComputedAddressInfo> infoByUnitIndex;
};

static ComputedAddressPlan
buildComputedAddressPlan(ModuleOp mod,
                         MutableArrayRef<PipeTransferAllocationUnit> units,
                         const PipeGraph &pipeGraph, bool updateFunctionAttrs) {
  ComputedAddressPlan plan;

  struct Candidate {
    unsigned unitIndex = 0;
    FuncOp senderFunc;
    const ReceiverDFBInfo *receiverInfo = nullptr;
    int64_t staticByteOffset = 0;
  };
  SmallVector<Candidate> candidates;
  llvm::MapVector<FuncOp, llvm::SmallSetVector<int64_t, 4>> dfbIndicesByFunc;

  for (auto indexedUnit : llvm::enumerate(units)) {
    PipeTransferAllocationUnit &unit = indexedUnit.value();
    const ReceiverDFBInfo *receiverInfo =
        pipeGraph.lookupReceiverDFB(unit.pipe);
    int64_t staticByteOffset = 0;
    if (!receiverInfo ||
        !isComputedAddressCandidate(unit, *receiverInfo, pipeGraph,
                                    staticByteOffset)) {
      continue;
    }
    std::optional<FuncOp> senderFunc = getSingleSenderFunc(unit);
    if (!senderFunc) {
      continue;
    }
    candidates.push_back(Candidate{static_cast<unsigned>(indexedUnit.index()),
                                   *senderFunc, receiverInfo,
                                   staticByteOffset});
    dfbIndicesByFunc[*senderFunc].insert(receiverInfo->dfbIndex);
  }

  if (candidates.empty()) {
    return plan;
  }

  OpBuilder builder(mod.getContext());
  int64_t defaultBaseCTA = getNextAvailableDFBIndex(mod);
  llvm::DenseMap<FuncOp, int64_t> baseCTAByFunc;
  llvm::DenseMap<FuncOp, SmallVector<int64_t>> sortedDFBIndicesByFunc;
  for (auto &[func, dfbSet] : dfbIndicesByFunc) {
    SmallVector<int64_t> sortedDFBIndices(dfbSet.begin(), dfbSet.end());
    llvm::sort(sortedDFBIndices);
    sortedDFBIndicesByFunc[func] = sortedDFBIndices;

    int64_t baseCTA = defaultBaseCTA;
    if (auto attr = func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
      baseCTA = attr.getInt();
    }
    baseCTAByFunc[func] = baseCTA;

    SmallVector<int32_t> dfbAttrs;
    dfbAttrs.reserve(sortedDFBIndices.size());
    for (int64_t dfbIndex : sortedDFBIndices) {
      dfbAttrs.push_back(static_cast<int32_t>(dfbIndex));
    }
    if (updateFunctionAttrs) {
      func->setAttr(kPipeComputedAddressDFBIndicesAttrName,
                    builder.getDenseI32ArrayAttr(dfbAttrs));
      func->setAttr(
          kBaseCTAIndexAttrName,
          builder.getI32IntegerAttr(baseCTA + sortedDFBIndices.size()));
    }
  }

  for (const Candidate &candidate : candidates) {
    FuncOp senderFunc = candidate.senderFunc;
    const SmallVector<int64_t> &dfbIndices = sortedDFBIndicesByFunc[senderFunc];
    auto dfbIt = llvm::find(dfbIndices, candidate.receiverInfo->dfbIndex);
    assert(dfbIt != dfbIndices.end() && "candidate DFB missing from func list");
    int64_t baseCompileTimeArgIndex =
        baseCTAByFunc[senderFunc] + std::distance(dfbIndices.begin(), dfbIt);

    assert(llvm::isInt<32>(candidate.staticByteOffset) &&
           "computed receiver byte offset must fit the i32 NoC address space");
    plan.infoByUnitIndex[candidate.unitIndex] = PipeComputedAddressInfo{
        candidate.receiverInfo->dfbIndex, baseCompileTimeArgIndex,
        candidate.staticByteOffset};
  }

  return plan;
}

LogicalResult buildPipeResourcePlan(ModuleOp mod, const PipeGraph &pipeGraph,
                                    PipeResourcePlan &info,
                                    bool enableComputedAddresses,
                                    const PipeCapacityPlan *pipeCapacityPlan,
                                    bool updateComputedAddressAttrs) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, dominanceInfo, postDominanceInfo);
  if (failed(maybeUnits)) {
    return failure();
  }
  SmallVector<PipeTransferAllocationUnit> &units = *maybeUnits;
  SourceColorMap colorUsersBySource =
      assignLiveIntervalColors(units, dominanceInfo);
  ComputedAddressPlan computedAddressPlan;
  if (enableComputedAddresses) {
    computedAddressPlan = buildComputedAddressPlan(mod, units, pipeGraph,
                                                   updateComputedAddressAttrs);
  }

  llvm::SmallSetVector<int64_t, 4> activePipeNetIds;
  for (const PipeTransferAllocationUnit &unit : units) {
    activePipeNetIds.insert(unit.pipe.pipeNetId);
  }

  SmallVector<int64_t> sortedPipeNetIds(activePipeNetIds.begin(),
                                        activePipeNetIds.end());
  llvm::sort(sortedPipeNetIds);

  int64_t firstSourceLocalReadyCounterSemIdx = 0;
  for (int64_t pipeNetId : sortedPipeNetIds) {
    int64_t receiverCompletionSemIdx = getReceiverCompletionSemIdx(pipeNetId);
    info.completionWaits[pipeNetId] =
        PipeCompletionWaitInfo{pipeNetId, receiverCompletionSemIdx};
    firstSourceLocalReadyCounterSemIdx = std::max(
        firstSourceLocalReadyCounterSemIdx, receiverCompletionSemIdx + 1);
  }

  llvm::MapVector<PipeSourceKey, llvm::DenseMap<int64_t, int64_t>>
      readyColorBySourceColor;
  int64_t maxReadyCountersPerSource = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    int64_t nextReadyColor = 0;
    llvm::DenseMap<int64_t, int64_t> &readyColors =
        readyColorBySourceColor[sourceKey];
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      bool colorUsesSenderReady = false;
      for (unsigned unitIndex : indexedColor.value()) {
        if (usesSenderReadyCounter(units[unitIndex], pipeCapacityPlan)) {
          colorUsesSenderReady = true;
          break;
        }
      }
      if (colorUsesSenderReady) {
        readyColors[static_cast<int64_t>(indexedColor.index())] =
            nextReadyColor++;
      }
    }
    maxReadyCountersPerSource =
        std::max(maxReadyCountersPerSource, nextReadyColor);
  }

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
      for (unsigned color = 0, colorCount = readyColors.size();
           color < colorCount; ++color) {
        indices.push_back(nextGlobalSemaphoreIndex++);
      }
    }
  }

  llvm::MapVector<PipeSourceKey, llvm::DenseMap<int64_t, int64_t>>
      addressColorBySourceColor;
  int64_t maxAddressTableBytes = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    int64_t nextAddressColor = 0;
    llvm::DenseMap<int64_t, int64_t> &addressColors =
        addressColorBySourceColor[sourceKey];
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      bool colorUsesAddressTable = false;
      for (unsigned unitIndex : indexedColor.value()) {
        if (computedAddressPlan.infoByUnitIndex.find(unitIndex) ==
            computedAddressPlan.infoByUnitIndex.end()) {
          colorUsesAddressTable = true;
          break;
        }
      }
      if (colorUsesAddressTable) {
        addressColors[static_cast<int64_t>(indexedColor.index())] =
            nextAddressColor++;
      }
    }
    maxAddressTableBytes = std::max<int64_t>(
        maxAddressTableBytes, nextAddressColor * kPipeAddressWordBytes);
  }

  for (auto indexedUnit : llvm::enumerate(units)) {
    const PipeTransferAllocationUnit &unit = indexedUnit.value();
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    std::optional<PipeReadyCounterInfo> readyCounter;
    if (usesSenderReadyCounter(unit, pipeCapacityPlan)) {
      auto sourceIt = readyColorBySourceColor.find(sourceKey);
      assert(sourceIt != readyColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      int64_t readyColor = colorIt->second;
      readyCounter = PipeReadyCounterInfo::localSemaphore(
          firstSourceLocalReadyCounterSemIdx + readyColor);
      if (useGlobalReadyCounters) {
        auto globalIt = globalIndexBySourceColor.find(sourceKey);
        assert(globalIt != globalIndexBySourceColor.end());
        assert(readyColor < static_cast<int64_t>(globalIt->second.size()));
        readyCounter =
            PipeReadyCounterInfo::globalSemaphore(globalIt->second[readyColor]);
      }
    }

    auto computedIt = computedAddressPlan.infoByUnitIndex.find(
        static_cast<unsigned>(indexedUnit.index()));
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
        readyCounter,
        addressStorage,
    };
    for (Operation *transferCreateOp : unit.transferCreateOps) {
      info.resources.insert({transferCreateOp, pipeResource});
    }
  }

  info.sramScratch.bytes =
      maxAddressTableBytes == 0
          ? 0
          : alignTo(maxAddressTableBytes, kPipeSramScratchAlignmentBytes);
  return success();
}

PipeResourceRequirements
getPipeResourceRequirements(const PipeResourcePlan &info,
                            const PipeCapacityPlan *pipeCapacityPlan) {
  struct RequirementsObserver final : PipeReadyCounterObserver {
    int64_t highestSyncSemaphoreIndex = -1;
    int64_t highestGlobalSemaphoreIndex = -1;

    void observeLocalSemaphore(int64_t index) override {
      highestSyncSemaphoreIndex = std::max(highestSyncSemaphoreIndex, index);
    }

    void observeGlobalSemaphore(int64_t index) override {
      highestGlobalSemaphoreIndex =
          std::max(highestGlobalSemaphoreIndex, index);
    }
  };

  RequirementsObserver observer;
  for (const auto &[pipeNetId, completion] : info.completionWaits) {
    (void)pipeNetId;
    observer.observeLocalSemaphore(completion.receiverCompletionSemIdx);
  }
  for (const auto &[transferCreateOp, resource] : info.resources) {
    (void)transferCreateOp;
    if (resource.readyCounter) {
      resource.readyCounter->observe(observer);
    }
  }

  int64_t syncSemaphoreCount = observer.highestSyncSemaphoreIndex + 1;
  if (pipeCapacityPlan) {
    syncSemaphoreCount =
        std::max(syncSemaphoreCount, pipeCapacityPlan->getSyncSemaphoreCount());
  }

  return PipeResourceRequirements{
      syncSemaphoreCount,
      observer.highestGlobalSemaphoreIndex + 1,
      info.sramScratch.bytes,
  };
}

/// Verify local semaphore ids before emitting ttkernel.get_semaphore. The
/// highest-id owner is tracked only to make over-limit diagnostics actionable.
LogicalResult
verifyPipeResourcePlanFitsHardware(ModuleOp mod, const PipeResourcePlan &info,
                                   const PipeCapacityPlan *pipeCapacityPlan,
                                   const PipeResourceRequirements &reqs) {
  enum class PipeSemaphoreKind {
    ReceiverCompletion,
    SenderReady,
    SenderCapacity,
  };

  struct HighestSemaphore {
    int64_t index = -1;
    PipeSemaphoreKind kind = PipeSemaphoreKind::ReceiverCompletion;
    std::optional<PipeKey> pipe;
  };

  struct SenderReadyObserver final : PipeReadyCounterObserver {
    HighestSemaphore &highest;
    const PipeKey &pipe;

    SenderReadyObserver(HighestSemaphore &highest, const PipeKey &pipe)
        : highest(highest), pipe(pipe) {}

    void observeLocalSemaphore(int64_t index) override {
      if (index > highest.index) {
        highest = HighestSemaphore{index, PipeSemaphoreKind::SenderReady, pipe};
      }
    }
  };

  HighestSemaphore highest;
  for (const auto &[pipeNetId, completion] : info.completionWaits) {
    (void)pipeNetId;
    if (completion.receiverCompletionSemIdx > highest.index) {
      highest =
          HighestSemaphore{completion.receiverCompletionSemIdx,
                           PipeSemaphoreKind::ReceiverCompletion, std::nullopt};
    }
  }
  for (const auto &[transferCreateOp, resource] : info.resources) {
    (void)transferCreateOp;
    if (resource.readyCounter) {
      SenderReadyObserver observer(highest, resource.pipe);
      resource.readyCounter->observe(observer);
    }
  }
  if (pipeCapacityPlan &&
      pipeCapacityPlan->getSyncSemaphoreCount() - 1 > highest.index) {
    highest = HighestSemaphore{pipeCapacityPlan->getSyncSemaphoreCount() - 1,
                               PipeSemaphoreKind::SenderCapacity, std::nullopt};
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
    note << "receiver-completion counter";
    break;
  case PipeSemaphoreKind::SenderReady:
    note << "sender-ready counter for ";
    assert(highest.pipe && "sender-ready resource must have a pipe");
    appendPipe(*highest.pipe);
    break;
  case PipeSemaphoreKind::SenderCapacity:
    note << "sender-capacity counter";
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
