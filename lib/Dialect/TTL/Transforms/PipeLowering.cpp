// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeLowering.h"

#include "PipePlanning.h"
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

static Value loadIndexTableEntry(Location loc, ArrayRef<int64_t> values,
                                 Value recordIndex,
                                 ConversionPatternRewriter &rewriter);

static Value buildSelectedPipeCounterAddress(
    Operation *op, Location loc, ArrayRef<PipeCounterInfo> counters,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  assert(!counters.empty() && "selected pipe counter table is empty");

  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "selected pipe operation must be inside a function");
  bool hasLocalCounter = llvm::any_of(counters, [](PipeCounterInfo counter) {
    return counter.getStorage() == PipeCounterStorage::LocalSemaphore;
  });
  bool hasGlobalCounter = llvm::any_of(counters, [](PipeCounterInfo counter) {
    return counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
  });
  if (!hasGlobalCounter) {
    SmallVector<int64_t> localIndices = llvm::map_to_vector(
        counters, [](PipeCounterInfo counter) { return counter.getIndex(); });
    Value semaphoreIndex =
        loadIndexTableEntry(loc, localIndices, recordIndex, rewriter);
    return ttk::GetSemaphoreOp::create(rewriter, loc, semaphoreIndex)
        .getResult();
  }
  auto getGlobalArgIndex = [&](PipeCounterInfo counter) {
    return getPipeRuntimeCommonArgIndex(
        func, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan) +
                  counter.getIndex());
  };
  if (!hasLocalCounter) {
    SmallVector<int64_t> globalArgIndices =
        llvm::map_to_vector(counters, getGlobalArgIndex);
    Value commonArgIndex =
        loadIndexTableEntry(loc, globalArgIndices, recordIndex, rewriter);
    return ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                          commonArgIndex)
        .getResult();
  }

  PipeCounterInfo validLocalCounter =
      *llvm::find_if(counters, [](PipeCounterInfo counter) {
        return counter.getStorage() == PipeCounterStorage::LocalSemaphore;
      });
  PipeCounterInfo validGlobalCounter =
      *llvm::find_if(counters, [](PipeCounterInfo counter) {
        return counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
      });
  SmallVector<int64_t> isGlobal;
  SmallVector<int64_t> localIndices;
  SmallVector<int64_t> globalArgIndices;
  for (PipeCounterInfo counter : counters) {
    bool usesGlobal =
        counter.getStorage() == PipeCounterStorage::GlobalSemaphore;
    isGlobal.push_back(usesGlobal ? 1 : 0);
    localIndices.push_back(
        (usesGlobal ? validLocalCounter : counter).getIndex());
    globalArgIndices.push_back(
        getGlobalArgIndex(usesGlobal ? counter : validGlobalCounter));
  }

  // arith.select cannot prevent either address operation from executing. Use
  // an existing index in the unused storage class so both addresses are valid.
  Value localIndex =
      loadIndexTableEntry(loc, localIndices, recordIndex, rewriter);
  Value localAddress =
      ttk::GetSemaphoreOp::create(rewriter, loc, localIndex).getResult();
  Value typedLocalAddress =
      ttk::CastToL1AddrOp::create(rewriter, loc, localAddress);
  Value globalArgIndex =
      loadIndexTableEntry(loc, globalArgIndices, recordIndex, rewriter);
  Value globalAddress =
      ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                     globalArgIndex)
          .getResult();
  Value typedGlobalAddress =
      ttk::CastToL1AddrOp::create(rewriter, loc, globalAddress);
  Value storageKind = loadIndexTableEntry(loc, isGlobal, recordIndex, rewriter);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value usesGlobal = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ne, storageKind, zero);
  return arith::SelectOp::create(rewriter, loc, usesGlobal, typedGlobalAddress,
                                 typedLocalAddress);
}

static Value buildSelectedReadyCounterAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  SmallVector<PipeCounterInfo> counters;
  counters.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    assert(resource.readyCounter &&
           "selected pipe missing sender-ready counter");
    counters.push_back(*resource.readyCounter);
  }
  return buildSelectedPipeCounterAddress(op, loc, counters, recordIndex,
                                         pipeResourcePlan, rewriter);
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

static Value addByteOffset(Location loc, Value baseAddress, Value byteOffset,
                           ConversionPatternRewriter &rewriter) {
  Value byteOffsetI32 = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), byteOffset);
  return arith::AddIOp::create(rewriter, loc, baseAddress, byteOffsetI32)
      .getResult();
}

static Value loadIndexTableEntry(Location loc, ArrayRef<int64_t> values,
                                 Value recordIndex,
                                 ConversionPatternRewriter &rewriter) {
  assert(!values.empty() && "selected pipe resource table must not be empty");
  return ttk::ConstantTableLookupOp::create(
      rewriter, loc, rewriter.getIndexType(), recordIndex,
      rewriter.getDenseI64ArrayAttr(values));
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

static Value buildSelectedAddressTableAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> byteOffsets;
  byteOffsets.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    assert(resource.addressStorage.mode ==
               PipeAddressMode::ReceiverPublishedAddressTable &&
           "selected pipe requires receiver-published addressing");
    assert(resource.addressStorage.sramAddressTable.has_value() &&
           "selected pipe missing address-table storage");
    byteOffsets.push_back(resource.addressStorage.sramAddressTable->byteOffset);
  }
  Value byteOffset =
      loadIndexTableEntry(loc, byteOffsets, recordIndex, rewriter);
  Value scratchBase = buildPipeRuntimeCommonArg(
      loc, rewriter, getPipeRuntimeCommonArgIndex(op, 0));
  return addByteOffset(loc, scratchBase, byteOffset, rewriter);
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

static void lowerPipeCapacityRelease(Location loc, FuncOp func,
                                     const PipeCapacityReleaseInfo &release,
                                     const PipeResourcePlan &pipeResourcePlan,
                                     Value nocVal,
                                     ConversionPatternRewriter &rewriter) {
  const PipeCapacityReleaseTarget &target = release.target;
  auto indexTy = rewriter.getIndexType();
  Value counterAddress = buildPipeCounterAddress(loc, func, release.counter,
                                                 pipeResourcePlan, rewriter);
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
                                sourceYTranslated, counterAddress, nocVal)
          .getResult();
  ttk::NocSemaphoreIncOp::create(rewriter, loc, remoteCapacityNocAddr,
                                 releaseCount, nocVal, /*posted=*/BoolAttr());
}

static Value
buildAddressTableDestinationAddress(Location loc, Value tableAddress,
                                    ConversionPatternRewriter &rewriter) {
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  auto tablePtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, tableAddress);
  auto zeroI32 = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
  return ttk::LoadFromL1Op::create(rewriter, loc, rewriter.getI32Type(),
                                   tablePtr, zeroI32)
      .getResult();
}

struct SelectedPipeFields {
  Value recordIndex;
  Value srcX;
  Value srcY;
  Value dstStartX;
  Value dstStartY;
  Value dstEndX;
  Value dstEndY;
  Value numDests;
  Value srcInDstRange;
  bool isCollective;
};

static SelectedPipeFields getSelectedPipeFields(const PipeReference &pipeRef) {
  assert(pipeRef.isSelected() && "expected selected pipe reference");
  if (pipeRef.isSelectedSrc()) {
    SelectPipeSrcOp op = pipeRef.getSelectedSrc();
    return SelectedPipeFields{
        op.getRecordIndex(),
        op.getSrcX(),
        op.getSrcY(),
        op.getDstStartX(),
        op.getDstStartY(),
        op.getDstEndX(),
        op.getDstEndY(),
        op.getNumDests(),
        op.getSrcInDstRange(),
        op.getRecords().getPipes().front().getIsCollective()};
  }
  SelectPipeDstOp op = pipeRef.getSelectedDst();
  return SelectedPipeFields{
      op.getRecordIndex(),
      op.getSrcX(),
      op.getSrcY(),
      op.getDstStartX(),
      op.getDstStartY(),
      op.getDstEndX(),
      op.getDstEndY(),
      op.getNumDests(),
      op.getSrcInDstRange(),
      op.getRecords().getPipes().front().getIsCollective()};
}

/// Compute the exact DFB address selected by ttl.copy(pipe, dst). Receivers
/// publish this address so senders do not have to infer receiver DFB state.
static Value
buildReceiverPublishedAddress(Value dst, Location loc,
                              const PipeReceiverAddressPublicationPlan &info,
                              ConversionPatternRewriter &rewriter) {
  auto receiverCBConverted =
      utils::convertTTLCBToTTKernel(info.receiverDFB, rewriter, loc);
  assert(succeeded(receiverCBConverted) &&
         "pipe post planning guarantees a convertible receiver DFB");

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
  auto pageSizeBytes =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                rewriter.getI32IntegerAttr(info.tileSizeBytes));
  auto byteOffset =
      arith::MulIOp::create(rewriter, loc, tileOffsetI32, pageSizeBytes);
  return arith::AddIOp::create(rewriter, loc, receiverWritePtr, byteOffset)
      .getResult();
}

namespace {

/// Emits transport-specific PipeNet synchronization and payload operations.
/// A transport returns failure when it cannot implement a selected operation.
class PipeTransportEmitter {
public:
  virtual ~PipeTransportEmitter() = default;

  virtual LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                                   Value publishedAddress) = 0;
  virtual void emitAddressPublishBarrier() = 0;
  virtual LogicalResult
  emitSenderReadyIncrement(Value senderReadyCounterAddr) = 0;
  virtual void preparePayloadWrite() = 0;
  virtual LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                         Value totalSizeBytes) = 0;
  virtual void emitPayloadWriteBarrier() = 0;
  virtual LogicalResult
  emitReceiverCompletionIncrement(Value receiverCompletionCounterAddr) = 0;
  virtual void emitCompletionSignalBarrier() = 0;
};

class NocPipeTransportEmitterBase : public PipeTransportEmitter {
protected:
  struct LogicalCore {
    Value x;
    Value y;
  };

  struct TranslatedCore {
    Value x;
    Value y;
  };

  struct DestinationRange {
    Value startX;
    Value startY;
    Value endX;
    Value endY;
  };

public:
  NocPipeTransportEmitterBase(Operation *op,
                              ConversionPatternRewriter &rewriter)
      : loc(op->getLoc()), rewriter(rewriter), nocIdx(getNocIndex(op)),
        nocVal(arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                         rewriter.getI8IntegerAttr(nocIdx))) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    LogicalCore sourceCore = getSourceLogicalCore();
    // The remote publish and the following ready signal reuse the translated
    // source coordinates, so they must be created before either branch.
    getSourceCore();
    Value currentX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    Value currentY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
    Value xMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentX, sourceCore.x);
    Value yMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentY, sourceCore.y);
    Value receiverIsSource =
        arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
    auto localPublish = scf::IfOp::create(rewriter, loc, receiverIsSource,
                                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&localPublish.getThenRegion().front());
      emitLocalReceiverAddressPublish(senderTableAddress, publishedAddress);
      rewriter.setInsertionPointToStart(&localPublish.getElseRegion().front());
      emitRemoteReceiverAddressPublish(senderTableAddress, publishedAddress);
    }
    rewriter.setInsertionPointAfter(localPublish);
    return success();
  }

  void emitLocalReceiverAddressPublish(Value senderTableAddress,
                                       Value publishedAddress) {
    auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
    Value tablePtr =
        ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderTableAddress);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    ttk::StoreToL1Op::create(rewriter, loc, publishedAddress, tablePtr, zero);
  }

  void emitRemoteReceiverAddressPublish(Value senderTableAddress,
                                        Value publishedAddress) {
    TranslatedCore sourceCore = getSourceCore();
    auto byteEnableAll = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
    // An inline NoC write does not update the sender's local SRAM when the
    // sender is also this receiver, so that case uses a direct L1 store.
    ttk::NocInlineDwWriteOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    senderTableAddress, publishedAddress,
                                    byteEnableAll, nocVal);
  }

  void emitAddressPublishBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  LogicalResult
  emitSenderReadyIncrement(Value senderReadyCounterAddr) override {
    TranslatedCore sourceCore = getSourceCore();
    auto senderReadyCounterNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                  senderReadyCounterAddr, nocVal);
    auto readyCounterIncrement =
        arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, senderReadyCounterNocAddr.getResult(),
        readyCounterIncrement, nocVal, /*posted=*/BoolAttr());
    return success();
  }

  void emitPayloadWriteBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  void emitCompletionSignalBarrier() override {
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

protected:
  virtual LogicalCore getSourceLogicalCore() = 0;
  virtual TranslatedCore getSourceCore() = 0;

  TranslatedCore buildTranslatedCore(Value logicalX, Value logicalY) {
    auto translatedX = ttk::ConvertLogicalXToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalX);
    auto translatedY = ttk::ConvertLogicalYToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalY);
    return {translatedX, translatedY};
  }

  TranslatedCore buildTranslatedCore(int64_t logicalX, int64_t logicalY) {
    auto logicalXValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalX);
    auto logicalYValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalY);
    auto translatedX = ttk::ConvertLogicalXToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalXValue);
    auto translatedY = ttk::ConvertLogicalYToTranslatedOp::create(
        rewriter, loc, rewriter.getIndexType(), logicalYValue);
    return {translatedX, translatedY};
  }

  Location loc;
  ConversionPatternRewriter &rewriter;
  int64_t nocIdx;
  Value nocVal;
};

class NocPipeTransportEmitter final : public NocPipeTransportEmitterBase {
public:
  NocPipeTransportEmitter(Operation *op, PipeType pipeType,
                          ConversionPatternRewriter &rewriter)
      : NocPipeTransportEmitterBase(op, rewriter), pipeType(pipeType) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    if (!pipeType.srcInDstRange()) {
      emitRemoteReceiverAddressPublish(senderTableAddress, publishedAddress);
      return success();
    }
    if (pipeType.hasSingleReceiver()) {
      emitLocalReceiverAddressPublish(senderTableAddress, publishedAddress);
      return success();
    }
    return NocPipeTransportEmitterBase::emitReceiverAddressPublish(
        senderTableAddress, publishedAddress);
  }

  void preparePayloadWrite() override {
    // Materialize destination coordinates before computing the payload address
    // so address selection does not change emitted operation order.
    if (pipeType.hasSingleReceiver()) {
      getDstStartCore();
      return;
    }
    getDestinationRange();
  }

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    if (pipeType.hasSingleReceiver()) {
      TranslatedCore dstStartCore = getDstStartCore();
      ttk::NocAsyncWriteOp::create(
          rewriter, loc, srcAddr, ValueRange{dstStartCore.x, dstStartCore.y},
          ValueRange{}, dstAddr, totalSizeBytes, nocVal);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    auto numDests = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(pipeType.getNumDests()));
    if (pipeType.srcInDstRange()) {
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
      return success();
    }
    ttk::NocAsyncWriteMulticastOp::create(
        rewriter, loc, srcAddr, totalSizeBytes, numDests,
        destinationRange.startX, destinationRange.startY, destinationRange.endX,
        destinationRange.endY, dstAddr, nocVal, /*linked=*/nullptr);
    return success();
  }

  void emitPayloadWriteBarrier() override {
    ttk::NocAsyncWriteBarrierOp::create(rewriter, loc, nocVal);
  }

  LogicalResult emitReceiverCompletionIncrement(
      Value receiverCompletionCounterAddr) override {
    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);

    if (pipeType.hasSingleReceiver()) {
      TranslatedCore dstStartCore = getDstStartCore();
      auto receiverCompletionNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, dstStartCore.x, dstStartCore.y,
          receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, receiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    int64_t numRemoteDests = pipeType.srcInDstRange()
                                 ? pipeType.getNumDests() - 1
                                 : pipeType.getNumDests();
    auto remoteReceiverCount =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(numRemoteDests));
    auto remoteReceiverCompletionMcastNocAddr =
        ttk::GetNocMulticastAddrOp::create(
            rewriter, loc, destinationRange.startX, destinationRange.startY,
            destinationRange.endX, destinationRange.endY,
            receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, remoteReceiverCompletionMcastNocAddr.getResult(),
        completionIncrement, remoteReceiverCount, nocVal,
        /*posted=*/BoolAttr());

    if (pipeType.srcInDstRange()) {
      TranslatedCore sourceCore = getSourceCore();
      auto localReceiverCompletionNocAddr =
          ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
    return success();
  }

private:
  LogicalCore getSourceLogicalCore() override {
    return {arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX()),
            arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY())};
  }

  TranslatedCore getSourceCore() override {
    if (!sourceCore) {
      sourceCore = buildTranslatedCore(pipeType.getSrcX(), pipeType.getSrcY());
    }
    return *sourceCore;
  }

  TranslatedCore getDstStartCore() {
    if (!dstStartCore) {
      dstStartCore =
          buildTranslatedCore(pipeType.getDstStartX(), pipeType.getDstStartY());
    }
    return *dstStartCore;
  }

  DestinationRange getDestinationRange() {
    if (destinationRange) {
      return *destinationRange;
    }
    TranslatedCore dstStartTranslatedCore = getDstStartCore();
    // Preserve the memoized start coordinate for unicast and completion uses.
    auto [dstStartX, dstStartY] = dstStartTranslatedCore;
    auto [dstEndX, dstEndY] =
        buildTranslatedCore(pipeType.getDstEndX(), pipeType.getDstEndY());
    // NoC 1 traverses the grid in reverse coordinate order, while multicast
    // operations require their endpoints in traversal order.
    if (nocIdx == 1) {
      std::swap(dstStartX, dstEndX);
      std::swap(dstStartY, dstEndY);
    }
    destinationRange = DestinationRange{dstStartX, dstStartY, dstEndX, dstEndY};
    return *destinationRange;
  }

  PipeType pipeType;
  std::optional<TranslatedCore> sourceCore;
  std::optional<TranslatedCore> dstStartCore;
  std::optional<DestinationRange> destinationRange;
};

/// Emit one protocol body for every record in a PipeNet table. Record fields
/// select the required unicast, multicast, and loopback hardware operations;
/// the conditions are transport semantics, not special cases for record
/// indices.
class SelectedNocPipeTransportEmitter final
    : public NocPipeTransportEmitterBase {
public:
  SelectedNocPipeTransportEmitter(Operation *op, SelectedPipeFields fields,
                                  ConversionPatternRewriter &rewriter)
      : NocPipeTransportEmitterBase(op, rewriter), fields(fields) {}

  void preparePayloadWrite() override {
    // Coordinate translations must dominate the conditional regions emitted
    // below, so cache the applicable coordinates before creating those regions.
    if (fields.isCollective) {
      getDestinationRange();
      return;
    }
    getDstStartCore();
  }

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    if (!fields.isCollective) {
      emitUnicastPayloadWrite(srcAddr, dstAddr, totalSizeBytes);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    Value numDests = getNumDests();
    // A one-receiver collective uses unicast hardware operations. This also
    // avoids a zero-recipient multicast completion for local loopback.
    auto singleReceiverIf = scf::IfOp::create(
        rewriter, loc, getHasSingleReceiver(), /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getThenRegion().front());
      emitUnicastPayloadWrite(srcAddr, dstAddr, totalSizeBytes);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getElseRegion().front());
      emitMulticastPayloadWrite(srcAddr, dstAddr, totalSizeBytes, numDests,
                                destinationRange);
    }
    rewriter.setInsertionPointAfter(singleReceiverIf);
    return success();
  }

  LogicalResult emitReceiverCompletionIncrement(
      Value receiverCompletionCounterAddr) override {
    auto completionIncrement = arith::ConstantIndexOp::create(rewriter, loc, 1);

    if (!fields.isCollective) {
      emitUnicastCompletionIncrement(receiverCompletionCounterAddr,
                                     completionIncrement);
      return success();
    }

    DestinationRange destinationRange = getDestinationRange();
    Value numDests = getNumDests();
    auto singleReceiverIf = scf::IfOp::create(
        rewriter, loc, getHasSingleReceiver(), /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getThenRegion().front());
      emitUnicastCompletionIncrement(receiverCompletionCounterAddr,
                                     completionIncrement);
      rewriter.setInsertionPointToStart(
          &singleReceiverIf.getElseRegion().front());
      emitMulticastCompletionIncrement(receiverCompletionCounterAddr,
                                       completionIncrement, numDests,
                                       destinationRange);
    }
    rewriter.setInsertionPointAfter(singleReceiverIf);
    return success();
  }

private:
  void emitUnicastPayloadWrite(Value srcAddr, Value dstAddr,
                               Value totalSizeBytes) {
    TranslatedCore dstStartCore = getDstStartCore();
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr,
                                 ValueRange{dstStartCore.x, dstStartCore.y},
                                 ValueRange{}, dstAddr, totalSizeBytes, nocVal);
  }

  void emitMulticastPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes, Value numDests,
                                 DestinationRange destinationRange) {
    // Standard multicast does not write the sender's local memory, so a
    // receiver range containing the sender requires the loopback operation.
    auto loopbackIf = scf::IfOp::create(rewriter, loc, fields.srcInDstRange,
                                        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&loopbackIf.getThenRegion().front());
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
      rewriter.setInsertionPointToStart(&loopbackIf.getElseRegion().front());
      ttk::NocAsyncWriteMulticastOp::create(
          rewriter, loc, srcAddr, totalSizeBytes, numDests,
          destinationRange.startX, destinationRange.startY,
          destinationRange.endX, destinationRange.endY, dstAddr, nocVal,
          /*linked=*/nullptr);
    }
    rewriter.setInsertionPointAfter(loopbackIf);
  }

  void emitUnicastCompletionIncrement(Value receiverCompletionCounterAddr,
                                      Value completionIncrement) {
    TranslatedCore dstStartCore = getDstStartCore();
    auto receiverCompletionNocAddr =
        ttk::GetNocAddrOp::create(rewriter, loc, dstStartCore.x, dstStartCore.y,
                                  receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncOp::create(
        rewriter, loc, receiverCompletionNocAddr.getResult(),
        completionIncrement, nocVal, /*posted=*/BoolAttr());
  }

  void emitMulticastCompletionIncrement(Value receiverCompletionCounterAddr,
                                        Value completionIncrement,
                                        Value numDests,
                                        DestinationRange destinationRange) {
    // The multicast atomic updates only remote receivers. If the sender is also
    // a receiver, exclude it from that count and update it with a local atomic.
    auto one = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         rewriter.getI32IntegerAttr(1));
    Value numRemoteWithLoopback =
        arith::SubIOp::create(rewriter, loc, numDests, one);
    Value numRemoteDests = arith::SelectOp::create(
        rewriter, loc, fields.srcInDstRange, numRemoteWithLoopback, numDests);
    auto remoteReceiverCompletionMcastNocAddr =
        ttk::GetNocMulticastAddrOp::create(
            rewriter, loc, destinationRange.startX, destinationRange.startY,
            destinationRange.endX, destinationRange.endY,
            receiverCompletionCounterAddr, nocVal);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, remoteReceiverCompletionMcastNocAddr.getResult(),
        completionIncrement, numRemoteDests, nocVal, /*posted=*/BoolAttr());

    auto localIncrementIf = scf::IfOp::create(
        rewriter, loc, fields.srcInDstRange, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          &localIncrementIf.getThenRegion().front());
      TranslatedCore sourceCore = getSourceCore();
      auto localReceiverCompletionNocAddr =
          ttk::GetNocAddrOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    receiverCompletionCounterAddr, nocVal);
      ttk::NocSemaphoreIncOp::create(
          rewriter, loc, localReceiverCompletionNocAddr.getResult(),
          completionIncrement, nocVal, /*posted=*/BoolAttr());
    }
    rewriter.setInsertionPointAfter(localIncrementIf);
  }

  Value getHasSingleReceiver() {
    if (!hasSingleReceiver) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      hasSingleReceiver = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, fields.numDests, one);
    }
    return hasSingleReceiver;
  }

  LogicalCore getSourceLogicalCore() override {
    return {fields.srcX, fields.srcY};
  }

  TranslatedCore getSourceCore() override {
    if (!sourceCore) {
      sourceCore = buildTranslatedCore(fields.srcX, fields.srcY);
    }
    return *sourceCore;
  }

  TranslatedCore getDstStartCore() {
    if (!dstStartCore) {
      dstStartCore = buildTranslatedCore(fields.dstStartX, fields.dstStartY);
    }
    return *dstStartCore;
  }

  DestinationRange getDestinationRange() {
    if (destinationRange) {
      return *destinationRange;
    }
    TranslatedCore dstStartTranslatedCore = getDstStartCore();
    auto [dstStartX, dstStartY] = dstStartTranslatedCore;
    auto [dstEndX, dstEndY] =
        buildTranslatedCore(fields.dstEndX, fields.dstEndY);
    // NoC 1 traverses the grid in reverse coordinate order, while multicast
    // operations require their endpoints in traversal order.
    if (nocIdx == 1) {
      std::swap(dstStartX, dstEndX);
      std::swap(dstStartY, dstEndY);
    }
    destinationRange = DestinationRange{dstStartX, dstStartY, dstEndX, dstEndY};
    return *destinationRange;
  }

  Value getNumDests() {
    if (!numDests) {
      numDests = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI32Type(), fields.numDests);
    }
    return numDests;
  }

  SelectedPipeFields fields;
  Value hasSingleReceiver;
  Value numDests;
  std::optional<TranslatedCore> sourceCore;
  std::optional<TranslatedCore> dstStartCore;
  std::optional<DestinationRange> destinationRange;
};

} // namespace

//===----------------------------------------------------------------------===//
// Receiver post sequence counter initialization
//===----------------------------------------------------------------------===//

void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &postSequenceCounters,
    PipeSelectedPostSequenceMap &selectedPostSequenceCounters) {
  llvm::MapVector<FuncOp, SmallVector<PipeCounterInfo>> staticCountersByFunc;
  for (const auto &[protocolOp, resource] : pipeResourcePlan.resources) {
    auto postOp = dyn_cast<PipeTransferPostOp>(protocolOp);
    if (!postOp) {
      continue;
    }
    FuncOp func = postOp->getParentOfType<FuncOp>();
    assert(func && "pipe transfer post must be inside a function");
    SmallVector<PipeCounterInfo> &counters = staticCountersByFunc[func];
    if (!llvm::is_contained(counters, resource.completion.counter)) {
      counters.push_back(resource.completion.counter);
    }
  }

  llvm::MapVector<FuncOp, SmallVector<PipeCounterInfo>> selectedCountersByFunc;
  for (const auto &[protocolOp, resources] :
       pipeResourcePlan.selectedResources) {
    auto postOp = dyn_cast<PipeTransferPostOp>(protocolOp);
    if (!postOp) {
      continue;
    }
    FuncOp func = postOp->getParentOfType<FuncOp>();
    assert(func && "pipe transfer post must be inside a function");
    SmallVector<PipeCounterInfo> &counters = selectedCountersByFunc[func];
    for (const PipeResourceInfo &resource : resources) {
      if (!llvm::is_contained(counters, resource.completion.counter)) {
        counters.push_back(resource.completion.counter);
      }
    }
  }

  auto sortCounters = [](SmallVectorImpl<PipeCounterInfo> &counters) {
    llvm::sort(counters, [](PipeCounterInfo lhs, PipeCounterInfo rhs) {
      return std::make_pair(lhs.getStorage(), lhs.getIndex()) <
             std::make_pair(rhs.getStorage(), rhs.getIndex());
    });
  };

  for (auto &[func, counters] : staticCountersByFunc) {
    // These entry-block values dominate posts nested in receiver control flow.
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto memrefTy = MemRefType::get({1}, builder.getI32Type());
    Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto &countersForFunc = postSequenceCounters[func];
    sortCounters(counters);
    for (PipeCounterInfo counterInfo : counters) {
      Value sequenceCounter = memref::AllocaOp::create(builder, loc, memrefTy);
      memref::StoreOp::create(builder, loc, zero, sequenceCounter,
                              ValueRange{zeroIndex});
      countersForFunc.push_back(
          PipeCounterProgress{counterInfo, sequenceCounter});
    }
  }

  for (auto &[func, counters] : selectedCountersByFunc) {
    // One function-local table lets every selected post preserve progress for
    // shared completion counters without expanding one branch per record.
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    sortCounters(counters);
    auto memrefType = MemRefType::get({static_cast<int64_t>(counters.size())},
                                      builder.getI32Type());
    Value completionSequences =
        memref::AllocaOp::create(builder, loc, memrefType);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    for (std::size_t counterIndex = 0; counterIndex < counters.size();
         ++counterIndex) {
      Value index = arith::ConstantIndexOp::create(builder, loc, counterIndex);
      memref::StoreOp::create(builder, loc, zero, completionSequences,
                              ValueRange{index});
    }
    selectedPostSequenceCounters[func] = PipeSelectedPostSequenceCounters{
        completionSequences, std::move(counters)};
  }
}

void initializePipeCapacityCounters(
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeResourcePlan &pipeResourcePlan,
    PipeCounterProgressMap &senderCapacityCounters) {
  for (const auto &entry : pipeCapacityPlan.getInitializations()) {
    FuncOp func = entry.first;
    const SmallVector<PipeCapacityInitInfo> &initializations = entry.second;
    SmallVector<PipeCapacityInitInfo> sortedInitializations(initializations);
    llvm::sort(sortedInitializations, [](const PipeCapacityInitInfo &lhs,
                                         const PipeCapacityInitInfo &rhs) {
      return std::make_pair(lhs.counter.getStorage(), lhs.counter.getIndex()) <
             std::make_pair(rhs.counter.getStorage(), rhs.counter.getIndex());
    });

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto counterMemrefTy = MemRefType::get({1}, builder.getI32Type());
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto &perFuncCounters = senderCapacityCounters[func];
    for (const PipeCapacityInitInfo &init : sortedInitializations) {
      Value capacityCounterPtr = buildPipeCounterPtr(loc, func, init.counter,
                                                     pipeResourcePlan, builder);
      Value initialCapacity =
          arith::ConstantIntOp::create(builder, loc, init.initialCapacity, 32);
      ttk::NocSemaphoreSetOp::create(builder, loc, capacityCounterPtr,
                                     initialCapacity);
      // The sender tracks its cumulative acquired count in a kernel-local
      // counter and waits for the shared capacity counter to reach it, so the
      // receiver's remote increment stays the only writer of the shared word.
      auto counter = memref::AllocaOp::create(builder, loc, counterMemrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, counter,
                              ValueRange{zeroIdx});
      perFuncCounters.push_back(
          PipeCounterProgress{init.counter, counter.getResult()});
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

static FailureOr<PipeCounterProgress>
lookupPipeCounterProgress(const PipeCounterProgressMap &progress, FuncOp func,
                          PipeCounterInfo counter) {
  auto funcIt = progress.find(func);
  if (funcIt == progress.end()) {
    return failure();
  }
  auto progressIt =
      llvm::find_if(funcIt->second, [&](const PipeCounterProgress &entry) {
        return entry.counter == counter;
      });
  if (progressIt == funcIt->second.end()) {
    return failure();
  }
  return *progressIt;
}

/// Assign a completion sequence when posting the receive. Tokens may be stored
/// or reordered, so each token must retain the sequence of its own post.
static Value incrementPipePostSequence(Location loc, Value sequenceCounter,
                                       Value sequenceIndex,
                                       ConversionPatternRewriter &rewriter) {
  Value previousSequence = memref::LoadOp::create(
      rewriter, loc, sequenceCounter, ValueRange{sequenceIndex});
  Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  Value tokenSequence =
      arith::AddIOp::create(rewriter, loc, previousSequence, one);
  memref::StoreOp::create(rewriter, loc, tokenSequence, sequenceCounter,
                          ValueRange{sequenceIndex});
  return tokenSequence;
}

static LogicalResult
lowerSelectedPipeTransferSend(PipeTransferSendOp op, Value srcCB,
                              const PipeTransferPlan &transferPlan,
                              const PipeResourcePlan &pipeResourcePlan,
                              ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  const PipeReference &pipeRef = transferPlan.getPipeReference();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources = transferPlan.getSelectedResources();
  const PipeSendPlan &sendPlan = transferPlan.getSend();

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  SelectedNocPipeTransportEmitter nocTransport(op, fields, rewriter);
  PipeTransportEmitter &transport = nocTransport;

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, pipeResourcePlan, rewriter);
  auto senderSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
  Value expectedSignals;
  if (fields.isCollective) {
    expectedSignals = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI32Type(), fields.numDests);
  } else {
    expectedSignals = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  }
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedSignals);
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked source DFB type");
  Value srcPtrIdx;
  if (sendPlan.usesReadPointer) {
    auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), cbReadPtr);
  } else {
    auto srcWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), srcWritePtr);
  }

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI32Type(), srcPtrIdx);
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(sendPlan.payloadSizeBytes));
  transport.preparePayloadWrite();

  Value tableAddress = buildSelectedAddressTableAddress(
      op, loc, resources, fields.recordIndex, rewriter);
  Value dstAddr =
      buildAddressTableDestinationAddress(loc, tableAddress, rewriter);
  if (failed(transport.emitPayloadWrite(srcAddr, dstAddr, totalSizeVal))) {
    return failure();
  }
  transport.emitPayloadWriteBarrier();

  SmallVector<PipeCounterInfo> completionCounters =
      llvm::map_to_vector(resources, [](const PipeResourceInfo &resource) {
        return resource.completion.counter;
      });
  Value completionCounterAddress = buildSelectedPipeCounterAddress(
      op, loc, completionCounters, fields.recordIndex, pipeResourcePlan,
      rewriter);
  if (failed(transport.emitReceiverCompletionIncrement(
          completionCounterAddress))) {
    return failure();
  }
  transport.emitCompletionSignalBarrier();

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Lower CB -> Pipe copy: write source DFB data to the selected destination
/// address, then signal arrival.
void lowerInactivePipeTransferSend(PipeTransferSendOp op,
                                   ConversionPatternRewriter &rewriter) {
  rewriter.replaceOp(op, makeZeroI32(op.getLoc(), rewriter));
}

LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, const PipeTransferPlan &transferPlan,
    const PipeResourcePlan &pipeResourcePlan,
    const PipeCapacityPlan &pipeCapacityPlan,
    const PipeCounterProgressMap &senderCapacityCounters,
    const PipeComputedAddressCounterMap &computedAddressCounters,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive sender must not use an active transfer plan");
  assert(transferPlan.isSend() && "sender operation has a non-send plan");
  const PipeReference &pipeRef = transferPlan.getPipeReference();
  if (pipeRef.isSelected()) {
    return lowerSelectedPipeTransferSend(op, srcCB, transferPlan,
                                         pipeResourcePlan, rewriter);
  }
  PipeType pipeType = pipeRef.getStaticPipeType();
  const PipeResourceInfo &pipeResource = transferPlan.getResources();
  const PipeSendPlan &sendPlan = transferPlan.getSend();
  PipeCompletionInfo completionInfo = pipeResource.completion;
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  NocPipeTransportEmitter nocTransport(op, pipeType, rewriter);
  PipeTransportEmitter &transport = nocTransport;

  bool usesCapacityProtocol = transferPlan.getSynchronizationProtocol() ==
                              PipeSynchronizationProtocol::Capacity;
  ArrayRef<PipeCapacityAcquireInfo> capacityAcquires =
      pipeCapacityPlan.lookupAcquires(op);
  assert(usesCapacityProtocol == !capacityAcquires.empty() &&
         "capacity-protocol send must have at least one capacity acquire");
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  assert(senderFunc && "pipe transfer send must be inside a function");
  SmallVector<Value> capacityCounters;
  if (!capacityAcquires.empty()) {
    for (const PipeCapacityAcquireInfo &capacityAcquire : capacityAcquires) {
      FailureOr<PipeCounterProgress> maybeCounter = lookupPipeCounterProgress(
          senderCapacityCounters, senderFunc, capacityAcquire.counter);
      if (failed(maybeCounter)) {
        op.emitError("pipe capacity acquire without sender counter; "
                     "initializePipeCapacityCounters must run before "
                     "convert-ttl-to-ttkernel");
        return failure();
      }
      capacityCounters.push_back(maybeCounter->value);
    }
  }
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) &&
         "pipe send planning guarantees a convertible source DFB");

  if (usesCapacityProtocol) {
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [capacityAcquire, senderCapacityCounter] :
         llvm::zip_equal(capacityAcquires, capacityCounters)) {
      Value capacityCounterPtr = buildPipeCounterPtr(
          loc, senderFunc, capacityAcquire.counter, pipeResourcePlan, rewriter);
      // Advance the sender's cumulative acquired count and block until the
      // shared capacity counter reaches it. The receiver's remote increment is
      // the only writer, so the acquire never writes the shared counter.
      Value previousAcquired = memref::LoadOp::create(
          rewriter, loc, senderCapacityCounter, ValueRange{zeroIdx});
      Value capacityCount = arith::ConstantIntOp::create(
          rewriter, loc, capacityAcquire.count, 32);
      Value nextAcquired =
          arith::AddIOp::create(rewriter, loc, previousAcquired, capacityCount);
      memref::StoreOp::create(rewriter, loc, nextAcquired,
                              senderCapacityCounter, ValueRange{zeroIdx});
      ttk::SemaphoreWaitMinOp::create(rewriter, loc, capacityCounterPtr,
                                      nextAcquired);
    }
  } else {
    assert(pipeResource.readyCounter &&
           "sender-ready protocol selected without a sender-ready counter");
    int64_t expectedReceiverPosts =
        isCollectiveTransfer(pipeResource.transferContract) ? numDests : 1;
    Value senderReadyCounterAddr =
        buildPipeCounterAddress(loc, senderFunc, *pipeResource.readyCounter,
                                pipeResourcePlan, rewriter);
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

  // Producer source address is at the source DFB's write_ptr (data is staged
  // there before push_back); consumer source address is at its read_ptr.
  Value srcPtrIdx;
  if (sendPlan.usesReadPointer) {
    auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbReadPtr);
  } else {
    auto srcWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, srcWritePtr);
  }
  transport.preparePayloadWrite();

  // Transfer the entire block in one NoC write. Tiles are contiguous in the
  // DFB, and destination DFB layout is uniform across nodes, so lowering sends
  // all tiles at once instead of one per tile.
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty,
      rewriter.getI32IntegerAttr(sendPlan.payloadSizeBytes));

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

  if (failed(transport.emitPayloadWrite(srcAddr, dstAddr, totalSizeVal))) {
    return failure();
  }

  // Wait for payload writes to complete before signaling receiver completion.
  // Without this barrier, the receiver may wake up before all data arrives.
  Value receiverCompletionCounterAddr = buildPipeCounterAddress(
      loc, senderFunc, completionInfo.counter, pipeResourcePlan, rewriter);
  transport.emitPayloadWriteBarrier();

  if (failed(transport.emitReceiverCompletionIncrement(
          receiverCompletionCounterAddr))) {
    return failure();
  }

  // Completion signals use non-posted atomics. The send ttl.wait lowers to a
  // no-op, so this barrier is the only flush before the kernel exits.
  transport.emitCompletionSignalBarrier();

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

void lowerInactivePipeTransferPost(PipeTransferPostOp op,
                                   ConversionPatternRewriter &rewriter) {
  auto token = UnrealizedConversionCastOp::create(
      rewriter, op.getLoc(), op.getToken().getType(), ValueRange{});
  rewriter.replaceOp(op, token.getResult(0));
}

static LogicalResult lowerSelectedPipeTransferPost(
    PipeTransferPostOp op, Value dst, const PipeTransferPlan &transferPlan,
    const PipeSelectedPostSequenceMap &selectedPostSequenceCounters,
    const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  const PipeReference &pipeRef = transferPlan.getPipeReference();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources = transferPlan.getSelectedResources();
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer post must be inside a function");
  auto sequenceIt = selectedPostSequenceCounters.find(func);
  if (sequenceIt == selectedPostSequenceCounters.end()) {
    op.emitError(
        "table-driven pipe receive has no completion sequence counters");
    return failure();
  }
  SmallVector<int64_t> sequenceIndices;
  for (const PipeResourceInfo &resource : resources) {
    auto counterIt =
        llvm::find(sequenceIt->second.counters, resource.completion.counter);
    if (counterIt == sequenceIt->second.counters.end()) {
      op.emitError(
          "table-driven pipe receive has no completion sequence counter");
      return failure();
    }
    sequenceIndices.push_back(
        std::distance(sequenceIt->second.counters.begin(), counterIt));
  }
  const PipePostPlan &postPlan = transferPlan.getPost();
  assert(postPlan.addressPublication &&
         "selected receiver post requires published addressing");

  SelectedNocPipeTransportEmitter nocTransport(op, fields, rewriter);
  PipeTransportEmitter &transport = nocTransport;

  Value publishedAddress = buildReceiverPublishedAddress(
      dst, loc, *postPlan.addressPublication, rewriter);
  Value tableAddress = buildSelectedAddressTableAddress(
      op, loc, resources, fields.recordIndex, rewriter);
  if (failed(transport.emitReceiverAddressPublish(tableAddress,
                                                  publishedAddress))) {
    return failure();
  }
  transport.emitAddressPublishBarrier();

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, pipeResourcePlan, rewriter);
  if (failed(transport.emitSenderReadyIncrement(senderSemAddr))) {
    return failure();
  }

  Value sequenceIndex =
      loadIndexTableEntry(loc, sequenceIndices, fields.recordIndex, rewriter);
  Value tokenSequence = incrementPipePostSequence(
      loc, sequenceIt->second.completionSequences, sequenceIndex, rewriter);
  rewriter.replaceOp(op, tokenSequence);
  return success();
}

LogicalResult lowerPipeTransferPost(
    PipeTransferPostOp op, Value dst, const PipeTransferPlan &transferPlan,
    const PipeCounterProgressMap &counters,
    const PipeSelectedPostSequenceMap &selectedPostSequenceCounters,
    const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive receiver post must not use an active transfer plan");
  assert(transferPlan.isPost() && "receiver post has another operation plan");
  const PipeReference &pipeRef = transferPlan.getPipeReference();
  if (pipeRef.isSelected()) {
    return lowerSelectedPipeTransferPost(op, dst, transferPlan,
                                         selectedPostSequenceCounters,
                                         pipeResourcePlan, rewriter);
  }
  PipeType pipeType = pipeRef.getStaticPipeType();
  const PipeResourceInfo &pipeResource = transferPlan.getResources();
  const PipePostPlan &postPlan = transferPlan.getPost();
  auto func = op->getParentOfType<func::FuncOp>();
  assert(func && "pipe transfer post must be inside a function");
  FailureOr<PipeCounterProgress> maybeSequenceCounter =
      lookupPipeCounterProgress(counters, func,
                                pipeResource.completion.counter);
  if (failed(maybeSequenceCounter)) {
    op.emitError("pipe receive post has no sequence counter for its completion "
                 "counter");
    return failure();
  }
  Value sequenceCounter = maybeSequenceCounter->value;

  NocPipeTransportEmitter nocTransport(op, pipeType, rewriter);
  PipeTransportEmitter &transport = nocTransport;

  if (postPlan.addressPublication) {
    AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
    Value publishedAddress = buildReceiverPublishedAddress(
        dst, loc, *postPlan.addressPublication, rewriter);
    Value tableAddress =
        buildAddressTableAddress(loc, addressTableInfo, rewriter);
    if (failed(transport.emitReceiverAddressPublish(tableAddress,
                                                    publishedAddress))) {
      return failure();
    }
    transport.emitAddressPublishBarrier();
  }

  if (transferPlan.getSynchronizationProtocol() ==
      PipeSynchronizationProtocol::ReceiverPost) {
    assert(pipeResource.readyCounter &&
           "sender-ready protocol selected without a sender-ready counter");
    Value senderReadyCounterAddr = buildPipeCounterAddress(
        loc, func, *pipeResource.readyCounter, pipeResourcePlan, rewriter);
    if (failed(transport.emitSenderReadyIncrement(senderReadyCounterAddr))) {
      return failure();
    }
  }

  Value sequenceIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value tokenSequence =
      incrementPipePostSequence(loc, sequenceCounter, sequenceIndex, rewriter);
  rewriter.replaceOp(op, tokenSequence);
  return success();
}

static Value computeDFBPopNumTiles(CBPopOp op, CircularBufferType dfbType,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc) {
  if (auto attr = op.getNumTilesAttr()) {
    return arith::ConstantIntOp::create(rewriter, loc, attr.getInt(), 32);
  }
  return arith::ConstantIntOp::create(rewriter, loc,
                                      dfbType.getElementsPerBlock(), 32);
}

LogicalResult lowerCBPop(CBPopOp op, Value cb,
                         const PipeCapacityPlan &pipeCapacityPlan,
                         const PipeResourcePlan &pipeResourcePlan,
                         ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value originalCb = op.getCb();
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(originalCb);
  if (failed(maybeDFBType)) {
    return rewriter.notifyMatchFailure(op, "failed to get TTL DFB type");
  }

  auto convertedCb = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
  if (failed(convertedCb)) {
    return rewriter.notifyMatchFailure(op, "failed to convert DFB operand");
  }

  Value numTiles = computeDFBPopNumTiles(op, *maybeDFBType, rewriter, loc);
  ttk::CBPopFrontOp::create(rewriter, loc, *convertedCb, numTiles);

  // A receiver DFB pop makes one slot available to its sender. Emit the
  // release immediately after the pop so both execute under the same control
  // flow.
  ArrayRef<PipeCapacityReleaseInfo> releases =
      pipeCapacityPlan.lookupReleases(op);
  if (!releases.empty()) {
    FuncOp func = op->getParentOfType<FuncOp>();
    assert(func && "DFB pop must be inside a function");
    int64_t nocIdx = getNocIndex(op);
    Value nocVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));
    for (const PipeCapacityReleaseInfo &release : releases) {
      lowerPipeCapacityRelease(loc, func, release, pipeResourcePlan, nocVal,
                               rewriter);
    }
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

  rewriter.eraseOp(op);
  return success();
}

/// Lower the receiver completion wait using the posted token's sequence.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
                                    const PipeTransferPlan &transferPlan,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  assert(!pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation()) &&
         "inactive receiver wait must not use an active transfer plan");
  assert(transferPlan.isWait() && "receiver wait has another operation plan");
  auto loc = op.getLoc();
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe transfer wait must be inside a function");
  Value receiverCompletionCounterAddress;
  if (transferPlan.isSelected()) {
    SelectedPipeFields fields =
        getSelectedPipeFields(transferPlan.getPipeReference());
    SmallVector<PipeCounterInfo> completionCounters =
        llvm::map_to_vector(transferPlan.getSelectedResources(),
                            [](const PipeResourceInfo &resource) {
                              return resource.completion.counter;
                            });
    receiverCompletionCounterAddress = buildSelectedPipeCounterAddress(
        op, loc, completionCounters, fields.recordIndex, pipeResourcePlan,
        rewriter);
  } else {
    PipeCompletionInfo completionInfo = transferPlan.getResources().completion;
    receiverCompletionCounterAddress = buildPipeCounterAddress(
        loc, func, completionInfo.counter, pipeResourcePlan, rewriter);
  }

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  Value receiverCompletionCounterPtr = ttk::CastToL1PtrOp::create(
      rewriter, loc, l1PtrTy, receiverCompletionCounterAddress);
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
    FailureOr<PipeReference> pipeRef = getPipeReference(op, op.getPipe());
    assert(succeeded(pipeRef) && "pipe transfer create verifier failed");
    PipeTransferContract contract = getPipeTransferContract(op);
    for (PipeType pipeType :
         getPipeTypesFromReference(op.getContext(), *pipeRef)) {
      int64_t netId = pipeType.getPipeNetId();
      PipeKey key{pipeType.getSrcX(),      pipeType.getSrcY(),
                  pipeType.getDstStartX(), pipeType.getDstStartY(),
                  pipeType.getDstEndX(),   pipeType.getDstEndY()};
      if (seenPerNet[netId].insert(key)) {
        index[netId].push_back(PipeInfo{pipeType, contract});
        continue;
      }
      if (!isCollectiveTransfer(contract)) {
        continue;
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
    }
  });
}

namespace {

/// Allocation unit for all resources owned by one transfer definition.
///
/// One send and its corresponding receiver posts share an address mechanism
/// and sender-ready counter. Each receiver wait uses the completion counter
/// assigned to the same unit.
struct PipeTransferAllocationUnit {
  PipeTransferNodeId transferNodeId = 0;
  Operation *sendOp = nullptr;
  /// Send, receiver-post, and receiver-wait operations for this transfer.
  SmallVector<Operation *> protocolOps;
  /// Record indices distinguish graph nodes that share one protocol operation.
  SmallVector<std::pair<Operation *, unsigned>> selectedProtocolRecords;

  /// Logical pipe whose source owns this unit's address and ready resources.
  PipeKey pipe;

  PipeType pipeType;

  PipeTransferContract transferContract = PipeTransferContract::PointToPoint;

  /// Stable tie-breaker for deterministic allocation.
  int64_t ordinal = 0;

  /// Conservative post-to-send lifetime for sender-owned resources.
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

static bool isSelectedTransferUnit(const PipeTransferAllocationUnit &unit) {
  return !unit.selectedProtocolRecords.empty();
}

} // namespace

static bool pipeTransferIntervalsOverlap(const PipeTransferAllocationUnit &lhs,
                                         const PipeTransferAllocationUnit &rhs,
                                         const DominanceInfo &dominanceInfo) {
  return intervalsOverlap(lhs.interval, rhs.interval, dominanceInfo);
}

static bool pipeResourceUnitsInterfere(const PipeTransferAllocationUnit &lhs,
                                       const PipeTransferAllocationUnit &rhs,
                                       const DominanceInfo &dominanceInfo) {
  if (isSelectedTransferUnit(lhs) || isSelectedTransferUnit(rhs)) {
    // One protocol operation executes for several records, so its
    // operation-level live interval cannot prove that two records are disjoint.
    return true;
  }
  return pipeTransferIntervalsOverlap(lhs, rhs, dominanceInfo);
}

static FailureOr<SmallVector<PipeTransferAllocationUnit, 0>>
collectPipeTransferAllocationUnits(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, const DominanceInfo &dominanceInfo,
    const PostDominanceInfo &postDominanceInfo,
    llvm::SmallPtrSetImpl<Operation *> &staticallyInactiveOps) {
  SmallVector<PipeTransferAllocationUnit, 0> units;
  llvm::DenseMap<Operation *, int64_t> operationOrdinals;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> waitOpsByPost;
  int64_t nextOperationOrdinal = 0;
  WalkResult provenanceWalkResult = mod.walk([&](Operation *op) {
    if (isa<PipeTransferPostOp, PipeTransferSendOp>(op)) {
      operationOrdinals[op] = nextOperationOrdinal++;
      if (!pipeGraph.hasPipeTransferNodeForProtocolOp(op)) {
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
      if (!pipeGraph.hasPipeTransferNodeForProtocolOp(postOp)) {
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

  llvm::DenseMap<Operation *, llvm::DenseMap<PipeKey, unsigned>>
      selectedOccurrencesByProtocolOp;
  auto recordSelectedProtocolRow = [&](PipeTransferAllocationUnit &unit,
                                       Operation *protocolOp) -> LogicalResult {
    FailureOr<PipeReference> pipeRef =
        getPipeReferenceForProtocolOp(protocolOp, transferIndex);
    if (failed(pipeRef)) {
      return failure();
    }
    if ((*pipeRef).isStatic()) {
      return success();
    }

    // Repeated identical records have the same PipeKey. Count earlier matches
    // so each graph node maps back to its original record index.
    unsigned selectedOccurrence =
        selectedOccurrencesByProtocolOp[protocolOp][unit.pipe]++;
    unsigned matchingOccurrence = 0;
    for (auto [recordIndex, record] :
         llvm::enumerate((*pipeRef).getRecords().getPipes())) {
      if (!(getPipeKey(record, (*pipeRef).getPipeNetId()) == unit.pipe)) {
        continue;
      }
      if (matchingOccurrence++ == selectedOccurrence) {
        unit.selectedProtocolRecords.push_back(
            {protocolOp, static_cast<unsigned>(recordIndex)});
        return success();
      }
    }
    return protocolOp->emitError(
        "selected pipe record does not match its transfer graph node");
  };

  units.reserve(pipeGraph.getPipeTransferNodes().size());
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    assert(transferNode.sendOp &&
           "pipe transfer graph node must have a send operation");
    auto sendOp = cast<PipeTransferSendOp>(transferNode.sendOp);
    PipeTransferAllocationUnit unit;
    unit.transferNodeId = transferNode.id;
    unit.sendOp = sendOp.getOperation();
    unit.pipe = transferNode.pipe;
    unit.pipeType = PipeType::get(mod.getContext(), unit.pipe.srcX,
                                  unit.pipe.srcY, unit.pipe.dstStartX,
                                  unit.pipe.dstStartY, unit.pipe.dstEndX,
                                  unit.pipe.dstEndY, unit.pipe.pipeNetId);
    unit.transferContract = transferNode.transferContract;
    unit.ordinal = static_cast<int64_t>(transferNode.id);
    unit.protocolOps.push_back(sendOp.getOperation());
    if (failed(recordSelectedProtocolRow(unit, sendOp.getOperation()))) {
      return failure();
    }
    updateIntervalEnd(unit.interval, sendOp.getOperation(), dominanceInfo);
    for (Operation *postOp : transferNode.receiverPostOps) {
      auto ordinalIt = operationOrdinals.find(postOp);
      assert(ordinalIt != operationOrdinals.end() &&
             "receiver post is missing an operation ordinal");
      unit.protocolOps.push_back(postOp);
      if (failed(recordSelectedProtocolRow(unit, postOp))) {
        return failure();
      }
      auto waitIt = waitOpsByPost.find(postOp);
      if (waitIt != waitOpsByPost.end()) {
        unit.protocolOps.append(waitIt->second.begin(), waitIt->second.end());
        for (Operation *waitOp : waitIt->second) {
          if (failed(recordSelectedProtocolRow(unit, waitOp))) {
            return failure();
          }
        }
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
              return pipeResourceUnitsInterfere(units[lhsIndex],
                                                units[rhsIndex], dominanceInfo);
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

static bool usesSenderReadyCounter(
    const PipeTransferAllocationUnit &unit,
    const PipeSynchronizationSelection *synchronizationSelection) {
  // Selected transfers publish receiver addresses, so their sender must wait
  // until the matching table entry has been initialized.
  if (!synchronizationSelection || isSelectedTransferUnit(unit)) {
    return true;
  }
  return !synchronizationSelection->usesCapacityProtocol(
      llvm::cast<PipeTransferSendOp>(unit.sendOp));
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
  llvm::MapVector<FuncOp, SmallVector<int32_t>> dfbIndices;
};

static ComputedAddressPlan
buildComputedAddressPlan(MutableArrayRef<PipeTransferAllocationUnit> units,
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
    // One loop body serves every record, so computed addresses would require
    // record-specific DFB sequence state. Published addresses preserve the
    // receiver's runtime reservation without adding that state to the loop.
    if (isSelectedTransferUnit(unit)) {
      continue;
    }
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

  llvm::DenseMap<FuncOp, SmallVector<int64_t>> sortedDFBIndicesByFunc;
  for (auto &[func, dfbSet] : dfbIndicesByFunc) {
    SmallVector<int64_t> sortedDFBIndices(dfbSet.begin(), dfbSet.end());
    llvm::sort(sortedDFBIndices);
    sortedDFBIndicesByFunc[func] = sortedDFBIndices;

    plan.dfbIndices[func] =
        llvm::map_to_vector(sortedDFBIndices, [](int64_t dfbIndex) {
          return static_cast<int32_t>(dfbIndex);
        });
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

LogicalResult buildPipeResourcePlan(
    ModuleOp mod, const PipeTransferIndex &transferIndex,
    const PipeGraph &pipeGraph, PipeResourcePlan &info,
    bool enableComputedAddresses, PipeCounterAllocationPolicy counterPolicy,
    const PipeSynchronizationSelection *synchronizationSelection) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit, 0>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, transferIndex, pipeGraph,
                                         dominanceInfo, postDominanceInfo,
                                         info.staticallyInactiveOps);
  if (failed(maybeUnits)) {
    return failure();
  }
  SmallVector<PipeTransferAllocationUnit, 0> &units = *maybeUnits;
  SourceColorMap colorUsersBySource =
      assignLiveIntervalColors(units, dominanceInfo);
  ComputedAddressPlan computedAddressPlan;
  if (enableComputedAddresses) {
    computedAddressPlan = buildComputedAddressPlan(units, pipeGraph);
  }
  info.computedAddressCounterInitializations =
      computedAddressPlan.counterInitializations;
  info.computedAddressDFBIndices = computedAddressPlan.dfbIndices;

  SmallVector<SmallVector<PipeKey>> pipesByCompletionCounterColor;
  for (PipeTransferAllocationUnit &unit : units) {
    unit.maybeCompletionCounterColor = allocateCompletionCounterColor(
        unit.pipe, pipesByCompletionCounterColor);
  }
  PipeCounterAllocator counterAllocator(PipeCounterAllocationCounts{},
                                        counterPolicy);
  SmallVector<PipeCounterInfo> completionCounters;
  completionCounters.reserve(pipesByCompletionCounterColor.size());
  while (completionCounters.size() < pipesByCompletionCounterColor.size()) {
    completionCounters.push_back(counterAllocator.allocate());
  }

  auto [readyColorBySourceColor, maxReadyCountersPerSource] =
      compactColors(colorUsersBySource, [&](std::size_t unitIndex) {
        return usesSenderReadyCounter(units[unitIndex],
                                      synchronizationSelection);
      });

  // The same ready color is reused on different source cores, so every source
  // must interpret that color as the same storage kind.
  PipeCounterAllocationCounts counterCounts = counterAllocator.getCounts();
  bool useGlobalReadyCounters =
      counterPolicy == PipeCounterAllocationPolicy::GlobalOnly ||
      counterCounts.localSemaphoreCount + maxReadyCountersPerSource >
          kMaxHardwareSemaphoreIds;

  // A global semaphore index refers to distinct storage on each source core.
  // Only counters live on the same source need distinct indices.
  SmallVector<PipeCounterInfo> readyCounterByColor;
  readyCounterByColor.reserve(maxReadyCountersPerSource);
  for (int64_t color = 0; color < maxReadyCountersPerSource; ++color) {
    readyCounterByColor.push_back(useGlobalReadyCounters
                                      ? counterAllocator.allocateGlobal()
                                      : counterAllocator.allocate());
  }

  auto [addressColorBySourceColor, maxAddressColorsPerSource] =
      compactColors(colorUsersBySource, [&](std::size_t unitIndex) {
        return computedAddressPlan.infoByUnitIndex.find(unitIndex) ==
               computedAddressPlan.infoByUnitIndex.end();
      });
  int64_t maxAddressTableBytes =
      maxAddressColorsPerSource * kPipeAddressWordBytes;
  llvm::MapVector<Operation *, SmallVector<std::optional<PipeResourceInfo>>>
      selectedResources;

  for (auto indexedUnit : llvm::enumerate(units)) {
    const PipeTransferAllocationUnit &unit = indexedUnit.value();
    assert(unit.maybeCompletionCounterColor &&
           "pipe transfer is missing a completion counter color");
    int64_t completionColor = *unit.maybeCompletionCounterColor;
    assert(completionColor < static_cast<int64_t>(completionCounters.size()));
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    std::optional<PipeCounterInfo> maybeReadyCounter;
    if (usesSenderReadyCounter(unit, synchronizationSelection)) {
      auto sourceIt = readyColorBySourceColor.find(sourceKey);
      assert(sourceIt != readyColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      int64_t readyColor = colorIt->second;
      assert(readyColor < static_cast<int64_t>(readyCounterByColor.size()));
      maybeReadyCounter = readyCounterByColor[readyColor];
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
        PipeCompletionInfo{completionCounters[completionColor]},
        maybeReadyCounter,
        addressStorage,
    };
    llvm::SmallPtrSet<Operation *, 4> selectedProtocolOps;
    for (auto [protocolOp, recordIndex] : unit.selectedProtocolRecords) {
      selectedProtocolOps.insert(protocolOp);
      SmallVector<std::optional<PipeResourceInfo>> &resources =
          selectedResources[protocolOp];
      if (resources.empty()) {
        FailureOr<PipeReference> pipeRef =
            getPipeReferenceForProtocolOp(protocolOp, transferIndex);
        assert(succeeded(pipeRef) &&
               "pipe transfer graph validated pipe reference");
        resources.resize((*pipeRef).getRecords().getPipes().size());
      }
      assert(recordIndex < resources.size() && "selected record index invalid");
      resources[recordIndex] = pipeResource;
    }
    for (Operation *protocolOp : unit.protocolOps) {
      if (selectedProtocolOps.contains(protocolOp)) {
        continue;
      }
      auto [resourceIt, inserted] =
          info.resources.insert({protocolOp, pipeResource});
      assert((inserted || resourceIt->second.pipe == pipeResource.pipe) &&
             "pipe protocol operation assigned to two transfers");
    }
  }

  for (auto &[protocolOp, optionalResources] : selectedResources) {
    SmallVector<PipeResourceInfo> resources;
    resources.reserve(optionalResources.size());
    auto firstActiveResource = llvm::find_if(
        optionalResources, [](const std::optional<PipeResourceInfo> &resource) {
          return resource.has_value();
        });
    assert(firstActiveResource != optionalResources.end() &&
           "selected resource table has no active records");
    for (const std::optional<PipeResourceInfo> &resource : optionalResources) {
      // A record proven inactive at this operation never reads its table row.
      // Reusing an active row keeps every table aligned with record indices.
      resources.push_back(resource.value_or(**firstActiveResource));
    }
    info.selectedResources.insert({protocolOp, std::move(resources)});
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
  PipeCounterAllocationCounts counts;
  for (const PipeResourceInfo &resource :
       llvm::make_second_range(info.resources)) {
    counts.include(resource.completion.counter);
    if (resource.readyCounter) {
      counts.include(*resource.readyCounter);
    }
  }

  for (const auto &[protocolOp, resources] : info.selectedResources) {
    (void)protocolOp;
    for (const PipeResourceInfo &resource : resources) {
      counts.include(resource.completion.counter);
      if (resource.readyCounter) {
        counts.include(*resource.readyCounter);
      }
    }
  }
  if (pipeCapacityPlan) {
    PipeCounterAllocationCounts capacityCounts =
        pipeCapacityPlan->getCounterAllocationCounts();
    assert(capacityCounts.localSemaphoreCount >= counts.localSemaphoreCount &&
           capacityCounts.globalSemaphoreCount >= counts.globalSemaphoreCount &&
           "capacity allocation must continue after pipe resource allocation");
    counts = capacityCounts;
  }
  return PipeResourceRequirements{
      counts.localSemaphoreCount,
      counts.globalSemaphoreCount,
      info.sramScratch.bytes,
  };
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
