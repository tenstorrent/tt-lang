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
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"
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
  static Key getEmptyKey() {
    int64_t sentinel = DenseMapInfo<int64_t>::getEmptyKey();
    return {sentinel, sentinel};
  }
  static Key getTombstoneKey() {
    int64_t sentinel = DenseMapInfo<int64_t>::getTombstoneKey();
    return {sentinel, sentinel};
  }
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
                       const PipeResourcePlan *pipeResourcePlan) {
  assert(pipeResourcePlan && "missing pipe resource plan");
  auto it = pipeResourcePlan->resources.find(createOp.getOperation());
  assert(it != pipeResourcePlan->resources.end() &&
         "pipe transfer missing from pipe resource plan");
  return it->second;
}

static PipeCompletionWaitInfo
lookupPipeCompletionWaitInfo(int64_t pipeNetId,
                             const PipeResourcePlan *pipeResourcePlan) {
  assert(pipeResourcePlan && "missing pipe resource plan");
  auto it = pipeResourcePlan->completionWaits.find(pipeNetId);
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

static Value loadIndexTableEntry(Operation *op, Location loc,
                                 ArrayRef<int64_t> values, Value recordIndex,
                                 ConversionPatternRewriter &rewriter);

static Value buildSelectedReadyCounterAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  assert(!resources.empty() && "selected pipe resource table is empty");
  if (std::holds_alternative<PipeLocalReadyCounterInfo>(
          resources.front().readyCounter)) {
    SmallVector<int64_t> senderReadySemIdxs;
    senderReadySemIdxs.reserve(resources.size());
    for (const PipeResourceInfo &resource : resources) {
      const auto *localCounter =
          std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter);
      assert(localCounter && "selected resource ready-counter kind mismatch");
      senderReadySemIdxs.push_back(localCounter->senderReadySemIdx);
    }
    Value senderReadySemIdx =
        loadIndexTableEntry(op, loc, senderReadySemIdxs, recordIndex, rewriter);
    return ttk::GetSemaphoreOp::create(rewriter, loc, senderReadySemIdx)
        .getResult();
  }

  SmallVector<int64_t> globalSemaphoreIndices;
  globalSemaphoreIndices.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    const auto *globalCounter =
        std::get_if<PipeGlobalReadyCounterInfo>(&resource.readyCounter);
    assert(globalCounter && "selected resource ready-counter kind mismatch");
    globalSemaphoreIndices.push_back(globalCounter->globalSemaphoreIndex);
  }
  Value globalIndex = loadIndexTableEntry(op, loc, globalSemaphoreIndices,
                                          recordIndex, rewriter);
  int64_t baseRuntimeArgIndex = getPipeRuntimeCommonArgIndex(
      op, getFirstPipeGlobalSemaphoreArgOffset(pipeResourcePlan));
  Value base =
      arith::ConstantIndexOp::create(rewriter, loc, baseRuntimeArgIndex);
  Value runtimeArgIndex =
      arith::AddIOp::create(rewriter, loc, base, globalIndex);
  return ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                        runtimeArgIndex)
      .getResult();
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

static Value
buildIndexTableAtFunctionEntry(Operation *op, Location loc,
                               ArrayRef<int64_t> values,
                               ConversionPatternRewriter &rewriter) {
  assert(!values.empty() && "selected pipe resource table must not be empty");
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "selected pipe op must be inside a function");

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  auto memrefTy = MemRefType::get({static_cast<int64_t>(values.size())},
                                  rewriter.getIndexType());
  Value table = memref::AllocaOp::create(rewriter, loc, memrefTy);
  for (auto [index, value] : llvm::enumerate(values)) {
    Value tableIndex = arith::ConstantIndexOp::create(rewriter, loc, index);
    Value tableValue = arith::ConstantIndexOp::create(rewriter, loc, value);
    memref::StoreOp::create(rewriter, loc, tableValue, table,
                            ValueRange{tableIndex});
  }
  return table;
}

static Value loadIndexTableEntry(Operation *op, Location loc,
                                 ArrayRef<int64_t> values, Value recordIndex,
                                 ConversionPatternRewriter &rewriter) {
  Value table = buildIndexTableAtFunctionEntry(op, loc, values, rewriter);
  return memref::LoadOp::create(rewriter, loc, table, ValueRange{recordIndex});
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

static Value buildSelectedAddressTableAddress(
    Operation *op, Location loc, ArrayRef<PipeResourceInfo> resources,
    Value recordIndex, const PipeResourcePlan &pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> byteOffsets;
  byteOffsets.reserve(resources.size());
  for (const PipeResourceInfo &resource : resources) {
    byteOffsets.push_back(resource.addressStorage.sramAddressTable.byteOffset);
  }
  Value byteOffset =
      loadIndexTableEntry(op, loc, byteOffsets, recordIndex, rewriter);
  Value scratchBase = buildPipeRuntimeCommonArg(
      loc, rewriter, getPipeRuntimeCommonArgIndex(op, 0));
  (void)pipeResourcePlan;
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

struct ReceiverPublishedAddressInfo {
  Value receiverDFB;
  ttcore::TileType tileType;
};

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
  int64_t pipeNetId;
};

static SelectedPipeFields getSelectedPipeFields(const PipeReference &pipeRef) {
  assert(pipeRef.isSelected() && "expected selected pipe reference");
  if (pipeRef.isSelectedSrc()) {
    SelectPipeSrcOp op = pipeRef.selectedSrc;
    return SelectedPipeFields{
        op.getRecordIndex(),   op.getSrcX(),
        op.getSrcY(),          op.getDstStartX(),
        op.getDstStartY(),     op.getDstEndX(),
        op.getDstEndY(),       op.getNumDests(),
        op.getSrcInDstRange(), static_cast<int64_t>(op.getPipeNetId())};
  }
  SelectPipeDstOp op = pipeRef.selectedDst;
  return SelectedPipeFields{
      op.getRecordIndex(),   op.getSrcX(),
      op.getSrcY(),          op.getDstStartX(),
      op.getDstStartY(),     op.getDstEndX(),
      op.getDstEndY(),       op.getNumDests(),
      op.getSrcInDstRange(), static_cast<int64_t>(op.getPipeNetId())};
}

static ArrayRef<PipeResourceInfo>
lookupSelectedPipeResources(PipeTransferCreateOp createOp,
                            const PipeResourcePlan *pipeResourcePlan) {
  assert(pipeResourcePlan && "missing pipe resource plan");
  auto it = pipeResourcePlan->selectedResources.find(createOp.getOperation());
  assert(it != pipeResourcePlan->selectedResources.end() &&
         "selected pipe transfer missing from pipe resource plan");
  return it->second;
}

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
        if (getAttachedCB(post.getDst())) {
          FailureOr<PipeReference> pipeRef =
              getPipeReference(post, createOp.getPipe());
          assert(succeeded(pipeRef) && "pipe transfer create verifier failed");
          pipeNetIds.insert((*pipeRef).getPipeNetId());
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
static LogicalResult
lowerSelectedPipeTransferSend(PipeTransferSendOp op, Value srcCB,
                              bool isConsumerCB, PipeTransferCreateOp createOp,
                              const PipeReference &pipeRef,
                              const PipeResourcePlan *pipeResourcePlan,
                              ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources =
      lookupSelectedPipeResources(createOp, pipeResourcePlan);
  PipeCompletionWaitInfo completionInfo =
      lookupPipeCompletionWaitInfo(fields.pipeNetId, pipeResourcePlan);
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto cbType = getTTLCBType(srcCB);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto tileType = llvm::dyn_cast<ttcore::TileType>(cbType.getElementType());
  if (!tileType) {
    return rewriter.notifyMatchFailure(op, "CB element type must be tile");
  }

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, *pipeResourcePlan, rewriter);
  auto senderSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
  Value expectedSignals = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), fields.numDests);
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedSignals);
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked source DFB type");
  Value srcPtrIdx;
  if (isConsumerCB) {
    auto cbReadPtr = ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), cbReadPtr);
  } else {
    auto srcWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted);
    srcPtrIdx = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), srcWritePtr);
  }

  int64_t cbNumTiles = 1;
  for (int64_t dimension : cbType.getShape()) {
    cbNumTiles *= dimension;
  }
  Value srcAddr = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI32Type(), srcPtrIdx);
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(cbNumTiles * tileType.getSizeBytes()));

  auto dstStartXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.dstStartX);
  auto dstStartYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.dstStartY);
  auto dstEndXVal = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.dstEndX);
  auto dstEndYVal = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.dstEndY);
  Value numDestsVal = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI32Type(), fields.numDests);
  Value oneDst = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value isCollective = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ne, fields.numDests, oneDst);

  int64_t nocIdx = getNocIndex(op);
  // The NOC ops take a required `noc` operand; always materialize it (noc 0
  // is a valid index), matching the static-pipe send path.
  Value nocVal = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));

  Value tableAddress = buildSelectedAddressTableAddress(
      op, loc, resources, fields.recordIndex, *pipeResourcePlan, rewriter);
  Value dstAddr =
      buildAddressTableDestinationAddress(loc, tableAddress, rewriter);

  auto writeIf =
      scf::IfOp::create(rewriter, loc, isCollective, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&writeIf.getThenRegion().front());
    auto loopbackIf = scf::IfOp::create(rewriter, loc, fields.srcInDstRange,
                                        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard loopbackGuard(rewriter);
      rewriter.setInsertionPointToStart(&loopbackIf.getThenRegion().front());
      ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
          rewriter, loc, srcAddr, totalSizeVal, numDestsVal, dstStartXVal,
          dstStartYVal, dstEndXVal, dstEndYVal, dstAddr, nocVal,
          /*linked=*/nullptr);
      rewriter.setInsertionPointToStart(&loopbackIf.getElseRegion().front());
      ttk::NocAsyncWriteMulticastOp::create(
          rewriter, loc, srcAddr, totalSizeVal, numDestsVal, dstStartXVal,
          dstStartYVal, dstEndXVal, dstEndYVal, dstAddr, nocVal,
          /*linked=*/nullptr);
    }

    rewriter.setInsertionPointToStart(&writeIf.getElseRegion().front());
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr,
                                 ValueRange{dstStartXVal, dstStartYVal},
                                 /*dstBankId=*/ValueRange{}, dstAddr,
                                 totalSizeVal);
  }
  rewriter.setInsertionPointAfter(writeIf);

  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);

  auto recvSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, completionInfo.receiverSemIdx);
  auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
  auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto completionIf =
      scf::IfOp::create(rewriter, loc, isCollective, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&completionIf.getThenRegion().front());
    auto recvSemMcastAddr = ttk::GetNocMulticastAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
        recvSemAddr, nocVal);
    auto oneI32 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    Value numRemoteWithLoopback =
        arith::SubIOp::create(rewriter, loc, numDestsVal, oneI32);
    Value numRemoteDests =
        arith::SelectOp::create(rewriter, loc, fields.srcInDstRange,
                                numRemoteWithLoopback, numDestsVal);
    ttk::NocSemaphoreIncMulticastOp::create(
        rewriter, loc, recvSemMcastAddr.getResult(), incrVal, numRemoteDests,
        /*noc_id=*/Value(), /*posted=*/BoolAttr());

    auto selfIncIf = scf::IfOp::create(rewriter, loc, fields.srcInDstRange,
                                       /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&selfIncIf.getThenRegion().front());
      auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
          rewriter, loc, rewriter.getIndexType(), fields.srcX);
      auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
          rewriter, loc, rewriter.getIndexType(), fields.srcY);
      auto selfRecvSemNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, srcXTranslated, srcYTranslated, recvSemAddr);
      ttk::NocSemaphoreIncOp::create(rewriter, loc,
                                     selfRecvSemNocAddr.getResult(), incrVal,
                                     /*noc_id=*/Value(), /*posted=*/BoolAttr());
    }
    rewriter.setInsertionPointAfter(selfIncIf);
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, /*noc_id=*/Value());

    rewriter.setInsertionPointToStart(&completionIf.getElseRegion().front());
    auto dstSemNocAddr = ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal,
                                                   dstStartYVal, recvSemAddr);
    ttk::NocSemaphoreIncOp::create(rewriter, loc, dstSemNocAddr.getResult(),
                                   incrVal, /*noc_id=*/Value(),
                                   /*posted=*/BoolAttr());
  }
  rewriter.setInsertionPointAfter(completionIf);

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeTransferSend(PipeTransferSendOp op, Value srcCB,
                                    bool isConsumerCB,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  FailureOr<PipeTransferCreateOp> createOp =
      getPipeTransferCreate(op, op.getTransfer());
  if (failed(createOp)) {
    return failure();
  }
  FailureOr<PipeReference> pipeRef =
      getPipeReference(op, (*createOp).getPipe());
  if (failed(pipeRef)) {
    return failure();
  }
  if ((*pipeRef).isSelected()) {
    return lowerSelectedPipeTransferSend(op, srcCB, isConsumerCB, *createOp,
                                         *pipeRef, pipeResourcePlan, rewriter);
  }
  auto pipeType = (*pipeRef).pipeType;
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(*createOp, pipeResourcePlan);
  PipeCompletionWaitInfo completionInfo =
      lookupPipeCompletionWaitInfo(pipeType.getPipeNetId(), pipeResourcePlan);
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

static LogicalResult lowerSelectedPipeTransferPost(
    PipeTransferPostOp op, Value dst, PipeTransferCreateOp createOp,
    const PipeReference &pipeRef, const PipeResourcePlan *pipeResourcePlan,
    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  SelectedPipeFields fields = getSelectedPipeFields(pipeRef);
  ArrayRef<PipeResourceInfo> resources =
      lookupSelectedPipeResources(createOp, pipeResourcePlan);
  FailureOr<ReceiverPublishedAddressInfo> publishedAddressInfo =
      getReceiverPublishedAddressInfo(op, dst, rewriter);
  if (failed(publishedAddressInfo)) {
    return failure();
  }

  int64_t nocIdx = getNocIndex(op);
  Value nocVal;
  Value inlineNocId = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(nocIdx));
  if (nocIdx > 0) {
    nocVal = inlineNocId;
  }

  auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.srcX);
  auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, rewriter.getIndexType(), fields.srcY);

  Value publishedAddress =
      buildReceiverPublishedAddress(dst, loc, *publishedAddressInfo, rewriter);
  Value tableAddress = buildSelectedAddressTableAddress(
      op, loc, resources, fields.recordIndex, *pipeResourcePlan, rewriter);
  auto senderTableNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, tableAddress);
  auto byteEnableAll = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
  ttk::NocInlineDwWriteOp::create(rewriter, loc, senderTableNocAddr.getResult(),
                                  publishedAddress, byteEnableAll, inlineNocId);
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);

  Value senderSemAddr = buildSelectedReadyCounterAddress(
      op, loc, resources, fields.recordIndex, *pipeResourcePlan, rewriter);
  auto senderSemNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, senderSemAddr);
  auto readyIncr = arith::ConstantIndexOp::create(rewriter, loc, 1);
  ttk::NocSemaphoreIncOp::create(rewriter, loc, senderSemNocAddr.getResult(),
                                 readyIncr, nocVal, /*posted=*/BoolAttr());

  auto token = UnrealizedConversionCastOp::create(
      rewriter, loc, op.getToken().getType(), ValueRange{});
  rewriter.replaceOp(op, token.getResult(0));
  return success();
}

LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  FailureOr<PipeTransferCreateOp> createOp =
      getPipeTransferCreate(op, op.getTransfer());
  if (failed(createOp)) {
    return failure();
  }
  FailureOr<PipeReference> pipeRef =
      getPipeReference(op, (*createOp).getPipe());
  if (failed(pipeRef)) {
    return failure();
  }
  if ((*pipeRef).isSelected()) {
    return lowerSelectedPipeTransferPost(op, dst, *createOp, *pipeRef,
                                         pipeResourcePlan, rewriter);
  }
  auto pipeType = (*pipeRef).pipeType;
  PipeResourceInfo pipeResource =
      lookupPipeResourceInfo(*createOp, pipeResourcePlan);
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

  auto token = UnrealizedConversionCastOp::create(
      rewriter, loc, op.getToken().getType(), ValueRange{});
  rewriter.replaceOp(op, token.getResult(0));
  return success();
}

/// Lower the receiver completion wait with a per-PipeNet runtime counter.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op,
                                    const PipeNetCounterMap *counters,
                                    const PipeResourcePlan *pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto tokenType = mlir::cast<PipeTokenType>(op.getToken().getType());
  auto completionIt =
      pipeResourcePlan->completionWaits.find(tokenType.getPipeNetId());
  if (completionIt == pipeResourcePlan->completionWaits.end()) {
    op.emitError("pipe transfer wait references PipeNet ")
        << tokenType.getPipeNetId() << " with no completion resource";
    return failure();
  }
  PipeCompletionWaitInfo completionInfo = completionIt->second;

  Value counter;
  if (counters) {
    auto func = op->getParentOfType<func::FuncOp>();
    auto fIt = counters->find(func);
    if (fIt != counters->end()) {
      auto pIt = fIt->second.find(tokenType.getPipeNetId());
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

/// One ordered post/send operation used to validate bounded rendezvous depth.
struct PipeTransferRendezvousEvent {
  /// Pipe transfer post or send operation.
  Operation *op;
  /// Whether the operation creates or consumes one posted rendezvous phase.
  PipeTransferRendezvousEventKind kind;
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

  /// Selected transfer create operations and row indices represented by this
  /// allocation unit.
  SmallVector<std::pair<Operation *, unsigned>> selectedTransferRows;

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

  /// True when at least one receive post contributes to the interval.
  bool hasPost = false;

  /// True when at least one send contributes to the interval.
  bool hasSend = false;

  /// Assigned first-fit color within the source node's allocation group.
  int64_t resourceColor = 0;
};

static bool isSelectedTransferUnit(const PipeTransferAllocationUnit &unit) {
  return !unit.selectedTransferRows.empty();
}

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
            "one live receive post per pipe before each send in a linear block";
}

static LogicalResult
validateMaxLivePostsPerLinearBlock(const PipeTransferAllocationUnit &unit,
                                   int64_t maxLivePosts) {
  if (unit.rendezvousEvents.size() <= static_cast<size_t>(maxLivePosts + 1)) {
    return success();
  }

  llvm::MapVector<Block *, SmallVector<PipeTransferRendezvousEvent>>
      eventsByBlock;
  for (const PipeTransferRendezvousEvent &event : unit.rendezvousEvents) {
    eventsByBlock[event.op->getBlock()].push_back(event);
  }

  for (auto &entry : eventsByBlock) {
    SmallVector<PipeTransferRendezvousEvent> &events = entry.second;
    if (events.size() <= static_cast<size_t>(maxLivePosts + 1)) {
      continue;
    }

    llvm::sort(events, [](const PipeTransferRendezvousEvent &lhs,
                          const PipeTransferRendezvousEvent &rhs) {
      return lhs.op->isBeforeInBlock(rhs.op);
    });

    int64_t livePosts = 0;
    for (const PipeTransferRendezvousEvent &event : events) {
      switch (event.kind) {
      case PipeTransferRendezvousEventKind::Post:
        ++livePosts;
        if (livePosts > maxLivePosts) {
          return emitUnsupportedQueueDepth(event.op, unit);
        }
        break;
      case PipeTransferRendezvousEventKind::Send:
        if (livePosts > 0) {
          --livePosts;
        }
        break;
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

static bool pipeResourceUnitsInterfere(const PipeTransferAllocationUnit &lhs,
                                       const PipeTransferAllocationUnit &rhs,
                                       const DominanceInfo &dominanceInfo) {
  if (isSelectedTransferUnit(lhs) || isSelectedTransferUnit(rhs)) {
    return true;
  }
  return pipeTransferIntervalsOverlap(lhs, rhs, dominanceInfo);
}

static Operation *getPipeTransferIntervalBoundary(Operation *protocolOp,
                                                  Value transfer) {
  FailureOr<PipeTransferCreateOp> createOp =
      getPipeTransferCreate(protocolOp, transfer);
  if (failed(createOp)) {
    return protocolOp;
  }
  FailureOr<PipeReference> pipeRef =
      getPipeReference(protocolOp, (*createOp).getPipe());
  if (failed(pipeRef) || !(*pipeRef).isSelected()) {
    return protocolOp;
  }
  Operation *selectedOp = (*pipeRef).isSelectedSrc()
                              ? (*pipeRef).selectedSrc.getOperation()
                              : (*pipeRef).selectedDst.getOperation();
  if (auto forOp = selectedOp->getParentOfType<scf::ForOp>()) {
    return forOp.getOperation();
  }
  return protocolOp;
}

static FailureOr<SmallVector<PipeTransferAllocationUnit, 0>>
collectPipeTransferAllocationUnits(ModuleOp mod,
                                   const DominanceInfo &dominanceInfo,
                                   const PostDominanceInfo &postDominanceInfo) {
  SmallVector<PipeTransferAllocationUnit, 0> units;
  llvm::MapVector<Operation *, unsigned> indexByTransferCreateOp;
  llvm::MapVector<Operation *, SmallVector<unsigned>>
      indicesBySelectedTransferCreateOp;
  llvm::MapVector<PipeKey, unsigned> indexByPipe;
  int64_t nextOrdinal = 0;
  int64_t nextEventOrdinal = 0;

  auto addUnitForPipe = [&](Operation *transferCreateOp, PipeType pipeType,
                            PipeTransferContract transferContract)
      -> PipeTransferAllocationUnit * {
    PipeKey pipe = getPipeKey(pipeType);
    auto existingPipe = indexByPipe.find(pipe);
    if (existingPipe != indexByPipe.end()) {
      PipeTransferAllocationUnit &unit = units[existingPipe->second];
      if (isCollectiveTransfer(transferContract)) {
        unit.transferContract = PipeTransferContract::Collective;
      }
      return &unit;
    }

    PipeTransferAllocationUnit unit;
    unit.pipe = pipe;
    unit.pipeType = pipeType;
    unit.transferContract = transferContract;
    unit.ordinal = nextOrdinal++;
    indexByPipe.insert({pipe, units.size()});
    units.push_back(unit);
    return &units.back();
  };

  auto getOrCreateUnits =
      [&](Operation *protocolOp,
          Value transfer) -> FailureOr<SmallVector<unsigned>> {
    FailureOr<PipeTransferCreateOp> createOp =
        getPipeTransferCreate(protocolOp, transfer);
    if (failed(createOp)) {
      return failure();
    }

    Operation *transferCreateOp = (*createOp).getOperation();
    FailureOr<PipeReference> pipeRef =
        getPipeReference(protocolOp, (*createOp).getPipe());
    if (failed(pipeRef)) {
      return failure();
    }

    SmallVector<unsigned> result;
    if ((*pipeRef).isSelected()) {
      auto existing = indicesBySelectedTransferCreateOp.find(transferCreateOp);
      if (existing != indicesBySelectedTransferCreateOp.end()) {
        for (unsigned unitIndex : existing->second) {
          result.push_back(unitIndex);
        }
        return result;
      }

      PipeNetRecordsAttr records = (*pipeRef).getRecords();
      SmallVector<unsigned> unitIndices;
      unitIndices.reserve(records.getPipes().size());
      for (auto [recordIndex, record] : llvm::enumerate(records.getPipes())) {
        PipeType pipeType = getPipeTypeFromRecord(
            protocolOp->getContext(), record, records.getPipeNetId());
        PipeTransferContract transferContract = getPipeTransferContract(record);
        PipeTransferAllocationUnit *unit =
            addUnitForPipe(transferCreateOp, pipeType, transferContract);
        unit->selectedTransferRows.push_back(
            {transferCreateOp, static_cast<unsigned>(recordIndex)});
        unsigned unitIndex = unit - units.data();
        unitIndices.push_back(unitIndex);
        result.push_back(unitIndex);
      }
      indicesBySelectedTransferCreateOp.insert(
          {transferCreateOp, std::move(unitIndices)});
      return result;
    }

    auto existing = indexByTransferCreateOp.find(transferCreateOp);
    if (existing != indexByTransferCreateOp.end()) {
      result.push_back(existing->second);
      return result;
    }

    PipeTransferAllocationUnit *unit =
        addUnitForPipe(transferCreateOp, (*pipeRef).pipeType,
                       getPipeTransferContract(*createOp));
    unit->transferCreateOps.push_back(transferCreateOp);
    unsigned unitIndex = unit - units.data();
    indexByTransferCreateOp.insert({transferCreateOp, unitIndex});
    result.push_back(unitIndex);
    return result;
  };

  // Resource allocation depends only on receive posts and sends. Walk the
  // module once in operation order to form per-pipe allocation units, record
  // rendezvous events for queue-depth validation, and build post-to-send live
  // intervals for coloring.
  WalkResult walkResult = mod.walk([&](Operation *op) {
    if (auto postOp = dyn_cast<PipeTransferPostOp>(op)) {
      int64_t eventOrdinal = nextEventOrdinal++;
      Operation *intervalBoundary =
          getPipeTransferIntervalBoundary(op, postOp.getTransfer());
      FailureOr<SmallVector<unsigned>> unitsForTransfer =
          getOrCreateUnits(op, postOp.getTransfer());
      if (failed(unitsForTransfer)) {
        return WalkResult::interrupt();
      }
      for (unsigned unitIndex : *unitsForTransfer) {
        PipeTransferAllocationUnit &unit = units[unitIndex];
        unit.hasPost = true;
        unit.rendezvousEvents.push_back(
            {op, PipeTransferRendezvousEventKind::Post});
        updateIntervalStart(unit.interval, intervalBoundary, eventOrdinal,
                            dominanceInfo);
      }
      return WalkResult::advance();
    }

    if (auto sendOp = dyn_cast<PipeTransferSendOp>(op)) {
      Operation *intervalBoundary =
          getPipeTransferIntervalBoundary(op, sendOp.getTransfer());
      FailureOr<SmallVector<unsigned>> unitsForTransfer =
          getOrCreateUnits(op, sendOp.getTransfer());
      if (failed(unitsForTransfer)) {
        return WalkResult::interrupt();
      }
      for (unsigned unitIndex : *unitsForTransfer) {
        PipeTransferAllocationUnit &unit = units[unitIndex];
        unit.hasSend = true;
        unit.rendezvousEvents.push_back(
            {op, PipeTransferRendezvousEventKind::Send});
        updateIntervalEnd(unit.interval, intervalBoundary, dominanceInfo);
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  for (PipeTransferAllocationUnit &unit : units) {
    if (failed(validateMaxLivePostsPerLinearBlock(unit,
                                                  /*maxLivePosts=*/1))) {
      return failure();
    }
    finalizeInterval(unit.interval, unit.hasPost, unit.hasSend, dominanceInfo,
                     postDominanceInfo);
  }

  return units;
}

static bool
isBeforeForDeterministicAllocation(const PipeTransferAllocationUnit &lhs,
                                   const PipeTransferAllocationUnit &rhs) {
  return std::make_tuple(lhs.interval.startOrdinal, lhs.pipe.srcX,
                         lhs.pipe.srcY, lhs.pipe.pipeNetId, lhs.pipe.dstStartX,
                         lhs.pipe.dstStartY, lhs.pipe.dstEndX, lhs.pipe.dstEndY,
                         lhs.ordinal) <
         std::make_tuple(rhs.interval.startOrdinal, rhs.pipe.srcX,
                         rhs.pipe.srcY, rhs.pipe.pipeNetId, rhs.pipe.dstStartX,
                         rhs.pipe.dstStartY, rhs.pipe.dstEndX, rhs.pipe.dstEndY,
                         rhs.ordinal);
}

using SourceColorMap =
    llvm::MapVector<PipeSourceKey, SmallVector<SmallVector<unsigned>>>;

using PipeResourceInterferenceGraph =
    SmallVector<llvm::SmallSetVector<unsigned, 8>>;

static void addPipeResourceInterference(PipeResourceInterferenceGraph &graph,
                                        unsigned lhsIndex, unsigned rhsIndex) {
  graph[lhsIndex].insert(rhsIndex);
  graph[rhsIndex].insert(lhsIndex);
}

static PipeResourceInterferenceGraph
buildPipeResourceInterferenceGraph(ArrayRef<unsigned> unitIndices,
                                   ArrayRef<PipeTransferAllocationUnit> units,
                                   const DominanceInfo &dominanceInfo) {
  PipeResourceInterferenceGraph graph;
  graph.resize(units.size());
  for (auto indexedLhs : llvm::enumerate(unitIndices)) {
    unsigned lhsIndex = indexedLhs.value();
    for (unsigned rhsPosition = indexedLhs.index() + 1;
         rhsPosition < unitIndices.size(); ++rhsPosition) {
      unsigned rhsIndex = unitIndices[rhsPosition];
      if (pipeResourceUnitsInterfere(units[lhsIndex], units[rhsIndex],
                                     dominanceInfo)) {
        addPipeResourceInterference(graph, lhsIndex, rhsIndex);
      }
    }
  }
  return graph;
}

static SmallVector<SmallVector<unsigned>>
colorPipeResourceInterferenceGraph(ArrayRef<unsigned> unitIndices,
                                   const PipeResourceInterferenceGraph &graph,
                                   ArrayRef<PipeTransferAllocationUnit> units) {
  SmallVector<unsigned> sortedUnitIndices(unitIndices.begin(),
                                          unitIndices.end());
  llvm::sort(sortedUnitIndices, [&](unsigned lhsIndex, unsigned rhsIndex) {
    return isBeforeForDeterministicAllocation(units[lhsIndex], units[rhsIndex]);
  });

  SmallVector<SmallVector<unsigned>> colorUsers;
  for (unsigned unitIndex : sortedUnitIndices) {
    unsigned selectedColor = 0;
    for (;; ++selectedColor) {
      if (selectedColor == colorUsers.size()) {
        colorUsers.push_back({});
        break;
      }
      bool hasConflict = llvm::any_of(
          colorUsers[selectedColor], [&](unsigned assignedUnitIndex) {
            return graph[unitIndex].count(assignedUnitIndex) != 0;
          });
      if (!hasConflict) {
        break;
      }
    }
    colorUsers[selectedColor].push_back(unitIndex);
  }

  return colorUsers;
}

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
    PipeResourceInterferenceGraph interferenceGraph =
        buildPipeResourceInterferenceGraph(entry.second, units, dominanceInfo);
    SmallVector<SmallVector<unsigned>> colorUsers =
        colorPipeResourceInterferenceGraph(entry.second, interferenceGraph,
                                           units);

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      for (unsigned unitIndex : indexedColor.value()) {
        units[unitIndex].resourceColor = indexedColor.index();
      }
    }

    colorUsersBySource.insert({entry.first, std::move(colorUsers)});
  }

  return colorUsersBySource;
}

LogicalResult buildPipeResourcePlan(ModuleOp mod, PipeResourcePlan &info) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit, 0>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, dominanceInfo, postDominanceInfo);
  if (failed(maybeUnits)) {
    return failure();
  }
  SmallVector<PipeTransferAllocationUnit, 0> &units = *maybeUnits;
  SourceColorMap colorUsersBySource =
      assignLiveIntervalColors(units, dominanceInfo);

  llvm::SmallSetVector<int64_t, 4> activePipeNetIds;
  for (const PipeTransferAllocationUnit &unit : units) {
    activePipeNetIds.insert(unit.pipe.pipeNetId);
  }

  SmallVector<int64_t> sortedPipeNetIds(activePipeNetIds.begin(),
                                        activePipeNetIds.end());
  llvm::sort(sortedPipeNetIds);

  int64_t firstSourceLocalSemIdx = 0;
  for (int64_t pipeNetId : sortedPipeNetIds) {
    int64_t receiverSemIdx = getReceiverCompletionSemIdx(pipeNetId);
    info.completionWaits[pipeNetId] =
        PipeCompletionWaitInfo{pipeNetId, receiverSemIdx};
    firstSourceLocalSemIdx =
        std::max(firstSourceLocalSemIdx, receiverSemIdx + 1);
  }

  int64_t maxReadyCountersPerSource = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    (void)sourceKey;
    maxReadyCountersPerSource =
        std::max<int64_t>(maxReadyCountersPerSource, colorUsers.size());
  }

  // Use one ready-counter kind per kernel so host allocation has one compact
  // descriptor layout.
  bool useGlobalReadyCounters =
      firstSourceLocalSemIdx + maxReadyCountersPerSource >
      kMaxHardwareSemaphoreIds;

  llvm::MapVector<PipeSourceKey, SmallVector<int64_t>> globalIndexBySourceColor;
  int64_t nextGlobalSemaphoreIndex = 0;
  if (useGlobalReadyCounters) {
    for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
      SmallVector<int64_t> &indices = globalIndexBySourceColor[sourceKey];
      indices.reserve(colorUsers.size());
      for (unsigned color = 0, colorCount = colorUsers.size();
           color < colorCount; ++color) {
        indices.push_back(nextGlobalSemaphoreIndex++);
      }
    }
  }

  int64_t maxAddressTableBytes = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    (void)sourceKey;
    maxAddressTableBytes = std::max<int64_t>(
        maxAddressTableBytes, colorUsers.size() * kPipeAddressWordBytes);
  }

  for (const PipeTransferAllocationUnit &unit : units) {
    PipeResourceInfo pipeResource{};
    pipeResource.pipe = unit.pipe;
    pipeResource.transferContract = unit.transferContract;
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    if (useGlobalReadyCounters) {
      auto globalIt = globalIndexBySourceColor.find(sourceKey);
      assert(globalIt != globalIndexBySourceColor.end());
      assert(unit.resourceColor <
             static_cast<int64_t>(globalIt->second.size()));
      pipeResource.readyCounter =
          PipeGlobalReadyCounterInfo{globalIt->second[unit.resourceColor]};
    } else {
      pipeResource.readyCounter = PipeLocalReadyCounterInfo{
          firstSourceLocalSemIdx + unit.resourceColor};
    }
    pipeResource.addressStorage.sramAddressTable =
        PipeSramAddressTableInfo{unit.resourceColor * kPipeAddressWordBytes};
    for (Operation *transferCreateOp : unit.transferCreateOps) {
      info.resources[transferCreateOp] = pipeResource;
    }
    for (auto [transferCreateOp, recordIndex] : unit.selectedTransferRows) {
      SmallVector<PipeResourceInfo> &resources =
          info.selectedResources[transferCreateOp];
      if (resources.empty()) {
        FailureOr<PipeReference> pipeRef = getPipeReference(
            transferCreateOp,
            cast<PipeTransferCreateOp>(transferCreateOp).getPipe());
        assert(succeeded(pipeRef) && "pipe transfer create verifier failed");
        resources.resize((*pipeRef).getRecords().getPipes().size());
      }
      assert(recordIndex < resources.size() && "selected record index invalid");
      resources[recordIndex] = pipeResource;
    }
  }

  info.sramScratch.bytes =
      maxAddressTableBytes == 0
          ? 0
          : alignTo(maxAddressTableBytes, kPipeSramScratchAlignmentBytes);
  return success();
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
  for (const auto &[transferCreateOp, resource] : info.resources) {
    (void)transferCreateOp;
    if (auto *localCounter =
            std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
      observe(localCounter->senderReadySemIdx);
    }
  }
  for (const auto &[transferCreateOp, resources] : info.selectedResources) {
    (void)transferCreateOp;
    for (const PipeResourceInfo &resource : resources) {
      if (auto *localCounter =
              std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
        observe(localCounter->senderReadySemIdx);
      }
    }
  }
  return highestSemaphoreIdx + 1;
}

int64_t getRequiredPipeGlobalSemaphoreCount(const PipeResourcePlan &info) {
  int64_t highestGlobalSemaphoreIndex = -1;
  for (const auto &[transferCreateOp, resource] : info.resources) {
    (void)transferCreateOp;
    if (auto *globalCounter =
            std::get_if<PipeGlobalReadyCounterInfo>(&resource.readyCounter)) {
      highestGlobalSemaphoreIndex = std::max(
          highestGlobalSemaphoreIndex, globalCounter->globalSemaphoreIndex);
    }
  }
  for (const auto &[transferCreateOp, resources] : info.selectedResources) {
    (void)transferCreateOp;
    for (const PipeResourceInfo &resource : resources) {
      if (auto *globalCounter =
              std::get_if<PipeGlobalReadyCounterInfo>(&resource.readyCounter)) {
        highestGlobalSemaphoreIndex = std::max(
            highestGlobalSemaphoreIndex, globalCounter->globalSemaphoreIndex);
      }
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
  for (const auto &[transferCreateOp, resource] : info.resources) {
    (void)transferCreateOp;
    if (auto *localCounter =
            std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
      observe(localCounter->senderReadySemIdx, ResourceKind::SenderReady,
              resource.pipe);
    }
  }
  for (const auto &[transferCreateOp, resources] : info.selectedResources) {
    (void)transferCreateOp;
    for (const PipeResourceInfo &resource : resources) {
      if (auto *localCounter =
              std::get_if<PipeLocalReadyCounterInfo>(&resource.readyCounter)) {
        observe(localCounter->senderReadySemIdx, ResourceKind::SenderReady,
                resource.pipe);
      }
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
