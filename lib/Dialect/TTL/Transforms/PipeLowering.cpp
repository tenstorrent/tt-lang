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
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
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
                      ValueOriginAnalysis &analysis);

static DeviceTransferAttr
getDeviceTransfer(PipeTransferCreateOp transferCreate) {
  auto createPipe = transferCreate.getPipe().getDefiningOp<CreatePipeOp>();
  return createPipe ? createPipe.getDeviceTransferAttr() : DeviceTransferAttr();
}

static unsigned addFabricRoute(SmallVectorImpl<FabricRoute> &routes,
                               DeviceRefAttr localDevice,
                               DeviceRefAttr remoteDevice,
                               PipeSourceKey sourceNode) {
  SmallVector<int64_t> sourceCoordinates{sourceNode.srcX, sourceNode.srcY};
  auto route = llvm::find_if(routes, [&](const FabricRoute &existing) {
    return existing.localDevice == localDevice &&
           existing.remoteDevice == remoteDevice;
  });
  if (route != routes.end()) {
    if (!llvm::is_contained(route->sourceNodes, sourceCoordinates)) {
      route->sourceNodes.push_back(sourceCoordinates);
    }
    return static_cast<unsigned>(std::distance(routes.begin(), route));
  }
  routes.push_back(FabricRoute{localDevice, remoteDevice, {sourceCoordinates}});
  return routes.size() - 1;
}

LogicalResult buildFabricRoutePlan(ModuleOp mod, ValueOriginAnalysis &analysis,
                                   FabricRoutePlan &plan) {
  LogicalResult result = success();
  llvm::DenseMap<Operation *, DeviceDomainAttr> domainsByFunction;

  auto recordRoute = [&](PipeTransferSendOp send,
                         PipeTransferCreateOp transferCreate) {
    DeviceTransferAttr transfer = getDeviceTransfer(transferCreate);
    if (!transfer) {
      return;
    }
    DeviceRefAttr destination = transfer.getEdge().getDestination();
    if (!destination) {
      send.emitError(
          "device-range fabric transfers require multicast target lowering");
      result = failure();
      return;
    }

    FuncOp func = send->getParentOfType<FuncOp>();
    DeviceDomainAttr &functionDomain = domainsByFunction[func];
    if (functionDomain && functionDomain != transfer.getDomain()) {
      send.emitError(
          "all device transfers in one kernel must use the same device domain");
      result = failure();
      return;
    }
    functionDomain = transfer.getDomain();

    DeviceRefAttr source = transfer.getEdge().getSource();
    auto pipeType = mlir::cast<PipeType>(transferCreate.getPipe().getType());
    unsigned routeIndex =
        addFabricRoute(plan.routesByFunction[func], source, destination,
                       getPipeSourceKey(pipeType));
    plan.sendRouteIndex[send] = routeIndex;
  };

  mod.walk([&](Operation *operation) {
    if (auto send = mlir::dyn_cast<PipeTransferSendOp>(operation)) {
      FailureOr<PipeTransferCreateOp> maybeTransferCreate =
          getPipeTransferCreate(send.getOperation(), send.getTransfer(),
                                analysis);
      if (failed(maybeTransferCreate)) {
        result = failure();
        return;
      }
      recordRoute(send, *maybeTransferCreate);
    }
  });
  if (failed(result)) {
    return failure();
  }

  Builder builder(mod.getContext());
  for (auto &[func, routes] : plan.routesByFunction) {
    SmallVector<Attribute> routeAttrs;
    routeAttrs.reserve(routes.size());
    for (const FabricRoute &route : routes) {
      SmallVector<Attribute> sourceNodes;
      sourceNodes.reserve(route.sourceNodes.size());
      for (const SmallVector<int64_t> &sourceNode : route.sourceNodes) {
        sourceNodes.push_back(builder.getDenseI64ArrayAttr(sourceNode));
      }
      routeAttrs.push_back(DictionaryAttr::get(
          mod.getContext(),
          {builder.getNamedAttr("local", route.localDevice),
           builder.getNamedAttr("remote", route.remoteDevice),
           builder.getNamedAttr("source_nodes",
                                builder.getArrayAttr(sourceNodes))}));
    }
    func->setAttr(kFabricRoutesAttrName,
                  ArrayAttr::get(mod.getContext(), routeAttrs));
    func->setAttr(kFabricDeviceDomainAttrName, domainsByFunction.lookup(func));
  }
  return success();
}

void initializeFabricRuntime(const FabricRoutePlan &plan,
                             FabricRuntimeMap &runtime) {
  for (const auto &entry : plan.routesByFunction) {
    FuncOp func = entry.first;
    const SmallVector<FabricRoute> &routes = entry.second;
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    Value connectionCountIndex =
        arith::ConstantIndexOp::create(builder, loc, 0);
    Value connectionCount = ttk::GetArgValOp::create(
        builder, loc, builder.getI32Type(), connectionCountIndex);
    Value manager = ttk::CreateRoutingPlaneConnectionManagerOp::create(
        builder, loc,
        ttk::RoutingPlaneConnectionManagerType::get(builder.getContext()));
    Value routeId = ttk::OpenRoutingPlaneConnectionsOp::create(
        builder, loc, builder.getI32Type(), manager, connectionCount,
        builder.getI64IntegerAttr(1 + 2 * routes.size()));
    SmallVector<Value> chipRoutes;
    chipRoutes.reserve(routes.size());
    for (unsigned routeIndex = 0; routeIndex < routes.size(); ++routeIndex) {
      Value chipRouteIndex = arith::ConstantIndexOp::create(
          builder, loc, 1 + routes.size() + routeIndex);
      chipRoutes.push_back(ttk::GetArgValOp::create(
          builder, loc, builder.getI32Type(), chipRouteIndex));
    }
    runtime[func] = FabricRuntimeInfo{manager, routeId, connectionCount,
                                      std::move(chipRoutes)};

    for (Block &block : func.getBody()) {
      auto returnOp = mlir::dyn_cast<func::ReturnOp>(block.getTerminator());
      if (!returnOp) {
        continue;
      }
      builder.setInsertionPoint(returnOp);
      ttk::CloseRoutingPlaneConnectionsOp::create(builder, returnOp.getLoc(),
                                                  manager, connectionCount);
    }
  }
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

static FailureOr<PipeResourceInfo>
lookupPipeResourceInfo(Operation *protocolOp,
                       const PipeResourcePlan &pipeResourcePlan) {
  auto it = pipeResourcePlan.resources.find(protocolOp);
  if (it == pipeResourcePlan.resources.end()) {
    return protocolOp->emitError("pipe transfer has no resource allocation");
  }
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
  auto dfbIndices = func->getAttrOfType<DenseI32ArrayAttr>(
      kPipeComputedAddressDFBIndicesAttrName);
  return dfbIndices ? static_cast<int64_t>(dfbIndices.size()) : 0;
}

/// Pipe kernels receive tensor buffer addresses, computed receiver DFB bases,
/// and then compiler-managed pipe resources as common runtime arguments.
/// [Device 2.0] Keep this as a resource-plan lookup so the final device API
/// lowering can replace common-arg plumbing without changing pipe semantics.
static int64_t getPipeRuntimeCommonArgIndex(Operation *op,
                                            int64_t pipeRuntimeArgIndex) {
  FuncOp func = op->getParentOfType<FuncOp>();
  assert(func && "pipe op is not inside a function");
  return getNumTensorFunctionArgs(func) +
         getNumComputedAddressRuntimeArgs(func) + pipeRuntimeArgIndex;
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

/// Find the slot counter allocated during resource planning. Missing state is
/// a pass-ordering bug because computed-address sends are planned before
/// conversion patterns mutate the IR.
static Value lookupComputedAddressCounter(
    PipeTransferSendOp op, int64_t counterIndex,
    const PipeComputedAddressCounterMap *computedAddressCounters) {
  assert(computedAddressCounters &&
         "computed-address counter allocation must run before pipe lowering");
  FuncOp senderFunc = op->getParentOfType<FuncOp>();
  auto funcIt = computedAddressCounters->find(senderFunc);
  assert(funcIt != computedAddressCounters->end() &&
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
    const PipeComputedAddressCounterMap *computedAddressCounters,
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
  emitReceiverCompletionIncrement(int64_t receiverCompletionSemIdx) = 0;
  virtual void emitCompletionSignalBarrier() = 0;
};

class NocPipeTransportEmitter final : public PipeTransportEmitter {
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
  NocPipeTransportEmitter(Operation *op, PipeType pipeType,
                          ConversionPatternRewriter &rewriter)
      : loc(op->getLoc()), pipeType(pipeType), rewriter(rewriter),
        nocIdx(getNocIndex(op)),
        nocVal(arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                         rewriter.getI8IntegerAttr(nocIdx))) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    TranslatedCore sourceCore = getSourceCore();
    auto byteEnableAll = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(0xF));
    ttk::NocInlineDwWriteOp::create(rewriter, loc, sourceCore.x, sourceCore.y,
                                    senderTableAddress, publishedAddress,
                                    byteEnableAll, nocVal);
    return success();
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

  void preparePayloadWrite() override {
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

  LogicalResult
  emitReceiverCompletionIncrement(int64_t receiverCompletionSemIdx) override {
    auto receiverCompletionCounterSemIdx =
        arith::ConstantIndexOp::create(rewriter, loc, receiverCompletionSemIdx);
    auto receiverCompletionCounterAddr = ttk::GetSemaphoreOp::create(
        rewriter, loc, receiverCompletionCounterSemIdx);
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

  void emitCompletionSignalBarrier() override {
    ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, nocVal);
  }

private:
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

  TranslatedCore getSourceCore() {
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
    auto [dstStartX, dstStartY] = dstStartTranslatedCore;
    auto [dstEndX, dstEndY] =
        buildTranslatedCore(pipeType.getDstEndX(), pipeType.getDstEndY());
    if (nocIdx == 1) {
      std::swap(dstStartX, dstEndX);
      std::swap(dstStartY, dstEndY);
    }
    destinationRange = DestinationRange{dstStartX, dstStartY, dstEndX, dstEndY};
    return *destinationRange;
  }

  Location loc;
  PipeType pipeType;
  ConversionPatternRewriter &rewriter;
  int64_t nocIdx;
  Value nocVal;
  std::optional<TranslatedCore> sourceCore;
  std::optional<TranslatedCore> dstStartCore;
  std::optional<DestinationRange> destinationRange;
};

class FabricPipeTransportEmitter final : public PipeTransportEmitter {
public:
  FabricPipeTransportEmitter(Operation *op, PipeType pipeType,
                             unsigned routeIndex,
                             const FabricRuntimeInfo &runtime,
                             ConversionPatternRewriter &rewriter)
      : op(op), loc(op->getLoc()), pipeType(pipeType), routeIndex(routeIndex),
        runtime(runtime), rewriter(rewriter),
        nocVal(
            arith::ConstantIntOp::create(rewriter, loc, getNocIndex(op), 8)) {}

  LogicalResult emitReceiverAddressPublish(Value senderTableAddress,
                                           Value publishedAddress) override {
    op->emitError("fabric pipes require computed receiver DFB addresses");
    return failure();
  }

  void emitAddressPublishBarrier() override {}

  LogicalResult
  emitSenderReadyIncrement(Value senderReadyCounterAddr) override {
    Value remoteSemaphoreAddress = buildRemoteNocAddress(
        pipeType.getSrcX(), pipeType.getSrcY(), senderReadyCounterAddr);
    ttk::RoutingPlaneAtomicIncOp::create(
        rewriter, loc, runtime.manager, runtime.routeId, buildConnectionIndex(),
        getChipRoute(), remoteSemaphoreAddress,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));
    return success();
  }

  void preparePayloadWrite() override {}

  LogicalResult emitPayloadWrite(Value srcAddr, Value dstAddr,
                                 Value totalSizeBytes) override {
    sourceAddress = srcAddr;
    destinationAddress = dstAddr;
    sizeBytes = totalSizeBytes;
    return success();
  }

  void emitPayloadWriteBarrier() override {}

  LogicalResult
  emitReceiverCompletionIncrement(int64_t receiverCompletionSemIdx) override {
    assert(sourceAddress && destinationAddress && sizeBytes &&
           "fabric payload must be prepared before completion signaling");
    Value remoteDestinationAddress = buildRemoteNocAddress(
        pipeType.getDstStartX(), pipeType.getDstStartY(), destinationAddress);
    Value completionSemaphoreIndex =
        arith::ConstantIndexOp::create(rewriter, loc, receiverCompletionSemIdx);
    Value completionSemaphoreAddress =
        ttk::GetSemaphoreOp::create(rewriter, loc, completionSemaphoreIndex);
    Value remoteCompletionSemaphoreAddress =
        buildRemoteNocAddress(pipeType.getDstStartX(), pipeType.getDstStartY(),
                              completionSemaphoreAddress);
    ttk::RoutingPlaneFusedWriteAtomicIncOp::create(
        rewriter, loc, runtime.manager, runtime.routeId, buildConnectionIndex(),
        getChipRoute(), sourceAddress, sizeBytes, remoteDestinationAddress,
        remoteCompletionSemaphoreAddress,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));
    return success();
  }

  void emitCompletionSignalBarrier() override {}

private:
  struct TranslatedNode {
    Value x;
    Value y;
  };

  TranslatedNode buildTranslatedNode(int64_t logicalX, int64_t logicalY) {
    Value logicalXValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalX);
    Value logicalYValue =
        arith::ConstantIndexOp::create(rewriter, loc, logicalY);
    return {
        ttk::ConvertLogicalXToTranslatedOp::create(
            rewriter, loc, rewriter.getIndexType(), logicalXValue),
        ttk::ConvertLogicalYToTranslatedOp::create(
            rewriter, loc, rewriter.getIndexType(), logicalYValue),
    };
  }

  Value buildRemoteNocAddress(int64_t logicalX, int64_t logicalY,
                              Value l1Address) {
    TranslatedNode node = buildTranslatedNode(logicalX, logicalY);
    return ttk::GetNocAddrOp::create(rewriter, loc, node.x, node.y, l1Address,
                                     nocVal)
        .getResult();
  }

  Value buildConnectionIndex() {
    Value argIndex =
        arith::ConstantIndexOp::create(rewriter, loc, 1 + routeIndex);
    return ttk::GetArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                    argIndex);
  }

  Value getChipRoute() const {
    assert(routeIndex < runtime.chipRoutes.size() &&
           "fabric route must have target routing data");
    return runtime.chipRoutes[routeIndex];
  }

  Operation *op;
  Location loc;
  PipeType pipeType;
  unsigned routeIndex;
  const FabricRuntimeInfo &runtime;
  ConversionPatternRewriter &rewriter;
  Value nocVal;
  Value sourceAddress;
  Value destinationAddress;
  Value sizeBytes;
};

//===----------------------------------------------------------------------===//
// Receiver post sequence counter initialization
//===----------------------------------------------------------------------===//

void initializePipePostSequenceCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeSemaphoreCounterMap &postSequenceCounters) {
  llvm::MapVector<FuncOp, llvm::SmallSetVector<int64_t, 4>> semaphoresByFunc;
  for (const auto &[protocolOp, resource] : pipeResourcePlan.resources) {
    auto postOp = dyn_cast<PipeTransferPostOp>(protocolOp);
    if (!postOp) {
      continue;
    }
    FuncOp func = postOp->getParentOfType<FuncOp>();
    assert(func && "pipe transfer post must be inside a function");
    semaphoresByFunc[func].insert(resource.completion.semaphoreIndex);
  }

  for (auto &[func, semaphoreSet] : semaphoresByFunc) {
    // These entry-block values dominate waits nested in receiver control flow.
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto memrefTy = MemRefType::get({1}, builder.getI32Type());
    auto i32Ty = builder.getI32Type();
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    Value zeroI32 = arith::ConstantOp::create(builder, loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
    auto &countersForFunc = postSequenceCounters[func];
    SmallVector<int64_t> semaphoreIndices(semaphoreSet.begin(),
                                          semaphoreSet.end());
    llvm::sort(semaphoreIndices);
    for (int64_t semaphoreIndex : semaphoreIndices) {
      auto counter = memref::AllocaOp::create(builder, loc, memrefTy);
      memref::StoreOp::create(builder, loc, zeroI32, counter,
                              ValueRange{zeroIdx});
      countersForFunc[semaphoreIndex] = counter.getResult();
    }
  }
}

void initializePipeCapacitySemaphores(
    const PipeCapacityPlan &pipeCapacityPlan,
    PipeSemaphoreCounterMap &senderCapacityCounters) {
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

void initializePipeComputedAddressCounters(
    const PipeResourcePlan &pipeResourcePlan,
    PipeComputedAddressCounterMap &computedAddressCounters) {
  for (const auto &entry :
       pipeResourcePlan.computedAddressCounterInitializations) {
    FuncOp func = entry.first;
    const SmallVector<PipeComputedAddressCounterInitInfo> &initializations =
        entry.second;
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

/// Lower CB -> Pipe copy: write source DFB data to the selected destination
/// address, then signal arrival.
LogicalResult lowerPipeTransferSend(
    PipeTransferSendOp op, Value srcCB, bool isConsumerCB,
    ValueOriginAnalysis &analysis, const PipeResourcePlan &pipeResourcePlan,
    const PipeCapacityPlan *pipeCapacityPlan,
    const PipeSemaphoreCounterMap *senderCapacityCounters,
    const PipeComputedAddressCounterMap *computedAddressCounters,
    const FabricRoutePlan *fabricRoutePlan,
    const FabricRuntimeMap *fabricRuntime,
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
  FailureOr<PipeResourceInfo> maybePipeResource =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan);
  if (failed(maybePipeResource)) {
    return failure();
  }
  PipeResourceInfo pipeResource = *maybePipeResource;
  PipeCompletionInfo completionInfo = pipeResource.completion;
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  std::optional<unsigned> fabricRouteIndex;
  if (fabricRoutePlan) {
    auto fabricRoute = fabricRoutePlan->sendRouteIndex.find(op);
    if (fabricRoute != fabricRoutePlan->sendRouteIndex.end()) {
      fabricRouteIndex = fabricRoute->second;
    }
  }
  bool usesFabric = fabricRouteIndex.has_value();
  if (usesFabric && !pipeResource.addressStorage.usesComputedReceiverDFB()) {
    op.emitError("fabric pipe send requires computed receiver DFB addresses");
    return failure();
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

  ArrayRef<PipeCapacityAcquireInfo> capacityAcquires =
      pipeCapacityPlan ? pipeCapacityPlan->lookupAcquires(op)
                       : ArrayRef<PipeCapacityAcquireInfo>{};
  SmallVector<Value> capacityCounters;
  if (!capacityAcquires.empty()) {
    FuncOp senderFunc = op->getParentOfType<FuncOp>();
    const llvm::MapVector<int64_t, Value> *countersForFunction = nullptr;
    if (senderCapacityCounters) {
      auto funcIt = senderCapacityCounters->find(senderFunc);
      if (funcIt != senderCapacityCounters->end()) {
        countersForFunction = &funcIt->second;
      }
    }
    for (const PipeCapacityAcquireInfo &capacityAcquire : capacityAcquires) {
      Value counter;
      if (countersForFunction) {
        auto counterIt =
            countersForFunction->find(capacityAcquire.semaphoreIndex);
        if (counterIt != countersForFunction->end()) {
          counter = counterIt->second;
        }
      }
      if (!counter) {
        op.emitError("pipe capacity acquire without sender counter; "
                     "initializePipeCapacitySemaphores must run before "
                     "convert-ttl-to-ttkernel");
        return failure();
      }
      capacityCounters.push_back(counter);
    }
  }

  std::unique_ptr<PipeTransportEmitter> transport;
  if (usesFabric) {
    FuncOp func = op->getParentOfType<FuncOp>();
    auto runtimeIt = fabricRuntime->find(func);
    assert(runtimeIt != fabricRuntime->end() &&
           "fabric route must have initialized kernel runtime state");
    transport = std::make_unique<FabricPipeTransportEmitter>(
        op, pipeType, *fabricRouteIndex, runtimeIt->second, rewriter);
  } else {
    transport =
        std::make_unique<NocPipeTransportEmitter>(op, pipeType, rewriter);
  }
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked source DFB type");

  if (!capacityAcquires.empty()) {
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [capacityAcquire, senderCapacityCounter] :
         llvm::zip_equal(capacityAcquires, capacityCounters)) {
      Value capacitySemaphorePtr =
          buildLocalSemaphorePtr(loc, rewriter, capacityAcquire.semaphoreIndex);
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
  } else if (!usesFabric) {
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
  transport->preparePayloadWrite();

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

  if (failed(transport->emitPayloadWrite(srcAddr, dstAddr, totalSizeVal))) {
    return failure();
  }

  // Wait for payload writes to complete before signaling receiver completion.
  // Without this barrier, the receiver may wake up before all data arrives.
  transport->emitPayloadWriteBarrier();

  if (failed(transport->emitReceiverCompletionIncrement(
          completionInfo.semaphoreIndex))) {
    return failure();
  }

  // Completion signals use non-posted atomics. The send ttl.wait lowers to a
  // no-op, so this barrier is the only flush before the kernel exits.
  transport->emitCompletionSignalBarrier();

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerPipeTransferPost(PipeTransferPostOp op, Value dst,
                                    ValueOriginAnalysis &analysis,
                                    const PipeSemaphoreCounterMap &counters,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    const PipeCapacityPlan *pipeCapacityPlan,
                                    const FabricRoutePlan *fabricRoutePlan,
                                    const FabricRuntimeMap *fabricRuntime,
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
  FailureOr<PipeResourceInfo> maybePipeResource =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan);
  if (failed(maybePipeResource)) {
    return failure();
  }
  PipeResourceInfo pipeResource = *maybePipeResource;
  auto func = op->getParentOfType<func::FuncOp>();
  auto functionCounters = counters.find(func);
  if (functionCounters == counters.end()) {
    op.emitError("pipe receive post has no sequence counter allocation");
    return failure();
  }
  auto sequenceCounter =
      functionCounters->second.find(pipeResource.completion.semaphoreIndex);
  if (sequenceCounter == functionCounters->second.end()) {
    op.emitError("pipe receive post has no sequence counter for completion "
                 "semaphore ")
        << pipeResource.completion.semaphoreIndex;
    return failure();
  }

  bool usesFabric = static_cast<bool>(getDeviceTransfer(createOp));

  // Preflight the only fallible validation before emitting any ops, so a match
  // failure leaves no partially-built IR for the conversion driver to roll
  // back.
  bool usesComputedReceiverDFB =
      pipeResource.addressStorage.usesComputedReceiverDFB();
  std::optional<ReceiverPublishedAddressInfo> maybePublishedAddressInfo;
  if (usesFabric && !usesComputedReceiverDFB) {
    op.emitError(
        "fabric pipe receive requires computed receiver DFB addresses");
    return failure();
  }
  if (!usesComputedReceiverDFB) {
    FailureOr<ReceiverPublishedAddressInfo> info =
        getReceiverPublishedAddressInfo(op, dst, rewriter);
    if (failed(info)) {
      return failure();
    }
    maybePublishedAddressInfo = *info;
  }

  std::unique_ptr<PipeTransportEmitter> transport;
  if (!usesFabric) {
    transport =
        std::make_unique<NocPipeTransportEmitter>(op, pipeType, rewriter);
  }

  if (!usesComputedReceiverDFB) {
    AddressTableInfo addressTableInfo = getAddressTableInfo(op, pipeResource);
    Value publishedAddress = buildReceiverPublishedAddress(
        dst, loc, *maybePublishedAddressInfo, rewriter);
    Value tableAddress =
        buildAddressTableAddress(loc, addressTableInfo, rewriter);
    if (failed(transport->emitReceiverAddressPublish(tableAddress,
                                                     publishedAddress))) {
      return failure();
    }
    transport->emitAddressPublishBarrier();
  }

  if (!usesFabric &&
      (!pipeCapacityPlan || !pipeCapacityPlan->usesCapacityProtocol(op))) {
    ReadyCounterAddressInfo readyCounterInfo =
        getReadyCounterAddressInfo(op, pipeResource, pipeResourcePlan);
    Value senderReadyCounterAddr =
        buildReadyCounterAddress(loc, readyCounterInfo, rewriter);
    if (failed(transport->emitSenderReadyIncrement(senderReadyCounterAddr))) {
      return failure();
    }
  }

  auto zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto previousSequence = memref::LoadOp::create(
      rewriter, loc, sequenceCounter->second, ValueRange{zeroIndex});
  auto oneI32 = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(1));
  auto tokenSequence =
      arith::AddIOp::create(rewriter, loc, previousSequence, oneI32);
  memref::StoreOp::create(rewriter, loc, tokenSequence, sequenceCounter->second,
                          ValueRange{zeroIndex});
  rewriter.replaceOp(op, tokenSequence.getResult());
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

  // A receiver DFB pop makes one slot available to its sender. Emit the
  // release immediately after the pop so both execute under the same control
  // flow.
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

/// Lower the receiver completion wait using the posted token's sequence.
LogicalResult lowerPipeTransferWait(PipeTransferWaitOp op, Value tokenSequence,
                                    const PipeResourcePlan &pipeResourcePlan,
                                    ConversionPatternRewriter &rewriter) {
  if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
    rewriter.eraseOp(op);
    return success();
  }
  auto loc = op.getLoc();
  FailureOr<PipeResourceInfo> maybePipeResource =
      lookupPipeResourceInfo(op.getOperation(), pipeResourcePlan);
  if (failed(maybePipeResource)) {
    return failure();
  }
  PipeCompletionInfo completionInfo = maybePipeResource->completion;

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto receiverCompletionCounterSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, completionInfo.semaphoreIndex);
  auto receiverCompletionCounterAddr = ttk::GetSemaphoreOp::create(
      rewriter, loc, receiverCompletionCounterSemIdx);
  // [Device 2.0] Completion waits should consume the allocated completion
  // object directly once device APIs expose typed semaphore waits.
  auto receiverCompletionCounterPtr = ttk::CastToL1PtrOp::create(
      rewriter, loc, l1PtrTy, receiverCompletionCounterAddr);

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
  int64_t resourceColor = 0;

  /// Local semaphore index used to signal this transfer's completion.
  std::optional<int64_t> maybeCompletionSemaphoreIndex;

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
    ModuleOp mod, ValueOriginAnalysis &analysis, const PipeGraph &pipeGraph,
    const DominanceInfo &dominanceInfo,
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
      FailureOr<PipeTransferPostOp> maybePostOp =
          findPipeTransferPostForToken(analysis, waitOp.getToken());
      if (failed(maybePostOp)) {
        waitOp.emitError(
            "pipe transfer wait requires every possible token value to derive "
            "from the same ttl.pipe_transfer.post");
        return WalkResult::interrupt();
      }
      PipeTransferPostOp postOp = *maybePostOp;
      if (!pipeGraph.getPipeTransferNodeForProtocolOp(postOp.getOperation())) {
        staticallyInactiveOps.insert(op);
        return WalkResult::advance();
      }
      waitOpsByPost[postOp.getOperation()].push_back(op);
    }
    return WalkResult::advance();
  });
  if (provenanceWalkResult.wasInterrupted()) {
    return failure();
  }

  units.reserve(pipeGraph.getPipeTransferNodes().size());
  for (const PipeTransferNode &transferNode :
       pipeGraph.getPipeTransferNodes()) {
    auto sendOp = dyn_cast_if_present<PipeTransferSendOp>(transferNode.sendOp);
    if (!sendOp) {
      emitError(transferNode.sendOp ? transferNode.sendOp->getLoc()
                                    : mod.getLoc(),
                "pipe transfer graph node has no send operation");
      return failure();
    }
    FailureOr<PipeTransferCreateOp> createOp = getPipeTransferCreate(
        sendOp.getOperation(), sendOp.getTransfer(), analysis);
    if (failed(createOp)) {
      return failure();
    }

    PipeTransferAllocationUnit unit;
    unit.transferNodeId = transferNode.id;
    unit.sendOp = sendOp.getOperation();
    unit.pipe = transferNode.pipe;
    unit.pipeType = mlir::cast<PipeType>((*createOp).getPipe().getType());
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

/// Return whether two closed destination rectangles share a receiver.
static bool receiverSetsOverlap(const PipeKey &lhs, const PipeKey &rhs) {
  return lhs.dstStartX <= rhs.dstEndX && rhs.dstStartX <= lhs.dstEndX &&
         lhs.dstStartY <= rhs.dstEndY && rhs.dstStartY <= lhs.dstEndY;
}

/// Reuse a semaphore index only across disjoint physical receiver sets.
/// Transfers sharing a receiver need distinct state because either send may
/// complete first.
static int64_t allocateCompletionSemaphore(
    const PipeKey &pipe,
    SmallVectorImpl<SmallVector<PipeKey>> &pipesBySemaphoreIndex) {
  for (auto indexedPipes : llvm::enumerate(pipesBySemaphoreIndex)) {
    bool overlapsAssignedReceiverSet =
        llvm::any_of(indexedPipes.value(), [&](const PipeKey &allocatedPipe) {
          return receiverSetsOverlap(pipe, allocatedPipe);
        });
    if (!overlapsAssignedReceiverSet) {
      indexedPipes.value().push_back(pipe);
      return static_cast<int64_t>(indexedPipes.index());
    }
  }
  pipesBySemaphoreIndex.push_back(SmallVector<PipeKey>{pipe});
  return static_cast<int64_t>(pipesBySemaphoreIndex.size() - 1);
}

static bool usesSenderReadyCounter(const PipeTransferAllocationUnit &unit,
                                   const PipeCapacityPlan *pipeCapacityPlan) {
  if (!pipeCapacityPlan) {
    return true;
  }
  return !pipeCapacityPlan->usesCapacityProtocol(
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

static bool
hasContiguousReceiverSlots(const ReceiverAddressSequenceProof &sequence,
                           int64_t slotSpanBlocks) {
  assert(sequence.recurrence && "materializable sequence needs recurrence");
  const ReceiverAddressRecurrence &recurrence = *sequence.recurrence;
  auto slotFits = [&](int64_t slot) {
    return slot >= 0 && slotSpanBlocks > 0 &&
           slot <= recurrence.blockCount - slotSpanBlocks;
  };
  int64_t period = recurrence.blockCount /
                   std::gcd(recurrence.blockCount, recurrence.repeatStride);
  std::uint64_t occurrenceCount = static_cast<std::uint64_t>(period);
  if (sequence.executionCount) {
    occurrenceCount = std::min(occurrenceCount, *sequence.executionCount);
  }
  int64_t slot = recurrence.initialSlot;
  for (std::uint64_t occurrence = 0; occurrence < occurrenceCount;
       ++occurrence) {
    if (!slotFits(slot)) {
      return false;
    }
    slot = recurrence.repeatStride >= recurrence.blockCount - slot
               ? recurrence.repeatStride - (recurrence.blockCount - slot)
               : slot + recurrence.repeatStride;
  }
  return true;
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
  if (!sequence.recurrence) {
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
  // A reachable slot with `slot + span > blockCount` requires two NOC writes,
  // which current lowering cannot emit.
  if (!hasContiguousReceiverSlots(sequence,
                                  receiverInfo.receiverSlotSpanBlocks)) {
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
  llvm::DenseMap<unsigned, PipeComputedAddressInfo> infoByUnitIndex;
  llvm::MapVector<FuncOp, SmallVector<PipeComputedAddressCounterInitInfo>>
      counterInitializations;
};

static ComputedAddressPlan
buildComputedAddressPlan(ModuleOp mod,
                         MutableArrayRef<PipeTransferAllocationUnit> units,
                         const PipeGraph &pipeGraph, bool updateFunctionAttrs) {
  ComputedAddressPlan plan;

  /// One transfer whose recurrence can be materialized by its sender.
  struct Candidate {
    unsigned unitIndex = 0;
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
    candidates.push_back(Candidate{static_cast<unsigned>(indexedUnit.index()),
                                   *maybeSenderFunc, *maybeComputedAddress});
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
    if (updateFunctionAttrs) {
      func->setAttr(kPipeComputedAddressDFBIndicesAttrName,
                    builder.getDenseI32ArrayAttr(dfbAttrs));
    }
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
    bool canRepeat = !sequence.executionCount || *sequence.executionCount > 1;
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
    llvm::MapVector<PipeSourceKey, llvm::DenseMap<int64_t, int64_t>>, int64_t>
compactColors(const SourceColorMap &colorUsersBySource,
              PredT unitNeedsResource) {
  llvm::MapVector<PipeSourceKey, llvm::DenseMap<int64_t, int64_t>>
      compactedBySource;
  int64_t maxPerSource = 0;
  for (const auto &[sourceKey, colorUsers] : colorUsersBySource) {
    int64_t nextColor = 0;
    llvm::DenseMap<int64_t, int64_t> &compacted = compactedBySource[sourceKey];
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      if (llvm::any_of(indexedColor.value(), unitNeedsResource)) {
        compacted[static_cast<int64_t>(indexedColor.index())] = nextColor++;
      }
    }
    maxPerSource = std::max(maxPerSource, nextColor);
  }
  return {std::move(compactedBySource), maxPerSource};
}

LogicalResult buildPipeResourcePlan(ModuleOp mod, ValueOriginAnalysis &analysis,
                                    const PipeGraph &pipeGraph,
                                    PipeResourcePlan &info,
                                    bool enableComputedAddresses,
                                    const PipeCapacityPlan *pipeCapacityPlan,
                                    bool updateComputedAddressAttrs) {
  DominanceInfo dominanceInfo(mod);
  PostDominanceInfo postDominanceInfo(mod);
  FailureOr<SmallVector<PipeTransferAllocationUnit>> maybeUnits =
      collectPipeTransferAllocationUnits(mod, analysis, pipeGraph,
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
    computedAddressPlan = buildComputedAddressPlan(mod, units, pipeGraph,
                                                   updateComputedAddressAttrs);
  }
  info.computedAddressCounterInitializations =
      computedAddressPlan.counterInitializations;

  SmallVector<SmallVector<PipeKey>> pipesByCompletionSemaphoreIndex;
  for (PipeTransferAllocationUnit &unit : units) {
    unit.maybeCompletionSemaphoreIndex =
        allocateCompletionSemaphore(unit.pipe, pipesByCompletionSemaphoreIndex);
  }
  int64_t firstSourceLocalReadyCounterSemIdx =
      static_cast<int64_t>(pipesByCompletionSemaphoreIndex.size());

  auto [readyColorBySourceColor, maxReadyCountersPerSource] =
      compactColors(colorUsersBySource, [&](unsigned unitIndex) {
        return usesSenderReadyCounter(units[unitIndex], pipeCapacityPlan);
      });

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

  auto [addressColorBySourceColor, maxAddressColorsPerSource] =
      compactColors(colorUsersBySource, [&](unsigned unitIndex) {
        return computedAddressPlan.infoByUnitIndex.find(unitIndex) ==
               computedAddressPlan.infoByUnitIndex.end();
      });
  int64_t maxAddressTableBytes =
      maxAddressColorsPerSource * kPipeAddressWordBytes;

  for (auto indexedUnit : llvm::enumerate(units)) {
    const PipeTransferAllocationUnit &unit = indexedUnit.value();
    assert(unit.maybeCompletionSemaphoreIndex &&
           "pipe transfer is missing a completion semaphore");
    PipeSourceKey sourceKey = getPipeSourceKey(unit.pipeType);
    std::optional<PipeReadyCounterInfo> maybeReadyCounter;
    if (usesSenderReadyCounter(unit, pipeCapacityPlan)) {
      auto sourceIt = readyColorBySourceColor.find(sourceKey);
      assert(sourceIt != readyColorBySourceColor.end());
      auto colorIt = sourceIt->second.find(unit.resourceColor);
      assert(colorIt != sourceIt->second.end());
      int64_t readyColor = colorIt->second;
      maybeReadyCounter = PipeReadyCounterInfo::localSemaphore(
          firstSourceLocalReadyCounterSemIdx + readyColor);
      if (useGlobalReadyCounters) {
        auto globalIt = globalIndexBySourceColor.find(sourceKey);
        assert(globalIt != globalIndexBySourceColor.end());
        assert(readyColor < static_cast<int64_t>(globalIt->second.size()));
        maybeReadyCounter =
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
        PipeCompletionInfo{*unit.maybeCompletionSemaphoreIndex},
        maybeReadyCounter,
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
  for (const auto &[protocolOp, resource] : info.resources) {
    (void)protocolOp;
    observer.observeLocalSemaphore(resource.completion.semaphoreIndex);
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
  for (const auto &[protocolOp, resource] : info.resources) {
    (void)protocolOp;
    if (resource.completion.semaphoreIndex > highest.index) {
      highest = HighestSemaphore{resource.completion.semaphoreIndex,
                                 PipeSemaphoreKind::ReceiverCompletion,
                                 resource.pipe};
    }
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
    note << "receiver-completion counter for ";
    assert(highest.pipe && "receiver-completion resource must have a pipe");
    appendPipe(*highest.pipe);
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
