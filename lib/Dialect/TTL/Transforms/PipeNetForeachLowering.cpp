// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeNetForeachLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/SmallSet.h"

#include <tuple>

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

namespace {

static CircularBufferType getTTLCBType(Value dataflowBuffer) {
  if (auto dataflowBufferType =
          mlir::dyn_cast<CircularBufferType>(dataflowBuffer.getType())) {
    return dataflowBufferType;
  }
  if (auto castOp =
          dataflowBuffer.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      if (auto dataflowBufferType = mlir::dyn_cast<CircularBufferType>(
              castOp.getInputs()[0].getType())) {
        return dataflowBufferType;
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

static FailureOr<PipeChannelLayout>
lookupPipeChannelLayout(Operation *op, PipeType pipeType,
                        const PipeRuntimeLayout *pipeRuntimeLayout) {
  if (!pipeRuntimeLayout) {
    return op->emitError("internal compiler error: missing pipe runtime "
                         "layout");
  }
  auto it = pipeRuntimeLayout->channels.find(getPipeKey(pipeType));
  if (it == pipeRuntimeLayout->channels.end()) {
    return op->emitError("internal compiler error: pipe missing from runtime "
                         "layout");
  }
  return it->second;
}

static FailureOr<SelectPipeDstOp> getSelectedDestinationPipe(Operation *op,
                                                             Value pipe) {
  auto selected = pipe.getDefiningOp<SelectPipeDstOp>();
  if (!selected) {
    return op->emitError() << "destination-selected pipe must be materialized "
                              "by ttl.select_pipe_dst";
  }
  return selected;
}

struct DataflowBufferTransferPayload {
  Value sourceAddress;
  Value totalSizeBytes;
};

struct SelectedPipeRouteValues {
  Value dstStartX;
  Value dstStartY;
  Value dstEndX;
  Value dstEndY;
  Value dstAddress;
};

static Value buildOptionalNocId(Operation *op, Location loc,
                                ConversionPatternRewriter &rewriter) {
  int64_t nocIdx = getNocIndex(op);
  if (nocIdx == 0) {
    return Value();
  }
  return arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                   rewriter.getI8IntegerAttr(nocIdx));
}

static LogicalResult
buildDataflowBufferTransferPayload(CopyOp op, Value sourceDFB,
                                   bool isConsumerDFB,
                                   ConversionPatternRewriter &rewriter,
                                   DataflowBufferTransferPayload &payload) {
  auto loc = op.getLoc();

  auto sourceDFBConverted =
      utils::convertTTLCBToTTKernel(sourceDFB, rewriter, loc);
  if (failed(sourceDFBConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert DFB operand");
  }

  auto sourceDFBType = getTTLCBType(sourceDFB);
  if (!sourceDFBType) {
    return rewriter.notifyMatchFailure(op, "failed to get DFB type");
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>(sourceDFBType.getElementType());
  if (!tileType) {
    return rewriter.notifyMatchFailure(op, "DFB element type must be tile");
  }

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();
  Value sourcePtrIdx;
  if (isConsumerDFB) {
    auto readPtr =
        ttk::GetReadPtrOp::create(rewriter, loc, *sourceDFBConverted);
    sourcePtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, readPtr);
  } else {
    auto writePtr =
        ttk::GetWritePtrOp::create(rewriter, loc, *sourceDFBConverted);
    sourcePtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, writePtr);
  }

  int64_t numTiles = 1;
  for (int64_t dim : sourceDFBType.getShape()) {
    numTiles *= dim;
  }
  payload.sourceAddress =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, sourcePtrIdx);
  payload.totalSizeBytes = arith::ConstantOp::create(
      rewriter, loc, i32Ty,
      rewriter.getI32IntegerAttr(numTiles * tileType.getSizeBytes()));
  return success();
}

static void
emitSelectedPipeSenderReadyWait(SelectPipeSrcOp selectedPipe, Location loc,
                                ConversionPatternRewriter &rewriter) {
  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);
  auto senderSemAddr = ttk::GetSemaphoreOp::create(
      rewriter, loc, selectedPipe.getSenderReadySemIdx());
  auto senderSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, senderSemAddr);
  Value expectedVal;
  if (selectedPipe.getIsMulticast()) {
    expectedVal = arith::IndexCastOp::create(rewriter, loc, i32Ty,
                                             selectedPipe.getNumDests());
  } else {
    expectedVal = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
  }
  ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedVal);
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);
}

static SelectedPipeRouteValues
buildSelectedPipeRouteValues(SelectPipeSrcOp selectedPipe, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  SelectedPipeRouteValues route;
  route.dstStartX = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getDstStartX());
  route.dstStartY = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getDstStartY());
  route.dstEndX = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getDstEndX());
  route.dstEndY = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getDstEndY());

  auto zeroI32 = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
  auto mailboxSemAddr = ttk::GetSemaphoreOp::create(
      rewriter, loc, selectedPipe.getMailboxSemIdxBase());
  auto mailboxPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, mailboxSemAddr);
  route.dstAddress =
      ttk::LoadFromL1Op::create(rewriter, loc, i32Ty, mailboxPtr, zeroI32);
  return route;
}

static void emitSelectedPipeDataWrite(
    SelectPipeSrcOp selectedPipe, const SelectedPipeRouteValues &route,
    const DataflowBufferTransferPayload &payload, Value nocVal, Location loc,
    ConversionPatternRewriter &rewriter) {
  auto i32Ty = rewriter.getI32Type();
  if (!selectedPipe.getIsMulticast()) {
    auto nocAddr = ttk::GetNocAddrOp::create(rewriter, loc, route.dstStartX,
                                             route.dstStartY, route.dstAddress);
    ttk::NocAsyncWriteOp::create(rewriter, loc, payload.sourceAddress,
                                 nocAddr.getResult(), payload.totalSizeBytes);
    return;
  }

  auto numDestsVal = arith::IndexCastOp::create(rewriter, loc, i32Ty,
                                                selectedPipe.getNumDests());
  auto mcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
      rewriter, loc, route.dstStartX, route.dstStartY, route.dstEndX,
      route.dstEndY, route.dstAddress, nocVal);
  auto writeIf =
      scf::IfOp::create(rewriter, loc, selectedPipe.getSrcInDstRange(),
                        /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&writeIf.getThenRegion().front());
  ttk::NocAsyncWriteMulticastLoopbackSrcOp::create(
      rewriter, loc, payload.sourceAddress, mcastAddr.getResult(),
      payload.totalSizeBytes, numDestsVal, /*linked=*/nullptr,
      /*multicast_path_reserve=*/nullptr, nocVal);
  rewriter.setInsertionPointToStart(&writeIf.getElseRegion().front());
  ttk::NocAsyncWriteMulticastOp::create(
      rewriter, loc, payload.sourceAddress, mcastAddr.getResult(),
      payload.totalSizeBytes, numDestsVal, /*linked=*/nullptr,
      /*multicast_path_reserve=*/nullptr, nocVal);
  rewriter.setInsertionPointAfter(writeIf);
}

static void emitSelectedPipeArrivalSignal(SelectPipeSrcOp selectedPipe,
                                          const SelectedPipeRouteValues &route,
                                          Value nocVal, Location loc,
                                          ConversionPatternRewriter &rewriter) {
  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();
  auto recvSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, getReceiverSemIdx(selectedPipe.getPipeNetId()));
  auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
  auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
  if (!selectedPipe.getIsMulticast()) {
    auto dstSemNocAddr = ttk::GetNocAddrOp::create(
        rewriter, loc, route.dstStartX, route.dstStartY, recvSemAddr);
    ttk::NocSemaphoreIncOp::create(rewriter, loc, dstSemNocAddr.getResult(),
                                   incrVal, /*noc_id=*/Value(),
                                   /*posted=*/BoolAttr());
    return;
  }

  auto oneIdx = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto remoteDestsIfLoopback =
      arith::SubIOp::create(rewriter, loc, selectedPipe.getNumDests(), oneIdx);
  auto numRemoteDests = arith::SelectOp::create(
      rewriter, loc, selectedPipe.getSrcInDstRange(), remoteDestsIfLoopback,
      selectedPipe.getNumDests());
  auto numRemoteDestsVal =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, numRemoteDests);
  auto recvSemMcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
      rewriter, loc, route.dstStartX, route.dstStartY, route.dstEndX,
      route.dstEndY, recvSemAddr, nocVal);
  ttk::NocSemaphoreIncMulticastOp::create(
      rewriter, loc, recvSemMcastAddr.getResult(), incrVal, numRemoteDestsVal,
      /*noc_id=*/Value(), /*posted=*/BoolAttr());

  auto selfIncIf =
      scf::IfOp::create(rewriter, loc, selectedPipe.getSrcInDstRange(),
                        /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(&selfIncIf.getThenRegion().front());
  auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getSrcX());
  auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, selectedPipe.getSrcY());
  auto selfRecvSemNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, recvSemAddr);
  ttk::NocSemaphoreIncOp::create(rewriter, loc, selfRecvSemNocAddr.getResult(),
                                 incrVal,
                                 /*noc_id=*/Value(), /*posted=*/BoolAttr());
  rewriter.setInsertionPointAfter(selfIncIf);
  ttk::NocAsyncAtomicBarrierOp::create(rewriter, loc, /*noc_id=*/Value());
}

static Value selectIndexByLoopIndex(RewriterBase &rewriter, Location loc,
                                    Value loopIndex, ArrayRef<int64_t> values) {
  assert(!values.empty());
  Value selected = arith::ConstantIndexOp::create(rewriter, loc, values[0]);
  for (size_t valueIndex = 1; valueIndex < values.size(); ++valueIndex) {
    Value valueIndexConst =
        arith::ConstantIndexOp::create(rewriter, loc, valueIndex);
    Value isSelected = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, loopIndex, valueIndexConst);
    Value candidate =
        arith::ConstantIndexOp::create(rewriter, loc, values[valueIndex]);
    selected =
        arith::SelectOp::create(rewriter, loc, isSelected, candidate, selected);
  }
  return selected;
}

static SmallVector<PipeType> getForeachPipeTypes(MLIRContext *context,
                                                 int64_t pipeNetId,
                                                 ArrayAttr pipeRecords) {
  SmallVector<PipeType> pipeTypes;
  pipeTypes.reserve(pipeRecords.size());
  for (Attribute attr : pipeRecords) {
    pipeTypes.push_back(getPipeTypeFromRecord(
        context, mlir::cast<PipeRecordAttr>(attr), pipeNetId));
  }
  return pipeTypes;
}

template <typename ForeachOp>
static void cloneForeachBody(ForeachOp foreachOp, Value selectedPipe,
                             ConversionPatternRewriter &rewriter) {
  IRMapping mapping;
  Block &sourceBlock = foreachOp.getBody().front();
  mapping.map(sourceBlock.getArgument(0), selectedPipe);
  for (Operation &bodyOp : sourceBlock.without_terminator()) {
    rewriter.clone(bodyOp, mapping);
  }
}

template <typename ForeachOp>
struct PipeNetForeachLoweringBase : OpConversionPattern<ForeachOp> {
  PipeNetForeachLoweringBase(const TypeConverter &typeConverter,
                             MLIRContext *context,
                             const PipeRuntimeLayout *layout)
      : OpConversionPattern<ForeachOp>(typeConverter, context),
        pipeRuntimeLayout(layout) {}
  const PipeRuntimeLayout *pipeRuntimeLayout;
};

struct PipeNetForeachSrcLowering
    : PipeNetForeachLoweringBase<PipeNetForeachSrcOp> {
  using PipeNetForeachLoweringBase::PipeNetForeachLoweringBase;

  LogicalResult
  matchAndRewrite(PipeNetForeachSrcOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<PipeType> pipeTypes =
        getForeachPipeTypes(op.getContext(), op.getPipeNetId(), op.getPipes());
    if (pipeTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "empty PipeNet record list");
    }
    bool isMulticast =
        mlir::cast<PipeRecordAttr>(op.getPipes()[0]).getIsMulticast();

    SmallVector<int64_t> srcXs, srcYs, dstStartXs, dstStartYs, dstEndXs,
        dstEndYs, senderSemIdxs, mailboxSemIdxs, numDests, srcInDstRanges;
    for (PipeType pipeType : pipeTypes) {
      FailureOr<PipeChannelLayout> layout =
          lookupPipeChannelLayout(op, pipeType, pipeRuntimeLayout);
      if (failed(layout)) {
        return failure();
      }
      srcXs.push_back(pipeType.getSrcX());
      srcYs.push_back(pipeType.getSrcY());
      dstStartXs.push_back(pipeType.getDstStartX());
      dstStartYs.push_back(pipeType.getDstStartY());
      dstEndXs.push_back(pipeType.getDstEndX());
      dstEndYs.push_back(pipeType.getDstEndY());
      senderSemIdxs.push_back(layout->senderReadySemIdx);
      mailboxSemIdxs.push_back(layout->mailboxSemIdxBase);
      numDests.push_back(pipeType.getNumDests());
      srcInDstRanges.push_back(pipeType.srcInDstRange() ? 1 : 0);
    }

    Location loc = op.getLoc();
    Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper =
        arith::ConstantIndexOp::create(rewriter, loc, pipeTypes.size());
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value loopIndex = forOp.getInductionVar();
    Value srcX = selectIndexByLoopIndex(rewriter, loc, loopIndex, srcXs);
    Value srcY = selectIndexByLoopIndex(rewriter, loc, loopIndex, srcYs);
    Value dstStartX =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, dstStartXs);
    Value dstStartY =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, dstStartYs);
    Value dstEndX = selectIndexByLoopIndex(rewriter, loc, loopIndex, dstEndXs);
    Value dstEndY = selectIndexByLoopIndex(rewriter, loc, loopIndex, dstEndYs);
    Value senderSemIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, senderSemIdxs);
    Value mailboxSemIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, mailboxSemIdxs);
    Value selectedNumDests =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, numDests);
    Value srcInDstRangeIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, srcInDstRanges);
    Value srcInDstRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, srcInDstRangeIdx, lower);

    auto selectedPipe = SelectPipeSrcOp::create(
        rewriter, loc, SelectedPipeSrcType::get(op.getContext()), srcX, srcY,
        dstStartX, dstStartY, dstEndX, dstEndY, selectedNumDests, senderSemIdx,
        mailboxSemIdx, srcInDstRange, rewriter.getBoolAttr(isMulticast),
        rewriter.getI64IntegerAttr(op.getPipeNetId()));

    auto nodeX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto nodeY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
    Value xMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, nodeX, srcX);
    Value yMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, nodeY, srcY);
    Value isSource = arith::AndIOp::create(rewriter, loc, xMatches, yMatches);

    auto ifOp = scf::IfOp::create(rewriter, loc, isSource,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    cloneForeachBody(op, selectedPipe.getPipe(), rewriter);
    rewriter.setInsertionPointAfter(forOp);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PipeNetForeachDstLowering
    : PipeNetForeachLoweringBase<PipeNetForeachDstOp> {
  using PipeNetForeachLoweringBase::PipeNetForeachLoweringBase;

  LogicalResult
  matchAndRewrite(PipeNetForeachDstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<PipeType> pipeTypes =
        getForeachPipeTypes(op.getContext(), op.getPipeNetId(), op.getPipes());
    if (pipeTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "empty PipeNet record list");
    }
    bool isMulticast =
        mlir::cast<PipeRecordAttr>(op.getPipes()[0]).getIsMulticast();

    SmallVector<int64_t> srcXs, srcYs, dstStartXs, dstStartYs, dstEndXs,
        dstEndYs, senderSemIdxs, mailboxSemIdxs, numDests, srcInDstRanges;
    for (PipeType pipeType : pipeTypes) {
      FailureOr<PipeChannelLayout> layout =
          lookupPipeChannelLayout(op, pipeType, pipeRuntimeLayout);
      if (failed(layout)) {
        return failure();
      }
      srcXs.push_back(pipeType.getSrcX());
      srcYs.push_back(pipeType.getSrcY());
      dstStartXs.push_back(pipeType.getDstStartX());
      dstStartYs.push_back(pipeType.getDstStartY());
      dstEndXs.push_back(pipeType.getDstEndX());
      dstEndYs.push_back(pipeType.getDstEndY());
      senderSemIdxs.push_back(layout->senderReadySemIdx);
      mailboxSemIdxs.push_back(layout->mailboxSemIdxBase);
      numDests.push_back(pipeType.getNumDests());
      srcInDstRanges.push_back(pipeType.srcInDstRange() ? 1 : 0);
    }

    Location loc = op.getLoc();
    Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper =
        arith::ConstantIndexOp::create(rewriter, loc, pipeTypes.size());
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value loopIndex = forOp.getInductionVar();
    Value srcX = selectIndexByLoopIndex(rewriter, loc, loopIndex, srcXs);
    Value srcY = selectIndexByLoopIndex(rewriter, loc, loopIndex, srcYs);
    Value dstStartX =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, dstStartXs);
    Value dstStartY =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, dstStartYs);
    Value dstEndX = selectIndexByLoopIndex(rewriter, loc, loopIndex, dstEndXs);
    Value dstEndY = selectIndexByLoopIndex(rewriter, loc, loopIndex, dstEndYs);
    Value senderSemIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, senderSemIdxs);
    Value mailboxSemIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, mailboxSemIdxs);
    Value selectedNumDests =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, numDests);
    Value srcInDstRangeIdx =
        selectIndexByLoopIndex(rewriter, loc, loopIndex, srcInDstRanges);
    Value srcInDstRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, srcInDstRangeIdx, lower);

    auto selectedPipe = SelectPipeDstOp::create(
        rewriter, loc, SelectedPipeDstType::get(op.getContext()), srcX, srcY,
        dstStartX, dstStartY, dstEndX, dstEndY, selectedNumDests, senderSemIdx,
        mailboxSemIdx, srcInDstRange, rewriter.getBoolAttr(isMulticast),
        rewriter.getI64IntegerAttr(op.getPipeNetId()));

    auto nodeX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    auto nodeY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
    Value xAtStart = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, nodeX, dstStartX);
    Value xAtEnd = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, nodeX, dstEndX);
    Value yAtStart = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, nodeY, dstStartY);
    Value yAtEnd = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, nodeY, dstEndY);
    Value xInRange = arith::AndIOp::create(rewriter, loc, xAtStart, xAtEnd);
    Value yInRange = arith::AndIOp::create(rewriter, loc, yAtStart, yAtEnd);
    Value isDestination =
        arith::AndIOp::create(rewriter, loc, xInRange, yInRange);

    auto ifOp = scf::IfOp::create(rewriter, loc, isDestination,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    cloneForeachBody(op, selectedPipe.getPipe(), rewriter);
    rewriter.setInsertionPointAfter(forOp);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

PipeType getPipeTypeFromRecord(MLIRContext *context, PipeRecordAttr record,
                               int64_t pipeNetId) {
  return PipeType::get(context, record.getSrcX(), record.getSrcY(),
                       record.getDstStartX(), record.getDstStartY(),
                       record.getDstEndX(), record.getDstEndY(), pipeNetId);
}

void addPipeNetForeachRecordsToIndex(ModuleOp mod, PipeNetIndex &index) {
  using PipeCoordinates =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  llvm::DenseMap<int64_t, llvm::SmallSet<PipeCoordinates, 4>> seenPerNet;
  for (const auto &[pipeNetId, pipeTypes] : index) {
    for (PipeType pipeType : pipeTypes) {
      PipeCoordinates coordinates{
          pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY()};
      seenPerNet[pipeNetId].insert(coordinates);
    }
  }

  auto addPipeType = [&](PipeType pipeType) {
    int64_t pipeNetId = pipeType.getPipeNetId();
    PipeCoordinates coordinates{
        pipeType.getSrcX(),      pipeType.getSrcY(),    pipeType.getDstStartX(),
        pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY()};
    if (seenPerNet[pipeNetId].insert(coordinates).second) {
      index[pipeNetId].push_back(pipeType);
    }
  };

  mod.walk([&](Operation *op) {
    if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(op)) {
      for (Attribute attr : foreachSrc.getPipes()) {
        addPipeType(getPipeTypeFromRecord(op->getContext(),
                                          mlir::cast<PipeRecordAttr>(attr),
                                          foreachSrc.getPipeNetId()));
      }
      return;
    }
    if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(op)) {
      for (Attribute attr : foreachDst.getPipes()) {
        addPipeType(getPipeTypeFromRecord(op->getContext(),
                                          mlir::cast<PipeRecordAttr>(attr),
                                          foreachDst.getPipeNetId()));
      }
    }
  });
}

void collectPipeNetForeachReceiveWaitCounterIds(
    PipeRecvWaitOp wait, llvm::SmallSet<int64_t, 4> &pipeNetIds) {
  if (!mlir::isa<SelectedPipeDstType>(wait.getPipe().getType())) {
    return;
  }
  if (auto foreachOp = wait->getParentOfType<PipeNetForeachDstOp>()) {
    pipeNetIds.insert(foreachOp.getPipeNetId());
  }
}

LogicalResult lowerDataflowBufferToSelectedPipe(
    CopyOp op, Value sourceDataflowBuffer, SelectPipeSrcOp selectedPipe,
    bool isConsumerDataflowBuffer, ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();

  DataflowBufferTransferPayload payload;
  if (failed(buildDataflowBufferTransferPayload(op, sourceDataflowBuffer,
                                                isConsumerDataflowBuffer,
                                                rewriter, payload))) {
    return failure();
  }

  Value nocVal = buildOptionalNocId(op, loc, rewriter);
  emitSelectedPipeSenderReadyWait(selectedPipe, loc, rewriter);
  SelectedPipeRouteValues route =
      buildSelectedPipeRouteValues(selectedPipe, loc, rewriter);
  emitSelectedPipeDataWrite(selectedPipe, route, payload, nocVal, loc,
                            rewriter);
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);
  emitSelectedPipeArrivalSignal(selectedPipe, route, nocVal, loc, rewriter);

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerSelectedPipeRecvPost(PipeRecvPostOp op, Value pipe,
                                        Value dst,
                                        const PipeRuntimeLayout *layout,
                                        ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  FailureOr<SelectPipeDstOp> selectedPipe =
      getSelectedDestinationPipe(op, pipe);
  if (failed(selectedPipe)) {
    return failure();
  }
  if (!layout) {
    return op.emitError("internal compiler error: missing pipe runtime layout");
  }
  int64_t nocIdx = getNocIndex(op);
  if (nocIdx >= layout->numMailboxStagingSems) {
    return op.emitError() << "pipe receive post uses NOC thread index "
                          << nocIdx << ", but pipe runtime layout has only "
                          << layout->numMailboxStagingSems
                          << " mailbox staging semaphores";
  }
  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  Value receiverDFB = getAttachedCB(dst);
  if (!receiverDFB) {
    return rewriter.notifyMatchFailure(
        op, "pipe receive destination is not attached to a DFB");
  }
  auto receiverDFBConverted =
      utils::convertTTLCBToTTKernel(receiverDFB, rewriter, loc);
  if (failed(receiverDFBConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert receiver DFB");
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

  Value nocVal;
  if (nocIdx > 0) {
    nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                       rewriter.getI8IntegerAttr(nocIdx));
  }

  auto receiverWritePtr =
      ttk::GetWritePtrOp::create(rewriter, loc, *receiverDFBConverted);
  Value publishedAddress = receiverWritePtr;
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value localTileIndex = zeroIdx;
  Value globalTileIndex =
      utils::addSliceOffset(dst, localTileIndex, rewriter, loc);
  if (globalTileIndex != localTileIndex) {
    auto tileOffsetI32 =
        arith::IndexCastOp::create(rewriter, loc, i32Ty, globalTileIndex);
    auto pageSizeBytes = arith::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(tileType.getSizeBytes()));
    auto byteOffset =
        arith::MulIOp::create(rewriter, loc, tileOffsetI32, pageSizeBytes);
    publishedAddress =
        arith::AddIOp::create(rewriter, loc, receiverWritePtr, byteOffset);
  }

  auto mailboxStagingSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, layout->mailboxStagingSemIdxBase + nocIdx);
  auto mailboxStagingSem =
      ttk::GetSemaphoreOp::create(rewriter, loc, mailboxStagingSemIdx);
  auto mailboxStagingPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, mailboxStagingSem);
  auto zeroI32 = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
  ttk::StoreToL1Op::create(rewriter, loc, publishedAddress, mailboxStagingPtr,
                           zeroI32);
  auto targetMailboxSem = ttk::GetSemaphoreOp::create(
      rewriter, loc, (*selectedPipe).getMailboxSemIdxBase());

  auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
      rewriter, loc, indexTy, (*selectedPipe).getSrcX());
  auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
      rewriter, loc, indexTy, (*selectedPipe).getSrcY());
  auto senderMailboxNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, targetMailboxSem);
  ttk::RemoteSramWriteU32Op::create(rewriter, loc, mailboxStagingSem,
                                    senderMailboxNocAddr.getResult(), nocVal);
  ttk::NocAsyncWriteBarrierOp::create(rewriter, loc);

  auto senderSemAddr = ttk::GetSemaphoreOp::create(
      rewriter, loc, (*selectedPipe).getSenderReadySemIdx());
  auto senderSemNocAddr = ttk::GetNocAddrOp::create(
      rewriter, loc, srcXTranslated, srcYTranslated, senderSemAddr);
  auto readyIncr = arith::ConstantIndexOp::create(rewriter, loc, 1);
  ttk::NocSemaphoreIncOp::create(rewriter, loc, senderSemNocAddr.getResult(),
                                 readyIncr, nocVal, /*posted=*/BoolAttr());

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

LogicalResult lowerSelectedPipeRecvWait(PipeRecvWaitOp op, Value pipe,
                                        const PipeNetCounterMap *counters,
                                        ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  FailureOr<SelectPipeDstOp> selectedPipe =
      getSelectedDestinationPipe(op, pipe);
  if (failed(selectedPipe)) {
    return failure();
  }
  auto i32Ty = rewriter.getI32Type();
  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), 32);

  auto recvSemIdx = arith::ConstantIndexOp::create(
      rewriter, loc, getReceiverSemIdx((*selectedPipe).getPipeNetId()));
  auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
  auto recvSemPtr =
      ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, recvSemAddr);

  Value counter;
  if (counters) {
    auto func = op->getParentOfType<func::FuncOp>();
    auto funcIt = counters->find(func);
    if (funcIt != counters->end()) {
      auto pipeNetIt = funcIt->second.find((*selectedPipe).getPipeNetId());
      if (pipeNetIt != funcIt->second.end()) {
        counter = pipeNetIt->second;
      }
    }
  }
  if (!counter) {
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

void populatePipeNetForeachLoweringPatterns(
    RewritePatternSet &patterns, const TypeConverter &typeConverter,
    const PipeRuntimeLayout &pipeRuntimeLayout) {
  patterns.add<PipeNetForeachSrcLowering, PipeNetForeachDstLowering>(
      typeConverter, patterns.getContext(), &pipeRuntimeLayout);
}

} // namespace mlir::tt::ttl
