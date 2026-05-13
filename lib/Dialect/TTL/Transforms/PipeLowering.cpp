// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeLowering.h"
#include "PipeGraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir::tt::ttl {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

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

static int64_t getSenderSemIdx(PipeType pipeType) {
  return pipeType.getPipeNetId() * 2;
}

static int64_t getReceiverSemIdx(PipeType pipeType) {
  return pipeType.getPipeNetId() * 2 + 1;
}

/// Lower CB -> Pipe copy: multicast tiles from source CB to destination cores.
/// For gather patterns, uses receiver's CB address from PipeGraph.
/// After multicast, signals destinations via semaphore.
///
/// Parameters:
/// - receiverInfo: If non-null, contains the receiver's CB index and runtime
///   arg index for the gather pattern. The receiver's CB address is loaded from
///   runtime args to ensure data lands at the correct L1 address on the
///   destination core (which may differ from the sender's CB address).
LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                            const ReceiverCBInfo *receiverInfo,
                            bool isConsumerCB,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = llvm::cast<PipeType>(pipe.getType());

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

  // Multicast handshake: wait for all receivers to signal ready before sending.
  // Each receiver increments the sender's semaphore after reserving CB space.
  // For loopback, the sender core skips the receiver handshake, so we wait
  // for numDests - 1 (remote receivers only).
  if (pipeType.isMulticast()) {
    int64_t expectedSignals =
        pipeType.srcInDstRange() ? numDests - 1 : numDests;
    auto senderSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, getSenderSemIdx(pipeType));
    auto senderSemAddr =
        ttk::GetSemaphoreOp::create(rewriter, loc, senderSemIdx);
    auto senderSemPtr =
        ttk::CastToL1PtrOp::create(rewriter, loc, senderSemAddr);
    auto expectedVal = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(expectedSignals));
    ttk::SemaphoreWaitOp::create(rewriter, loc, senderSemPtr, expectedVal);
    auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    ttk::NocSemaphoreSetOp::create(rewriter, loc, senderSemPtr, zeroIdx);
  }

  SmallVector<int64_t> cbBounds(cbShape.begin(), cbShape.end());
  int64_t cbNumTiles = 1;
  for (int64_t d : cbBounds) {
    cbNumTiles *= d;
  }
  auto numTilesI32 = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(cbNumTiles));

  // Destination DFB on the sender's local view: same as the source when both
  // endpoints share a DFB index, otherwise a fresh handle on the receiver's
  // index so the sender can advance its own fifo_wr_ptr.
  std::optional<int64_t> senderCBIndex = getCBIndex(srcCB);
  Value senderRecvCB = *cbConverted;
  if (receiverInfo && senderCBIndex.has_value() &&
      senderCBIndex.value() != receiverInfo->cbIndex) {
    auto srcCBType = mlir::cast<ttk::CBType>(cbConverted->getType());
    senderRecvCB = ttk::GetCompileArgValOp::create(
        rewriter, loc, srcCBType, static_cast<int32_t>(receiverInfo->cbIndex));
  }

  // In loopback the user's receive callback runs on the sender core and
  // already issues reserve_back / push_back on the destination DFB; emitting
  // the sender-side pair would double-advance regardless of whether the
  // source and destination DFB indices coincide.
  bool skipSenderReserve = pipeType.srcInDstRange();

  if (!skipSenderReserve) {
    ttk::CBReserveBackOp::create(rewriter, loc, senderRecvCB, numTilesI32);
  }

  // Sender's local write_ptr is advanced in lockstep with the receiver via
  // the surrounding reserve_back / push_back.
  auto cbWritePtr = ttk::GetWritePtrOp::create(rewriter, loc, senderRecvCB);
  Value dstBaseIdx =
      arith::IndexCastOp::create(rewriter, loc, indexTy, cbWritePtr);

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

  // Destination coordinates for multicast - convert logical to virtual coords
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

  // For gather patterns (multiple sources to one destination), each source
  // writes to a different slot in the destination CB to avoid overwrites.
  // Slot indices are assigned by PipeGraph based on actual destination sharing.
  int64_t slotIdx = receiverInfo ? receiverInfo->gatherSlotIdx : 0;
  int64_t slotByteOffset = slotIdx * pageSizeBytes * cbNumTiles;

  // Transfer the entire block in a single NOC write. Tiles are contiguous in
  // the CB, and destination CB layout is uniform across cores, so we can send
  // all tiles at once instead of one per tile.
  int64_t totalSizeBytes = cbNumTiles * pageSizeBytes;
  auto totalSizeVal = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(totalSizeBytes));

  Value srcAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, srcPtrIdx);

  Value dstAddrIdx = dstBaseIdx;
  if (slotByteOffset > 0) {
    auto slotOffsetIdx =
        arith::ConstantIndexOp::create(rewriter, loc, slotByteOffset);
    dstAddrIdx =
        arith::AddIOp::create(rewriter, loc, dstAddrIdx, slotOffsetIdx);
  }
  Value dstAddr = arith::IndexCastOp::create(rewriter, loc, i32Ty, dstAddrIdx);

  if (pipeType.isUnicast()) {
    auto nocAddr = ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal,
                                             dstStartYVal, dstAddr);
    ttk::NocAsyncWriteOp::create(rewriter, loc, srcAddr, nocAddr.getResult(),
                                 totalSizeVal);
  } else {
    auto mcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
        dstAddr, nocVal);
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
    auto semIdx = arith::ConstantIndexOp::create(rewriter, loc,
                                                 getSenderSemIdx(pipeType));
    auto semAddr = ttk::GetSemaphoreOp::create(rewriter, loc, semIdx);
    auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto dstSemNocAddr = ttk::GetNocAddrOp::create(rewriter, loc, dstStartXVal,
                                                   dstStartYVal, semAddr);
    ttk::NocSemaphoreIncOp::create(rewriter, loc, dstSemNocAddr.getResult(),
                                   incrVal, /*noc_id=*/Value(),
                                   /*posted=*/BoolAttr());
  } else {
    // Multicast: signal all receivers by setting receiver_sem = VALID (1).
    auto recvSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, getReceiverSemIdx(pipeType));
    auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
    auto recvSemPtr = ttk::CastToL1PtrOp::create(rewriter, loc, recvSemAddr);
    auto validVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
    ttk::NocSemaphoreSetOp::create(rewriter, loc, recvSemPtr, validVal);

    auto recvSemMcastAddr = ttk::ExperimentalGetNocMulticastAddrOp::create(
        rewriter, loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
        recvSemAddr, nocVal);

    if (pipeType.srcInDstRange()) {
      ttk::NocSemaphoreSetMulticastLoopbackOp::create(
          rewriter, loc, recvSemAddr, recvSemMcastAddr.getResult(), numDestsVal,
          /*linked=*/rewriter.getBoolAttr(false));
    } else {
      ttk::NocSemaphoreSetMulticastOp::create(
          rewriter, loc, recvSemAddr, recvSemMcastAddr.getResult(), numDestsVal,
          /*linked=*/nullptr, /*multicast_path_reserve=*/nullptr);
    }
  }

  if (!skipSenderReserve) {
    ttk::CBPushBackOp::create(rewriter, loc, senderRecvCB, numTilesI32);
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Lower Pipe -> CB copy: destination side of pipe transfer.
/// At the destination, data arrives via multicast/unicast from source core.
///
/// For unicast: waits for sender's atomic increment signal.
/// For multicast: performs handshake -- signals sender "ready", then waits
/// for sender to set receiver_sem = VALID. On the sender core (loopback),
/// the handshake is skipped since data is already in the CB from the
/// DRAM read in if_src.
LogicalResult lowerPipeToCB(CopyOp op, Value pipe, Value dstCB,
                            const PipeGraph *pipeGraph,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = llvm::cast<PipeType>(pipe.getType());
  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  if (pipeType.isUnicast()) {
    // Point-to-point: wait for sender's atomic increment.
    // For gather (N senders to 1 receiver), use cumulative waits:
    // 1st recv waits for sem >= 1, 2nd for >= 2, etc. Only reset after last.
    int64_t waitVal = 1;
    bool resetAfterWait = true;
    if (pipeGraph) {
      auto [recvIdx, total] =
          pipeGraph->getGatherRecvProgress(op.getOperation());
      waitVal = recvIdx;
      resetAfterWait = (recvIdx == total);
    }
    auto semIdx = arith::ConstantIndexOp::create(rewriter, loc,
                                                 getSenderSemIdx(pipeType));
    auto semAddr = ttk::GetSemaphoreOp::create(rewriter, loc, semIdx);
    auto semPtr = ttk::CastToL1PtrOp::create(rewriter, loc, semAddr);
    auto waitValConst = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(waitVal));
    ttk::SemaphoreWaitMinOp::create(rewriter, loc, semPtr, waitValConst);
    if (resetAfterWait) {
      auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
      ttk::NocSemaphoreSetOp::create(rewriter, loc, semPtr, zeroIdx);
    }
  } else {
    // Multicast handshake: signal sender "ready", wait for data.
    // For loopback, skip on the sender core (data already in CB).
    auto recvSemIdx = arith::ConstantIndexOp::create(
        rewriter, loc, getReceiverSemIdx(pipeType));
    auto recvSemAddr = ttk::GetSemaphoreOp::create(rewriter, loc, recvSemIdx);
    auto recvSemPtr = ttk::CastToL1PtrOp::create(rewriter, loc, recvSemAddr);

    // Build the handshake body as a lambda to avoid duplication.
    auto emitHandshake = [&]() {
      // Reset receiver_sem to 0 (prepare for sender's VALID signal).
      auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
      ttk::NocSemaphoreSetOp::create(rewriter, loc, recvSemPtr, zeroIdx);

      // Signal sender that this receiver is ready (atomic inc).
      auto senderSemIdx = arith::ConstantIndexOp::create(
          rewriter, loc, getSenderSemIdx(pipeType));
      auto senderSemAddr =
          ttk::GetSemaphoreOp::create(rewriter, loc, senderSemIdx);
      auto srcXLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
      auto srcYLogical =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
      auto srcXTranslated = ttk::ConvertLogicalXToTranslatedOp::create(
          rewriter, loc, indexTy, srcXLogical);
      auto srcYTranslated = ttk::ConvertLogicalYToTranslatedOp::create(
          rewriter, loc, indexTy, srcYLogical);
      auto senderSemNocAddr = ttk::GetNocAddrOp::create(
          rewriter, loc, srcXTranslated, srcYTranslated, senderSemAddr);
      auto incrVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
      ttk::NocSemaphoreIncOp::create(rewriter, loc,
                                     senderSemNocAddr.getResult(), incrVal,
                                     /*noc_id=*/Value(),
                                     /*posted=*/BoolAttr());

      // Wait for sender to set receiver_sem = VALID (1).
      auto validVal = arith::ConstantOp::create(rewriter, loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
      ttk::SemaphoreWaitOp::create(rewriter, loc, recvSemPtr, validVal);
    };

    if (pipeType.srcInDstRange()) {
      // Loopback: skip handshake on the sender core. The sender already
      // has data in its CB from the DRAM read, and will set receiver_sem
      // via loopback multicast.
      auto myX = ttk::MyLogicalXOp::create(rewriter, loc, indexTy);
      auto myY = ttk::MyLogicalYOp::create(rewriter, loc, indexTy);
      auto srcXConst =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcX());
      auto srcYConst =
          arith::ConstantIndexOp::create(rewriter, loc, pipeType.getSrcY());
      auto xNeq = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        myX, srcXConst);
      auto yNeq = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        myY, srcYConst);
      auto notSender = arith::OrIOp::create(rewriter, loc, xNeq, yNeq);
      auto ifOp =
          scf::IfOp::create(rewriter, loc, /*resultTypes=*/
                            TypeRange{}, notSender, /*withElseRegion=*/false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      emitHandshake();
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      emitHandshake();
    }
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

//===----------------------------------------------------------------------===//
// Pipe conditional operation lowering patterns
//===----------------------------------------------------------------------===//

namespace {

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

    // Create scf.if with empty body (the builder adds a yield for us).
    auto ifOp =
        scf::IfOp::create(rewriter, loc, isSrc, /*withElseRegion=*/false);

    // Move ops from the original body into the then block (before the yield).
    // Using inlineBlockBefore moves rather than clones, preserving SSA.
    // The original body's `ttl.yield` terminator is dropped — the new
    // scf.if's own yield is what closes the region.
    Block &srcBlock = op.getBody().front();
    Block &thenBlock = ifOp.getThenRegion().front();
    if (Operation *terminator = srcBlock.getTerminator();
        terminator && isa<YieldOp>(terminator)) {
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());

    rewriter.eraseOp(op);
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

    // Create scf.if with empty body (the builder adds a yield for us).
    auto ifOp =
        scf::IfOp::create(rewriter, loc, isDst, /*withElseRegion=*/false);

    // Move ops from the original body into the then block (before the yield).
    // Using inlineBlockBefore moves rather than clones, preserving SSA.
    // The original body's `ttl.yield` terminator is dropped — the new
    // scf.if's own yield is what closes the region.
    Block &srcBlock = op.getBody().front();
    Block &thenBlock = ifOp.getThenRegion().front();
    if (Operation *terminator = srcBlock.getTerminator();
        terminator && isa<YieldOp>(terminator)) {
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());

    rewriter.eraseOp(op);
    return success();
  }
};

// Collect every pipe type whose net id matches `netId`. Walks the parent
// module so it works regardless of whether matching `ttl.create_pipe` ops
// have already been replaced by their unrealized-conversion-cast stand-ins.
static SmallVector<PipeType> collectPipesForNet(Operation *op, int64_t netId) {
  SmallVector<PipeType> result;
  using PipeKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  llvm::SmallSet<PipeKey, 4> seen;
  op->getParentOfType<ModuleOp>().walk([&](Operation *o) {
    for (Type t : o->getResultTypes()) {
      auto pt = dyn_cast<PipeType>(t);
      if (!pt || pt.getPipeNetId() != netId) {
        continue;
      }
      PipeKey key{pt.getSrcX(),      pt.getSrcY(),    pt.getDstStartX(),
                  pt.getDstStartY(), pt.getDstEndX(), pt.getDstEndY()};
      if (seen.insert(key).second) {
        result.push_back(pt);
      }
    }
  });
  return result;
}

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
    llvm::function_ref<Value(OpBuilder &, Location, Value, Value, PipeType)>
        roleBuilder) {
  auto loc = op.getLoc();
  int64_t netId = op.getPipeNetId();
  auto pipes = collectPipesForNet(op, netId);
  if (pipes.empty()) {
    return op->emitError() << op->getName() << " references unknown PipeNet "
                           << netId;
  }
  auto coreX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  auto coreY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value result;
  for (PipeType pt : pipes) {
    Value match = roleBuilder(rewriter, loc, coreX, coreY, pt);
    result = result ? Value(arith::OrIOp::create(rewriter, loc, result, match))
                    : match;
  }
  rewriter.replaceOp(op, result);
  return success();
}

struct IsSrcLowering : OpConversionPattern<IsSrcOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IsSrcOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, buildSrcMatch);
  }
};

struct IsDstLowering : OpConversionPattern<IsDstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IsDstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(op, rewriter, buildDstMatch);
  }
};

struct IsActiveLowering : OpConversionPattern<IsActiveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IsActiveOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerRolePredicate(
        op, rewriter,
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

void populatePipeLoweringPatterns(RewritePatternSet &patterns,
                                  const TypeConverter &typeConverter) {
  patterns.add<IfSrcLowering, IfDstLowering, IsSrcLowering, IsDstLowering,
               IsActiveLowering, CreatePipeLowering>(typeConverter,
                                                     patterns.getContext());
}

} // namespace mlir::tt::ttl
