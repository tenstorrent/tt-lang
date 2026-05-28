// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/// \file
/// Per-pipe metadata abstraction shared by static-pipe and
/// selected-pipe (foreach) lowering paths.

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEMETADATA_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEMETADATA_H

#include "PipeLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include <optional>

namespace mlir::tt::ttl {

/// Per-pipe metadata interface used by `emitSendProtocol`,
/// `emitRecvPostProtocol`, and `emitRecvWaitProtocol`. Implementations
/// supply per-pipe coordinates and semaphore indices either as compile-time
/// constants (`StaticPipeMetadata`) or as runtime operands of a
/// `ttl.select_pipe_src` / `ttl.select_pipe_dst` op (`SelectedPipeMetadata`).
///
/// Accessors that return `Value` build the SSA values lazily at the current
/// insertion point of the rewriter. The static impl emits `arith.constant`
/// ops; the selected impl returns existing SSA operands of the select op.
class PipeMetadata {
public:
  virtual ~PipeMetadata() = default;

  /// PipeNet id this pipe belongs to.
  int64_t getPipeNetId() const { return pipeNetId; }
  /// True when the pipe has a single destination (point-to-point).
  bool isUnicast() const { return isUnicast_; }
  /// True when the pipe has more than one destination (rectangular range).
  bool isMulticast() const { return !isUnicast_; }

  /// Whether the source coord falls inside the destination range. Static
  /// impl knows this at compile time; selected impl only at runtime. The
  /// unicast path never reads it.
  virtual std::optional<bool> staticSrcInDstRange() const = 0;
  virtual Value srcInDstRangeI1(OpBuilder &builder, Location loc) const = 0;

  /// Per-call logical-coordinate values, all in `index` type. Returned
  /// values may already exist (selected) or be freshly emitted (static).
  virtual Value srcXLogical(OpBuilder &builder, Location loc) const = 0;
  virtual Value srcYLogical(OpBuilder &builder, Location loc) const = 0;
  virtual Value dstStartXLogical(OpBuilder &builder, Location loc) const = 0;
  virtual Value dstStartYLogical(OpBuilder &builder, Location loc) const = 0;
  virtual Value dstEndXLogical(OpBuilder &builder, Location loc) const = 0;
  virtual Value dstEndYLogical(OpBuilder &builder, Location loc) const = 0;

  /// `i32` value carrying the number of destinations for the multicast
  /// write helpers.
  virtual Value numDestsI32(OpBuilder &builder, Location loc) const = 0;

  /// `i32` value carrying numDests - (srcInDstRange ? 1 : 0), used by the
  /// multicast semaphore-inc. Static folds at build time; selected emits
  /// `arith.subi` + `arith.select`.
  virtual Value numRemoteDestsI32(OpBuilder &builder, Location loc) const = 0;

  /// `i32` result of `ttkernel.get_semaphore` for the sender-ready
  /// semaphore.
  virtual Value senderReadySemAddr(OpBuilder &builder, Location loc) const = 0;

  /// `i32` result of `ttkernel.get_semaphore` for the mailbox slot.
  virtual Value mailboxSemAddr(OpBuilder &builder, Location loc) const = 0;

protected:
  int64_t pipeNetId = 0;
  bool isUnicast_ = false;
};

/// Reads metadata from a static `PipeType` operand and its
/// `PipeChannelLayout` entry. All accessors emit `arith.constant` ops at
/// the current insertion point.
class StaticPipeMetadata : public PipeMetadata {
public:
  StaticPipeMetadata(PipeType type, PipeChannelLayout l)
      : pipeType(type), layout(l), srcInDstRange_(type.srcInDstRange()) {
    pipeNetId = type.getPipeNetId();
    isUnicast_ = type.isUnicast();
  }

  std::optional<bool> staticSrcInDstRange() const override {
    return srcInDstRange_;
  }
  Value srcInDstRangeI1(OpBuilder &b, Location loc) const override {
    return arith::ConstantIntOp::create(b, loc, srcInDstRange_ ? 1 : 0,
                                        /*width=*/1);
  }
  Value srcXLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getSrcX());
  }
  Value srcYLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getSrcY());
  }
  Value dstStartXLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getDstStartX());
  }
  Value dstStartYLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getDstStartY());
  }
  Value dstEndXLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getDstEndX());
  }
  Value dstEndYLogical(OpBuilder &b, Location loc) const override {
    return arith::ConstantIndexOp::create(b, loc, pipeType.getDstEndY());
  }
  Value numDestsI32(OpBuilder &b, Location loc) const override {
    return arith::ConstantOp::create(
        b, loc, b.getI32Type(),
        b.getI32IntegerAttr(pipeType.getNumDests()));
  }
  Value numRemoteDestsI32(OpBuilder &b, Location loc) const override {
    int64_t value =
        srcInDstRange_ ? pipeType.getNumDests() - 1 : pipeType.getNumDests();
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(value));
  }
  Value senderReadySemAddr(OpBuilder &b, Location loc) const override {
    Value idx =
        arith::ConstantIndexOp::create(b, loc, layout.senderReadySemIdx);
    return ttkernel::GetSemaphoreOp::create(b, loc, idx);
  }
  Value mailboxSemAddr(OpBuilder &b, Location loc) const override {
    Value idx =
        arith::ConstantIndexOp::create(b, loc, layout.mailboxSemIdxBase);
    return ttkernel::GetSemaphoreOp::create(b, loc, idx);
  }

private:
  PipeType pipeType;
  PipeChannelLayout layout;
  bool srcInDstRange_;
};

/// Reads metadata from a `ttl.select_pipe_src` / `ttl.select_pipe_dst` op.
/// The two materializer ops have identical operand layout (only the result
/// type differs), so one impl serves both.
class SelectedPipeMetadata : public PipeMetadata {
public:
  explicit SelectedPipeMetadata(SelectPipeSrcOp op)
      : srcXOperand(op.getSrcX()), srcYOperand(op.getSrcY()),
        dstStartXOperand(op.getDstStartX()),
        dstStartYOperand(op.getDstStartY()), dstEndXOperand(op.getDstEndX()),
        dstEndYOperand(op.getDstEndY()), numDestsOperand(op.getNumDests()),
        senderReadySemIdxOperand(op.getSenderReadySemIdx()),
        mailboxSemIdxBaseOperand(op.getMailboxSemIdxBase()),
        srcInDstRangeOperand(op.getSrcInDstRange()) {
    pipeNetId = op.getPipeNetId();
    isUnicast_ = !op.getIsMulticast();
  }
  explicit SelectedPipeMetadata(SelectPipeDstOp op)
      : srcXOperand(op.getSrcX()), srcYOperand(op.getSrcY()),
        dstStartXOperand(op.getDstStartX()),
        dstStartYOperand(op.getDstStartY()), dstEndXOperand(op.getDstEndX()),
        dstEndYOperand(op.getDstEndY()), numDestsOperand(op.getNumDests()),
        senderReadySemIdxOperand(op.getSenderReadySemIdx()),
        mailboxSemIdxBaseOperand(op.getMailboxSemIdxBase()),
        srcInDstRangeOperand(op.getSrcInDstRange()) {
    pipeNetId = op.getPipeNetId();
    isUnicast_ = !op.getIsMulticast();
  }

  std::optional<bool> staticSrcInDstRange() const override {
    return std::nullopt;
  }
  Value srcInDstRangeI1(OpBuilder &, Location) const override {
    return srcInDstRangeOperand;
  }
  Value srcXLogical(OpBuilder &, Location) const override {
    return srcXOperand;
  }
  Value srcYLogical(OpBuilder &, Location) const override {
    return srcYOperand;
  }
  Value dstStartXLogical(OpBuilder &, Location) const override {
    return dstStartXOperand;
  }
  Value dstStartYLogical(OpBuilder &, Location) const override {
    return dstStartYOperand;
  }
  Value dstEndXLogical(OpBuilder &, Location) const override {
    return dstEndXOperand;
  }
  Value dstEndYLogical(OpBuilder &, Location) const override {
    return dstEndYOperand;
  }
  Value numDestsI32(OpBuilder &b, Location loc) const override {
    return arith::IndexCastOp::create(b, loc, b.getI32Type(), numDestsOperand);
  }
  Value numRemoteDestsI32(OpBuilder &b, Location loc) const override {
    Value oneIdx = arith::ConstantIndexOp::create(b, loc, 1);
    Value remoteIfLoopback =
        arith::SubIOp::create(b, loc, numDestsOperand, oneIdx);
    Value remote = arith::SelectOp::create(b, loc, srcInDstRangeOperand,
                                           remoteIfLoopback, numDestsOperand);
    return arith::IndexCastOp::create(b, loc, b.getI32Type(), remote);
  }
  Value senderReadySemAddr(OpBuilder &b, Location loc) const override {
    return ttkernel::GetSemaphoreOp::create(b, loc, senderReadySemIdxOperand);
  }
  Value mailboxSemAddr(OpBuilder &b, Location loc) const override {
    return ttkernel::GetSemaphoreOp::create(b, loc, mailboxSemIdxBaseOperand);
  }

private:
  Value srcXOperand;
  Value srcYOperand;
  Value dstStartXOperand;
  Value dstStartYOperand;
  Value dstEndXOperand;
  Value dstEndYOperand;
  Value numDestsOperand;
  Value senderReadySemIdxOperand;
  Value mailboxSemIdxBaseOperand;
  Value srcInDstRangeOperand;
};

/// Sender-side pipe protocol: write `srcCB` payload to the receiver-published
/// destination address, then signal arrival. Replaces `op` with a zero i32.
LogicalResult emitSendProtocol(CopyOp op, Value srcCB, bool isConsumerCB,
                               const PipeMetadata &meta,
                               ConversionPatternRewriter &rewriter);

/// Receiver-side pipe receive post: publish the receiver DFB slot address to
/// the sender's mailbox and bump the sender-ready semaphore. Replaces `op`
/// with a zero i32 transfer handle.
LogicalResult emitRecvPostProtocol(PipeRecvPostOp op, Value dst,
                                   const PipeMetadata &meta,
                                   const PipeRuntimeLayout *runtime,
                                   ConversionPatternRewriter &rewriter);

/// Receiver-side pipe receive completion: cumulative `semaphore_wait_min`
/// against the per-PipeNet counter. Erases `op`.
LogicalResult emitRecvWaitProtocol(PipeRecvWaitOp op, const PipeMetadata &meta,
                                   const PipeNetCounterMap *counters,
                                   ConversionPatternRewriter &rewriter);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEMETADATA_H
