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

/// Build a stack-allocated index-typed table of the given values at the
/// current insertion point. Generated form is one `memref.alloca` plus N
/// `memref.store`s of `arith.constant` index values. Per-iteration access
/// inside the foreach loop becomes a single `memref.load`, bounding the
/// loop-body code size at O(1) per field instead of an O(N) `arith.select`
/// chain.
static Value buildPipeIndexTable(OpBuilder &b, Location loc,
                                 ArrayRef<int64_t> values) {
  assert(!values.empty());
  auto memrefTy = MemRefType::get(
      {static_cast<int64_t>(values.size())}, b.getIndexType());
  Value table = memref::AllocaOp::create(b, loc, memrefTy);
  for (size_t i = 0; i < values.size(); ++i) {
    Value v = arith::ConstantIndexOp::create(b, loc, values[i]);
    Value idx = arith::ConstantIndexOp::create(b, loc, i);
    memref::StoreOp::create(b, loc, v, table, ValueRange{idx});
  }
  return table;
}

/// Load the per-iteration index value from a table built by
/// `buildPipeIndexTable`.
static Value loadPipeTableEntry(OpBuilder &b, Location loc, Value table,
                                Value iv) {
  return memref::LoadOp::create(b, loc, table, ValueRange{iv});
}

static SmallVector<PipeType>
getForeachPipeTypes(MLIRContext *context, int64_t pipeNetId,
                    ::llvm::ArrayRef<PipeRecordAttr> pipeRecords) {
  SmallVector<PipeType> pipeTypes;
  pipeTypes.reserve(pipeRecords.size());
  for (PipeRecordAttr record : pipeRecords) {
    pipeTypes.push_back(getPipeTypeFromRecord(context, record, pipeNetId));
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
    PipeNetRecordsAttr records = op.getRecords();
    SmallVector<PipeType> pipeTypes = getForeachPipeTypes(
        op.getContext(), records.getPipeNetId(), records.getPipes());
    if (pipeTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "empty PipeNet record list");
    }
    bool isMulticast =
        mlir::cast<PipeRecordAttr>(records.getPipes()[0]).getIsMulticast();

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
    // Build per-field stack tables before entering the loop; each per-iter
    // access becomes one memref.load instead of an N-deep select chain.
    Value srcXTable = buildPipeIndexTable(rewriter, loc, srcXs);
    Value srcYTable = buildPipeIndexTable(rewriter, loc, srcYs);
    Value dstStartXTable = buildPipeIndexTable(rewriter, loc, dstStartXs);
    Value dstStartYTable = buildPipeIndexTable(rewriter, loc, dstStartYs);
    Value dstEndXTable = buildPipeIndexTable(rewriter, loc, dstEndXs);
    Value dstEndYTable = buildPipeIndexTable(rewriter, loc, dstEndYs);
    Value senderSemIdxTable =
        buildPipeIndexTable(rewriter, loc, senderSemIdxs);
    Value mailboxSemIdxTable =
        buildPipeIndexTable(rewriter, loc, mailboxSemIdxs);
    Value numDestsTable = buildPipeIndexTable(rewriter, loc, numDests);
    Value srcInDstRangeTable =
        buildPipeIndexTable(rewriter, loc, srcInDstRanges);

    Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper =
        arith::ConstantIndexOp::create(rewriter, loc, pipeTypes.size());
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value loopIndex = forOp.getInductionVar();
    Value srcX = loadPipeTableEntry(rewriter, loc, srcXTable, loopIndex);
    Value srcY = loadPipeTableEntry(rewriter, loc, srcYTable, loopIndex);
    Value dstStartX =
        loadPipeTableEntry(rewriter, loc, dstStartXTable, loopIndex);
    Value dstStartY =
        loadPipeTableEntry(rewriter, loc, dstStartYTable, loopIndex);
    Value dstEndX =
        loadPipeTableEntry(rewriter, loc, dstEndXTable, loopIndex);
    Value dstEndY =
        loadPipeTableEntry(rewriter, loc, dstEndYTable, loopIndex);
    Value senderSemIdx =
        loadPipeTableEntry(rewriter, loc, senderSemIdxTable, loopIndex);
    Value mailboxSemIdx =
        loadPipeTableEntry(rewriter, loc, mailboxSemIdxTable, loopIndex);
    Value selectedNumDests =
        loadPipeTableEntry(rewriter, loc, numDestsTable, loopIndex);
    Value srcInDstRangeIdx =
        loadPipeTableEntry(rewriter, loc, srcInDstRangeTable, loopIndex);
    Value srcInDstRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, srcInDstRangeIdx, lower);

    auto selectedPipe = SelectPipeSrcOp::create(
        rewriter, loc, SelectedPipeSrcType::get(op.getContext()), srcX, srcY,
        dstStartX, dstStartY, dstEndX, dstEndY, selectedNumDests, senderSemIdx,
        mailboxSemIdx, srcInDstRange, rewriter.getBoolAttr(isMulticast),
        rewriter.getI64IntegerAttr(records.getPipeNetId()));

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
    PipeNetRecordsAttr records = op.getRecords();
    SmallVector<PipeType> pipeTypes = getForeachPipeTypes(
        op.getContext(), records.getPipeNetId(), records.getPipes());
    if (pipeTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "empty PipeNet record list");
    }
    bool isMulticast =
        mlir::cast<PipeRecordAttr>(records.getPipes()[0]).getIsMulticast();

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
    Value srcXTable = buildPipeIndexTable(rewriter, loc, srcXs);
    Value srcYTable = buildPipeIndexTable(rewriter, loc, srcYs);
    Value dstStartXTable = buildPipeIndexTable(rewriter, loc, dstStartXs);
    Value dstStartYTable = buildPipeIndexTable(rewriter, loc, dstStartYs);
    Value dstEndXTable = buildPipeIndexTable(rewriter, loc, dstEndXs);
    Value dstEndYTable = buildPipeIndexTable(rewriter, loc, dstEndYs);
    Value senderSemIdxTable =
        buildPipeIndexTable(rewriter, loc, senderSemIdxs);
    Value mailboxSemIdxTable =
        buildPipeIndexTable(rewriter, loc, mailboxSemIdxs);
    Value numDestsTable = buildPipeIndexTable(rewriter, loc, numDests);
    Value srcInDstRangeTable =
        buildPipeIndexTable(rewriter, loc, srcInDstRanges);

    Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper =
        arith::ConstantIndexOp::create(rewriter, loc, pipeTypes.size());
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value loopIndex = forOp.getInductionVar();
    Value srcX = loadPipeTableEntry(rewriter, loc, srcXTable, loopIndex);
    Value srcY = loadPipeTableEntry(rewriter, loc, srcYTable, loopIndex);
    Value dstStartX =
        loadPipeTableEntry(rewriter, loc, dstStartXTable, loopIndex);
    Value dstStartY =
        loadPipeTableEntry(rewriter, loc, dstStartYTable, loopIndex);
    Value dstEndX =
        loadPipeTableEntry(rewriter, loc, dstEndXTable, loopIndex);
    Value dstEndY =
        loadPipeTableEntry(rewriter, loc, dstEndYTable, loopIndex);
    Value senderSemIdx =
        loadPipeTableEntry(rewriter, loc, senderSemIdxTable, loopIndex);
    Value mailboxSemIdx =
        loadPipeTableEntry(rewriter, loc, mailboxSemIdxTable, loopIndex);
    Value selectedNumDests =
        loadPipeTableEntry(rewriter, loc, numDestsTable, loopIndex);
    Value srcInDstRangeIdx =
        loadPipeTableEntry(rewriter, loc, srcInDstRangeTable, loopIndex);
    Value srcInDstRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, srcInDstRangeIdx, lower);

    auto selectedPipe = SelectPipeDstOp::create(
        rewriter, loc, SelectedPipeDstType::get(op.getContext()), srcX, srcY,
        dstStartX, dstStartY, dstEndX, dstEndY, selectedNumDests, senderSemIdx,
        mailboxSemIdx, srcInDstRange, rewriter.getBoolAttr(isMulticast),
        rewriter.getI64IntegerAttr(records.getPipeNetId()));

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
      PipeNetRecordsAttr records = foreachSrc.getRecords();
      for (PipeRecordAttr record : records.getPipes()) {
        addPipeType(getPipeTypeFromRecord(op->getContext(), record,
                                          records.getPipeNetId()));
      }
      return;
    }
    if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(op)) {
      PipeNetRecordsAttr records = foreachDst.getRecords();
      for (PipeRecordAttr record : records.getPipes()) {
        addPipeType(getPipeTypeFromRecord(op->getContext(), record,
                                          records.getPipeNetId()));
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
    pipeNetIds.insert(foreachOp.getRecords().getPipeNetId());
  }
}

void populatePipeNetForeachLoweringPatterns(
    RewritePatternSet &patterns, const TypeConverter &typeConverter,
    const PipeRuntimeLayout &pipeRuntimeLayout) {
  patterns.add<PipeNetForeachSrcLowering, PipeNetForeachDstLowering>(
      typeConverter, patterns.getContext(), &pipeRuntimeLayout);
}

} // namespace mlir::tt::ttl
