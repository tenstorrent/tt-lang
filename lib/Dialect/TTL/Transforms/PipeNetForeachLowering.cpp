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
  auto memrefTy =
      MemRefType::get({static_cast<int64_t>(values.size())}, b.getIndexType());
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

/// Per-pipe scalar fields the foreach lowering needs: coords, semaphore
/// indices, num_dests, and srcInDstRange (as 0/1). Kept as parallel
/// SmallVectors so they map directly to one stack table each.
struct PipeForeachFields {
  SmallVector<int64_t> srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY;
  SmallVector<int64_t> senderSemIdx, mailboxSemIdx, numDests, srcInDstRange;

  void append(PipeType pipeType, const PipeChannelLayout &layout) {
    srcX.push_back(pipeType.getSrcX());
    srcY.push_back(pipeType.getSrcY());
    dstStartX.push_back(pipeType.getDstStartX());
    dstStartY.push_back(pipeType.getDstStartY());
    dstEndX.push_back(pipeType.getDstEndX());
    dstEndY.push_back(pipeType.getDstEndY());
    senderSemIdx.push_back(layout.senderReadySemIdx);
    mailboxSemIdx.push_back(layout.mailboxSemIdxBase);
    numDests.push_back(pipeType.getNumDests());
    srcInDstRange.push_back(pipeType.srcInDstRange() ? 1 : 0);
  }
};

/// `scf.for` and the `SelectPipe*Op` it materializes per iteration. Each
/// pattern still owns the role-specific coord predicate and body cloning.
template <typename SelectOp>
struct PipeForeachShell {
  scf::ForOp forOp;
  SelectOp selectedPipe;
};

/// Gather fields, build per-field stack tables, emit the foreach `scf.for`,
/// emit per-iteration loads + the matching `SelectOp`. Insertion point on
/// return is at the start of `forOp.getBody()`, after the SelectOp.
template <typename SelectOp, typename SelectedType>
static FailureOr<PipeForeachShell<SelectOp>>
buildPipeForeachShell(Operation *op, ArrayRef<PipeType> pipeTypes,
                      bool isMulticast, int64_t pipeNetId,
                      const PipeRuntimeLayout *runtime,
                      ConversionPatternRewriter &rewriter) {
  PipeForeachFields f;
  for (PipeType pipeType : pipeTypes) {
    FailureOr<PipeChannelLayout> layout =
        lookupPipeChannelLayout(op, pipeType, runtime);
    if (failed(layout)) {
      return failure();
    }
    f.append(pipeType, *layout);
  }

  Location loc = op->getLoc();
  // Per-field stack tables built once before the loop. Loop body becomes
  // one memref.load per field instead of an N-deep arith.select chain.
  Value srcXT = buildPipeIndexTable(rewriter, loc, f.srcX);
  Value srcYT = buildPipeIndexTable(rewriter, loc, f.srcY);
  Value dstStartXT = buildPipeIndexTable(rewriter, loc, f.dstStartX);
  Value dstStartYT = buildPipeIndexTable(rewriter, loc, f.dstStartY);
  Value dstEndXT = buildPipeIndexTable(rewriter, loc, f.dstEndX);
  Value dstEndYT = buildPipeIndexTable(rewriter, loc, f.dstEndY);
  Value senderSemIdxT = buildPipeIndexTable(rewriter, loc, f.senderSemIdx);
  Value mailboxSemIdxT = buildPipeIndexTable(rewriter, loc, f.mailboxSemIdx);
  Value numDestsT = buildPipeIndexTable(rewriter, loc, f.numDests);
  Value srcInDstRangeT = buildPipeIndexTable(rewriter, loc, f.srcInDstRange);

  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper = arith::ConstantIndexOp::create(rewriter, loc, pipeTypes.size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value srcInDstRangeIdx =
      loadPipeTableEntry(rewriter, loc, srcInDstRangeT, iv);
  Value srcInDstRangeI1 = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ne, srcInDstRangeIdx, lower);
  auto selectedPipe = SelectOp::create(
      rewriter, loc, SelectedType::get(op->getContext()),
      loadPipeTableEntry(rewriter, loc, srcXT, iv),
      loadPipeTableEntry(rewriter, loc, srcYT, iv),
      loadPipeTableEntry(rewriter, loc, dstStartXT, iv),
      loadPipeTableEntry(rewriter, loc, dstStartYT, iv),
      loadPipeTableEntry(rewriter, loc, dstEndXT, iv),
      loadPipeTableEntry(rewriter, loc, dstEndYT, iv),
      loadPipeTableEntry(rewriter, loc, numDestsT, iv),
      loadPipeTableEntry(rewriter, loc, senderSemIdxT, iv),
      loadPipeTableEntry(rewriter, loc, mailboxSemIdxT, iv), srcInDstRangeI1,
      rewriter.getBoolAttr(isMulticast), rewriter.getI64IntegerAttr(pipeNetId));

  return PipeForeachShell<SelectOp>{forOp, selectedPipe};
}

template <typename ForeachOp>
struct PipeNetForeachLoweringBase : OpConversionPattern<ForeachOp> {
  PipeNetForeachLoweringBase(const TypeConverter &typeConverter,
                             MLIRContext *context,
                             const PipeRuntimeLayout *layout)
      : OpConversionPattern<ForeachOp>(typeConverter, context),
        pipeRuntimeLayout(layout) {}
  const PipeRuntimeLayout *pipeRuntimeLayout;

  /// Gather per-pipe types, build the foreach loop shell, and inline the
  /// body under the role-specific coord predicate. `buildPredicate` reads
  /// per-iteration coords off the materialized SelectOp and returns an i1
  /// "execute this iteration on this node" value.
  template <typename SelectOp, typename SelectedType, typename PredicateFn>
  LogicalResult lowerForeach(ForeachOp op, ConversionPatternRewriter &rewriter,
                             PredicateFn &&buildPredicate) const {
    PipeNetRecordsAttr records = op.getRecords();
    SmallVector<PipeType> pipeTypes = getForeachPipeTypes(
        op.getContext(), records.getPipeNetId(), records.getPipes());
    if (pipeTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "empty PipeNet record list");
    }
    bool isMulticast =
        mlir::cast<PipeRecordAttr>(records.getPipes()[0]).getIsMulticast();

    FailureOr<PipeForeachShell<SelectOp>> shellOr =
        buildPipeForeachShell<SelectOp, SelectedType>(
            op, pipeTypes, isMulticast, records.getPipeNetId(),
            this->pipeRuntimeLayout, rewriter);
    if (failed(shellOr)) {
      return failure();
    }
    PipeForeachShell<SelectOp> &shell = *shellOr;

    Location loc = op.getLoc();
    Value predicate = buildPredicate(rewriter, loc, shell.selectedPipe);
    auto ifOp = scf::IfOp::create(rewriter, loc, predicate,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    cloneForeachBody(op, shell.selectedPipe.getPipe(), rewriter);
    rewriter.setInsertionPointAfter(shell.forOp);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PipeNetForeachSrcLowering
    : PipeNetForeachLoweringBase<PipeNetForeachSrcOp> {
  using PipeNetForeachLoweringBase::PipeNetForeachLoweringBase;

  LogicalResult
  matchAndRewrite(PipeNetForeachSrcOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerForeach<SelectPipeSrcOp, SelectedPipeSrcType>(
        op, rewriter,
        [](OpBuilder &b, Location loc, SelectPipeSrcOp selectedPipe) {
          Value nodeX = ttk::MyLogicalXOp::create(b, loc, b.getIndexType());
          Value nodeY = ttk::MyLogicalYOp::create(b, loc, b.getIndexType());
          Value xMatches = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, nodeX, selectedPipe.getSrcX());
          Value yMatches = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, nodeY, selectedPipe.getSrcY());
          return arith::AndIOp::create(b, loc, xMatches, yMatches).getResult();
        });
  }
};

struct PipeNetForeachDstLowering
    : PipeNetForeachLoweringBase<PipeNetForeachDstOp> {
  using PipeNetForeachLoweringBase::PipeNetForeachLoweringBase;

  LogicalResult
  matchAndRewrite(PipeNetForeachDstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerForeach<SelectPipeDstOp, SelectedPipeDstType>(
        op, rewriter,
        [](OpBuilder &b, Location loc, SelectPipeDstOp selectedPipe) {
          Value nodeX = ttk::MyLogicalXOp::create(b, loc, b.getIndexType());
          Value nodeY = ttk::MyLogicalYOp::create(b, loc, b.getIndexType());
          Value xAtStart =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, nodeX,
                                    selectedPipe.getDstStartX());
          Value xAtEnd =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, nodeX,
                                    selectedPipe.getDstEndX());
          Value yAtStart =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, nodeY,
                                    selectedPipe.getDstStartY());
          Value yAtEnd =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, nodeY,
                                    selectedPipe.getDstEndY());
          Value xInRange = arith::AndIOp::create(b, loc, xAtStart, xAtEnd);
          Value yInRange = arith::AndIOp::create(b, loc, yAtStart, yAtEnd);
          return arith::AndIOp::create(b, loc, xInRange, yInRange).getResult();
        });
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
