// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeNetForeachLowering.h"

#include "PipeGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeNetParticipantPlan.h"
#include "ttlang/Dialect/TTL/Transforms/PipeRecordLoweringUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>

namespace mlir::tt::ttl {
namespace {

namespace ttk = mlir::tt::ttkernel;

// Duplicating up to four callback bodies avoids table lookups for small nets.
// Larger nets use one loop so the transfer protocol body is not duplicated for
// every record.
constexpr size_t kPipeNetForeachDirectRecordLimit = 4;

static bool shouldLowerPipeNetForeachDirect(PipeNetRecordsAttr records) {
  return !records.getPipes().front().getDeviceTransfer() &&
         records.getPipes().size() <= kPipeNetForeachDirectRecordLimit;
}

struct GridMajorPipeRecordIndexTables {
  int64_t gridX = 0;
  int64_t gridArea = 0;
  SmallVector<int64_t> edgeOffsetsByDevice;
  SmallVector<int64_t> edgeBlocks;
  SmallVector<int64_t> sourceDeviceIndices;
  SmallVector<int64_t> destinationDeviceIndices;
};

static FailureOr<int64_t> getDeviceCount(DeviceDomainAttr domain) {
  std::uint64_t deviceCount = 1;
  for (DeviceDomainComponentAttr component : domain.getComponents()) {
    for (int64_t extent : component.getExtent().asArrayRef()) {
      if (extent <= 0) {
        return failure();
      }
      std::optional<std::uint64_t> maybeDeviceCount = llvm::checkedMulUnsigned(
          deviceCount, static_cast<std::uint64_t>(extent));
      if (!maybeDeviceCount ||
          *maybeDeviceCount >
              static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
        return failure();
      }
      deviceCount = *maybeDeviceCount;
    }
  }
  return static_cast<int64_t>(deviceCount);
}

static FailureOr<GridMajorPipeRecordIndexTables>
buildGridMajorPipeRecordIndexTables(PipeNetRecordsAttr records, PipeRole role,
                                    Operation *op) {
  DeviceTransferAttr firstTransfer =
      records.getPipes().front().getDeviceTransfer();
  if (!firstTransfer) {
    return failure();
  }
  FailureOr<std::pair<int64_t, int64_t>> launchGrid = getLaunchGrid(op);
  FailureOr<int64_t> deviceCount = getDeviceCount(firstTransfer.getDomain());
  if (failed(launchGrid) || failed(deviceCount)) {
    return failure();
  }
  auto [gridX, gridY] = *launchGrid;
  std::optional<int64_t> maybeGridArea = llvm::checkedMul(gridX, gridY);
  if (!maybeGridArea || records.getPipes().size() % *maybeGridArea != 0) {
    return failure();
  }
  int64_t gridArea = *maybeGridArea;
  if (records.getPipes().size() >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return failure();
  }
  int64_t recordCount = static_cast<int64_t>(records.getPipes().size());
  int64_t edgeCount = recordCount / gridArea;
  if (*deviceCount >= recordCount) {
    return failure();
  }

  SmallVector<SmallVector<int64_t>> edgeBlocksByDevice(
      static_cast<std::size_t>(*deviceCount));
  GridMajorPipeRecordIndexTables tables;
  tables.gridX = gridX;
  tables.gridArea = gridArea;
  tables.sourceDeviceIndices.reserve(edgeCount);
  tables.destinationDeviceIndices.reserve(edgeCount);
  for (int64_t edgeBlock = 0; edgeBlock < edgeCount; ++edgeBlock) {
    int64_t blockStart = edgeBlock * gridArea;
    PipeRecordAttr firstRecord = records.getPipes()[blockStart];
    DeviceTransferAttr transfer = firstRecord.getDeviceTransfer();
    if (!transfer || transfer.getDomain() != firstTransfer.getDomain()) {
      return failure();
    }
    for (int64_t nodeIndex = 0; nodeIndex < gridArea; ++nodeIndex) {
      PipeRecordAttr record = records.getPipes()[blockStart + nodeIndex];
      int64_t expectedX = nodeIndex % gridX;
      int64_t expectedY = nodeIndex / gridX;
      if (record.getDeviceTransfer() != transfer ||
          record.getSrcX() != expectedX || record.getSrcY() != expectedY ||
          record.getDstStartX() != expectedX ||
          record.getDstStartY() != expectedY ||
          record.getDstEndX() != expectedX ||
          record.getDstEndY() != expectedY) {
        return failure();
      }
    }

    int64_t sourceDeviceIndex = getLogicalDeviceIndex(
        transfer.getDomain(), transfer.getEdge().getSource());
    int64_t destinationDeviceIndex = getLogicalDeviceIndex(
        transfer.getDomain(), transfer.getEdge().getDestination());
    if (sourceDeviceIndex < 0 || sourceDeviceIndex >= *deviceCount ||
        destinationDeviceIndex < 0 || destinationDeviceIndex >= *deviceCount) {
      return failure();
    }
    tables.sourceDeviceIndices.push_back(sourceDeviceIndex);
    tables.destinationDeviceIndices.push_back(destinationDeviceIndex);
    int64_t endpointDeviceIndex =
        role == PipeRole::Source ? sourceDeviceIndex : destinationDeviceIndex;
    edgeBlocksByDevice[endpointDeviceIndex].push_back(edgeBlock);
  }

  tables.edgeOffsetsByDevice.push_back(0);
  for (ArrayRef<int64_t> edgeBlocks : edgeBlocksByDevice) {
    tables.edgeBlocks.append(edgeBlocks.begin(), edgeBlocks.end());
    tables.edgeOffsetsByDevice.push_back(
        static_cast<int64_t>(tables.edgeBlocks.size()));
  }
  return tables;
}

template <typename SelectOp, typename SelectedType>
static SelectOp
buildSelectedPipe(OpBuilder &builder, Location loc, PipeNetRecordsAttr records,
                  const PipeRecordTables &tables, Value recordIndex) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value srcInDstRangeIndex = buildConstantIndexTableLookup(
      builder, loc, tables.srcInDstRange, recordIndex);
  Value srcInDstRange = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, srcInDstRangeIndex, zero);
  return SelectOp::create(
      builder, loc, SelectedType::get(builder.getContext()), recordIndex,
      buildConstantIndexTableLookup(builder, loc, tables.srcX, recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.srcY, recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstStartX,
                                    recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstStartY,
                                    recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstEndX, recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstEndY, recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.numDests, recordIndex),
      srcInDstRange,
      buildConstantIndexTableLookup(builder, loc, tables.sourceDeviceIndex,
                                    recordIndex),
      buildConstantIndexTableLookup(builder, loc, tables.destinationDeviceIndex,
                                    recordIndex),
      records);
}

template <typename SelectOp, typename SelectedType>
static SelectOp buildGridMajorSelectedPipe(
    OpBuilder &builder, Location loc, PipeNetRecordsAttr records,
    const GridMajorPipeRecordIndexTables &tables, Value recordIndex,
    Value edgeBlock, Value nodeX, Value nodeY) {
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value sourceDeviceIndex = buildConstantIndexTableLookup(
      builder, loc, tables.sourceDeviceIndices, edgeBlock);
  Value destinationDeviceIndex = buildConstantIndexTableLookup(
      builder, loc, tables.destinationDeviceIndices, edgeBlock);
  Value srcInDstRange = arith::ConstantIntOp::create(builder, loc, 1, 1);
  return SelectOp::create(builder, loc, SelectedType::get(builder.getContext()),
                          recordIndex, nodeX, nodeY, nodeX, nodeY, nodeX, nodeY,
                          one, srcInDstRange, sourceDeviceIndex,
                          destinationDeviceIndex, records);
}

static void collectOutermostPipeNetForeachOps(
    Operation *root, SmallVectorImpl<Operation *> &foreachWorklist) {
  if (mlir::isa<PipeNetForeachSrcOp, PipeNetForeachDstOp>(root)) {
    foreachWorklist.push_back(root);
    return;
  }
  root->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (nestedOp == root ||
        !mlir::isa<PipeNetForeachSrcOp, PipeNetForeachDstOp>(nestedOp)) {
      return WalkResult::advance();
    }
    foreachWorklist.push_back(nestedOp);
    return WalkResult::skip();
  });
}

template <typename ForeachOp>
static void
clonePipeForeachBody(ForeachOp foreachOp, Value selectedPipe,
                     OpBuilder &builder,
                     SmallVectorImpl<Operation *> &foreachWorklist) {
  IRMapping mapping;
  Block &sourceBlock = foreachOp.getBody().front();
  mapping.map(sourceBlock.getArgument(0), selectedPipe);
  for (Operation &bodyOp : sourceBlock) {
    if (mlir::isa<YieldOp>(bodyOp)) {
      continue;
    }
    Operation *clonedOp = builder.clone(bodyOp, mapping);
    collectOutermostPipeNetForeachOps(clonedOp, foreachWorklist);
  }
}

using RecordInductionMap =
    std::map<std::pair<LaunchExecutionLocation, std::uint64_t>, std::uint64_t>;

static RecordInductionMap buildLocalRecordInductionValues(
    const LocalPipeNetParticipantPlan &participantPlan) {
  RecordInductionMap inductionValues;
  for (auto [nodeIndex, recordOffset, recordCount] :
       llvm::enumerate(participantPlan.recordOffsetsByNode,
                       participantPlan.recordCountsByNode)) {
    int64_t nodeX = static_cast<int64_t>(nodeIndex) % participantPlan.gridX;
    int64_t nodeY = static_cast<int64_t>(nodeIndex) / participantPlan.gridX;
    for (int64_t iterationIndex = recordOffset;
         iterationIndex < recordOffset + recordCount; ++iterationIndex) {
      bool inserted =
          inductionValues
              .try_emplace(
                  std::make_pair(
                      LaunchExecutionLocation({nodeX, nodeY}),
                      static_cast<std::uint64_t>(
                          participantPlan.recordIndices[iterationIndex])),
                  static_cast<std::uint64_t>(iterationIndex))
              .second;
      assert(inserted && "participant plan must select each record once");
    }
  }
  return inductionValues;
}

static RecordInductionMap buildGridMajorRecordInductionValues(
    PipeNetRecordsAttr records, PipeRole role,
    const GridMajorPipeRecordIndexTables &tables) {
  RecordInductionMap inductionValues;
  for (auto [iterationIndex, edgeBlock] : llvm::enumerate(tables.edgeBlocks)) {
    int64_t blockStart = edgeBlock * tables.gridArea;
    for (int64_t nodeIndex = 0; nodeIndex < tables.gridArea; ++nodeIndex) {
      int64_t recordIndex = blockStart + nodeIndex;
      DeviceTransferAttr transfer =
          records.getPipes()[recordIndex].getDeviceTransfer();
      DeviceRefAttr endpoint = role == PipeRole::Source
                                   ? transfer.getEdge().getSource()
                                   : transfer.getEdge().getDestination();
      bool inserted =
          inductionValues
              .try_emplace(
                  std::make_pair(
                      LaunchExecutionLocation(
                          {nodeIndex % tables.gridX, nodeIndex / tables.gridX},
                          transfer.getDomain(), endpoint),
                      static_cast<std::uint64_t>(recordIndex)),
                  static_cast<std::uint64_t>(iterationIndex))
              .second;
      assert(inserted && "grid-major plan must select each record once");
    }
  }
  return inductionValues;
}

template <typename ForeachOp, typename SelectOp, typename SelectedPipeType>
static bool
tryLowerLocalPipeNetForeach(ForeachOp op, RewriterBase &rewriter,
                            PipeForeachLoweringInfo &foreachLoweringInfo,
                            PipeRole role,
                            PipeNetRecordSelection recordSelection,
                            SmallVectorImpl<Operation *> &foreachWorklist) {
  PipeNetRecordsAttr records = op.getRecords();
  FailureOr<std::pair<int64_t, int64_t>> launchGrid = getLaunchGrid(op);
  if (failed(launchGrid)) {
    return false;
  }
  auto [gridX, gridY] = *launchGrid;
  FailureOr<LocalPipeNetParticipantPlan> participantPlan =
      buildLocalPipeNetParticipantPlan(records, role, gridX, gridY);
  if (failed(participantPlan)) {
    return false;
  }

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value gridXValue =
      arith::ConstantIndexOp::create(rewriter, loc, participantPlan->gridX);
  Value nodeRowOffset = arith::MulIOp::create(rewriter, loc, nodeY, gridXValue);
  Value nodeIndex = arith::AddIOp::create(rewriter, loc, nodeRowOffset, nodeX);
  Value lower = buildConstantIndexTableLookup(
      rewriter, loc, participantPlan->recordOffsetsByNode, nodeIndex);
  Value recordCount = buildConstantIndexTableLookup(
      rewriter, loc, participantPlan->recordCountsByNode, nodeIndex);
  Value upper = arith::AddIOp::create(rewriter, loc, lower, recordCount);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, one);
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.recordLoops[forOp] = {
      records, recordSelection,
      buildLocalRecordInductionValues(*participantPlan)};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = buildConstantIndexTableLookup(
      rewriter, loc, participantPlan->recordIndices, forOp.getInductionVar());
  PipeRecordTables recordTables = buildPipeRecordTables(records);
  auto selectedPipe = buildSelectedPipe<SelectOp, SelectedPipeType>(
      rewriter, loc, records, recordTables, recordIndex);
  clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter, foreachWorklist);
  rewriter.eraseOp(op);
  return true;
}

template <typename ForeachOp, typename SelectOp, typename SelectedPipeType>
static bool
tryLowerGridMajorPipeNetForeach(ForeachOp op, RewriterBase &rewriter,
                                PipeForeachLoweringInfo &foreachLoweringInfo,
                                PipeRole role,
                                PipeNetRecordSelection recordSelection,
                                SmallVectorImpl<Operation *> &foreachWorklist) {
  PipeNetRecordsAttr records = op.getRecords();
  FailureOr<GridMajorPipeRecordIndexTables> tables =
      buildGridMajorPipeRecordIndexTables(records, role, op);
  if (failed(tables)) {
    return false;
  }
  const GridMajorPipeRecordIndexTables &tableData = *tables;
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value gridX = arith::ConstantIndexOp::create(rewriter, loc, tableData.gridX);
  Value gridArea =
      arith::ConstantIndexOp::create(rewriter, loc, tableData.gridArea);
  Value nodeRowOffset = arith::MulIOp::create(rewriter, loc, nodeY, gridX);
  Value nodeIndex = arith::AddIOp::create(rewriter, loc, nodeRowOffset, nodeX);
  DeviceDomainAttr deviceDomain =
      records.getPipes().front().getDeviceTransfer().getDomain();
  Value currentDevice = CurrentDeviceIndexOp::create(
      rewriter, loc, rewriter.getIndexType(), deviceDomain);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value nextDevice = arith::AddIOp::create(rewriter, loc, currentDevice, one);
  Value lower = buildConstantIndexTableLookup(
      rewriter, loc, tableData.edgeOffsetsByDevice, currentDevice);
  Value upper = buildConstantIndexTableLookup(
      rewriter, loc, tableData.edgeOffsetsByDevice, nextDevice);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, one);
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.recordLoops[forOp] = {
      records, recordSelection,
      buildGridMajorRecordInductionValues(records, role, tableData)};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value edgeBlock = buildConstantIndexTableLookup(
      rewriter, loc, tableData.edgeBlocks, forOp.getInductionVar());
  Value edgeRecordOffset =
      arith::MulIOp::create(rewriter, loc, edgeBlock, gridArea);
  Value recordIndex =
      arith::AddIOp::create(rewriter, loc, edgeRecordOffset, nodeIndex);
  auto selectedPipe = buildGridMajorSelectedPipe<SelectOp, SelectedPipeType>(
      rewriter, loc, records, tableData, recordIndex, edgeBlock, nodeX, nodeY);
  clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter, foreachWorklist);
  rewriter.eraseOp(op);
  return true;
}

static Value buildRecordRoleMatch(RewriterBase &rewriter, Location loc,
                                  Value nodeX, Value nodeY,
                                  PipeRecordAttr record, PipeRole role) {
  SmallVector<PipeRecordRoleFacts, 2> roleFacts =
      getPipeRecordRoleFacts(record, role);
  assert(roleFacts.size() == 1 && !roleFacts.front().device &&
         "direct record lowering requires one local endpoint role");
  const PipeRecordRoleFacts &facts = roleFacts.front();
  Value minX = arith::ConstantIndexOp::create(rewriter, loc, facts.minX);
  Value minY = arith::ConstantIndexOp::create(rewriter, loc, facts.minY);
  Value maxX = arith::ConstantIndexOp::create(rewriter, loc, facts.maxX);
  Value maxY = arith::ConstantIndexOp::create(rewriter, loc, facts.maxY);
  return buildNodeRangeMatch(rewriter, loc, nodeX, nodeY, minX, minY, maxX,
                             maxY);
}

static CreatePipeOp buildStaticPipeForRecord(RewriterBase &rewriter,
                                             Location loc,
                                             PipeNetRecordsAttr records,
                                             PipeRecordAttr record) {
  PipeType pipeType =
      getPipeTypeFromRecord(rewriter.getContext(), record,
                            static_cast<int64_t>(records.getPipeNetId()));
  BoolAttr isCollectiveAttr =
      record.getIsCollective() ? rewriter.getBoolAttr(true) : BoolAttr();
  return CreatePipeOp::create(
      rewriter, loc, pipeType, rewriter.getI64IntegerAttr(record.getSrcX()),
      rewriter.getI64IntegerAttr(record.getSrcY()),
      rewriter.getI64IntegerAttr(record.getDstStartX()),
      rewriter.getI64IntegerAttr(record.getDstStartY()),
      rewriter.getI64IntegerAttr(record.getDstEndX()),
      rewriter.getI64IntegerAttr(record.getDstEndY()),
      rewriter.getI64IntegerAttr(records.getPipeNetId()),
      records.getPipeNetName(), isCollectiveAttr, DeviceTransferAttr());
}

template <typename ForeachOp>
static void
lowerPipeNetForeachDirect(ForeachOp op, RewriterBase &rewriter, PipeRole role,
                          PipeForeachLoweringInfo &foreachLoweringInfo,
                          SmallVectorImpl<Operation *> &foreachWorklist) {
  Location loc = op.getLoc();
  PipeNetRecordsAttr records = op.getRecords();
  rewriter.setInsertionPoint(op);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  for (PipeRecordAttr record : records.getPipes()) {
    Value staticPipe =
        buildStaticPipeForRecord(rewriter, loc, records, record).getResult();
    Value isActiveRecord =
        buildRecordRoleMatch(rewriter, loc, nodeX, nodeY, record, role);
    auto ifOp = scf::IfOp::create(rewriter, loc, isActiveRecord,
                                  /*withElseRegion=*/false);
    foreachLoweringInfo.controlOps.push_back(ifOp);
    foreachLoweringInfo.ifThenDomains[ifOp] =
        getPipeRecordRoleLaunchNodeDomain(record, role);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    clonePipeForeachBody(op, staticPipe, rewriter, foreachWorklist);
    rewriter.setInsertionPointAfter(ifOp);
  }
  rewriter.eraseOp(op);
}

template <typename ForeachOp, typename SelectOp, typename SelectedPipeType>
static void lowerPipeNetForeach(ForeachOp op, RewriterBase &rewriter,
                                PipeForeachLoweringInfo &foreachLoweringInfo,
                                PipeRole role,
                                PipeNetRecordSelection recordSelection,
                                SmallVectorImpl<Operation *> &foreachWorklist) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  PipeNetRecordsAttr records = op.getRecords();
  if (shouldLowerPipeNetForeachDirect(records)) {
    lowerPipeNetForeachDirect(op, rewriter, role, foreachLoweringInfo,
                              foreachWorklist);
    return;
  }
  if (tryLowerLocalPipeNetForeach<ForeachOp, SelectOp, SelectedPipeType>(
          op, rewriter, foreachLoweringInfo, role, recordSelection,
          foreachWorklist)) {
    return;
  }
  if (tryLowerGridMajorPipeNetForeach<ForeachOp, SelectOp, SelectedPipeType>(
          op, rewriter, foreachLoweringInfo, role, recordSelection,
          foreachWorklist)) {
    return;
  }

  PipeRecordTables tables = buildPipeRecordTables(records);
  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper =
      arith::ConstantIndexOp::create(rewriter, loc, records.getPipes().size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);
  foreachLoweringInfo.recordLoops[forOp] = {records, recordSelection, {}};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = forOp.getInductionVar();
  auto selectedPipe = buildSelectedPipe<SelectOp, SelectedPipeType>(
      rewriter, loc, records, tables, recordIndex);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value roleMatches;
  if (role == PipeRole::Source) {
    roleMatches =
        buildNodePointMatch(rewriter, loc, nodeX, nodeY, selectedPipe.getSrcX(),
                            selectedPipe.getSrcY());
  } else {
    roleMatches = buildNodeRangeMatch(
        rewriter, loc, nodeX, nodeY, selectedPipe.getDstStartX(),
        selectedPipe.getDstStartY(), selectedPipe.getDstEndX(),
        selectedPipe.getDstEndY());
  }
  if (DeviceTransferAttr transfer =
          records.getPipes().front().getDeviceTransfer()) {
    Value currentDevice = CurrentDeviceIndexOp::create(
        rewriter, loc, rewriter.getIndexType(), transfer.getDomain());
    Value endpointDevice = role == PipeRole::Source
                               ? selectedPipe.getSourceDeviceIndex()
                               : selectedPipe.getDestinationDeviceIndex();
    Value deviceMatches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentDevice, endpointDevice);
    roleMatches =
        arith::AndIOp::create(rewriter, loc, roleMatches, deviceMatches);
  }
  auto ifOp = scf::IfOp::create(rewriter, loc, roleMatches,
                                /*withElseRegion=*/false);
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.controlOps.push_back(ifOp);
  foreachLoweringInfo.ifThenDomains[ifOp] =
      getPipeRecordsRoleLaunchNodeDomain(records, role);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter, foreachWorklist);
  rewriter.eraseOp(op);
}

static void
lowerPipeNetForeachSrc(PipeNetForeachSrcOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       SmallVectorImpl<Operation *> &foreachWorklist) {
  lowerPipeNetForeach<PipeNetForeachSrcOp, SelectPipeSrcOp,
                      SelectedPipeSrcType>(
      op, rewriter, foreachLoweringInfo, PipeRole::Source,
      PipeNetRecordSelection::Source, foreachWorklist);
}

static void
lowerPipeNetForeachDst(PipeNetForeachDstOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       SmallVectorImpl<Operation *> &foreachWorklist) {
  lowerPipeNetForeach<PipeNetForeachDstOp, SelectPipeDstOp,
                      SelectedPipeDstType>(
      op, rewriter, foreachLoweringInfo, PipeRole::Destination,
      PipeNetRecordSelection::Destination, foreachWorklist);
}

} // namespace

void lowerPipeNetForeachOps(ModuleOp module,
                            PipeForeachLoweringInfo &foreachLoweringInfo) {
  // A module-wide greedy rewrite also deletes unrelated unused pure reads.
  // Rewrite only foreach operations so this expansion cannot change other IR.
  IRRewriter rewriter(module.getContext());
  SmallVector<Operation *> foreachWorklist;
  collectOutermostPipeNetForeachOps(module, foreachWorklist);
  for (size_t worklistIndex = 0; worklistIndex < foreachWorklist.size();
       ++worklistIndex) {
    Operation *foreachOp = foreachWorklist[worklistIndex];

    // Lower an outer callback before its nested callbacks. The outer rewrite
    // queues only the outermost callbacks cloned from its body.
    if (auto foreachSrcOp = mlir::dyn_cast<PipeNetForeachSrcOp>(foreachOp)) {
      lowerPipeNetForeachSrc(foreachSrcOp, rewriter, foreachLoweringInfo,
                             foreachWorklist);
      continue;
    }
    lowerPipeNetForeachDst(mlir::cast<PipeNetForeachDstOp>(foreachOp), rewriter,
                           foreachLoweringInfo, foreachWorklist);
  }
}

} // namespace mlir::tt::ttl
