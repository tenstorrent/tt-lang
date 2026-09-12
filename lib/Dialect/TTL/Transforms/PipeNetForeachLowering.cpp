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
#include <map>
#include <utility>

namespace mlir::tt::ttl {
namespace {

namespace ttk = mlir::tt::ttkernel;

// Duplicating up to four callback bodies avoids table lookups for small nets.
// Larger nets use one loop so the transfer protocol body is not duplicated for
// every record.
constexpr size_t kPipeNetForeachDirectRecordLimit = 4;

static bool shouldLowerPipeNetForeachDirect(PipeNetRecordsAttr records) {
  return records.getMappings().empty() &&
         !records.getPipes().front().getDeviceTransfer() &&
         records.getPipes().size() <= kPipeNetForeachDirectRecordLimit;
}

template <typename SelectOp, typename SelectedType>
static SelectOp
buildSelectedPipe(OpBuilder &builder, Location loc, PipeNetRecordsAttr records,
                  const PipeRecordTables &tables, Value recordIndex,
                  Value pipeIndex, Value sourceDeviceIndex,
                  Value destinationDeviceIndex) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value srcInDstRangeIndex = buildConstantIndexTableLookup(
      builder, loc, tables.srcInDstRange, pipeIndex);
  Value srcInDstRange = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, srcInDstRangeIndex, zero);
  return SelectOp::create(
      builder, loc, SelectedType::get(builder.getContext()), recordIndex,
      buildConstantIndexTableLookup(builder, loc, tables.srcX, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.srcY, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstStartX, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstStartY, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstEndX, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.dstEndY, pipeIndex),
      buildConstantIndexTableLookup(builder, loc, tables.numDests, pipeIndex),
      srcInDstRange, sourceDeviceIndex, destinationDeviceIndex, records);
}

template <typename SelectOp, typename SelectedType>
static SelectOp buildGridMajorSelectedPipe(
    OpBuilder &builder, Location loc, PipeNetRecordsAttr records,
    const DevicePipeNetParticipantPlan &tables, Value recordIndex,
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
    const DevicePipeNetParticipantPlan &tables) {
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

// Associate each concrete graph record with its endpoint-local callback
// iteration so execution-count analysis can evaluate the cloned body.
static RecordInductionMap
buildGraphRecordInductionValues(PipeNetRecordsAttr records, PipeRole role) {
  FailureOr<SmallVector<PipeRecordLocalIndex>> localIndices =
      getPipeRecordLocalIndices(records, role);
  assert(succeeded(localIndices) &&
         "verified graph records must have endpoint-local indices");

  RecordInductionMap inductionValues;
  forEachPipeRecord(records, [&](std::uint64_t recordIndex,
                                 PipeRecordAttr record) {
    assert(recordIndex < localIndices->size() &&
           "each graph record must have a local index");
    for (const PipeRecordRoleFacts &facts :
         getPipeRecordRoleFacts(record, role)) {
      assert(facts.device &&
             "graph records must identify their endpoint device");
      for (int64_t nodeY = facts.minY; nodeY <= facts.maxY; ++nodeY) {
        for (int64_t nodeX = facts.minX; nodeX <= facts.maxX; ++nodeX) {
          bool inserted =
              inductionValues
                  .try_emplace(
                      std::make_pair(LaunchExecutionLocation({nodeX, nodeY},
                                                             facts.deviceDomain,
                                                             facts.device),
                                     recordIndex),
                      (*localIndices)[recordIndex].index)
                  .second;
          assert(inserted &&
                 "a graph record must select each endpoint node once");
        }
      }
    }
  });
  return inductionValues;
}

// Map each same-coordinate transfer to its position among graph edges that
// select the same logical device and endpoint role.
static RecordInductionMap buildGraphSameCoordinateRecordInductionValues(
    PipeNetRecordsAttr records, PipeRole role, std::uint64_t nodePipeCount) {
  RecordInductionMap inductionValues =
      buildGraphRecordInductionValues(records, role);
  for (auto &entry : inductionValues) {
    entry.second /= nodePipeCount;
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
  forOp->setAttr(kPipeNetLocalRecordLoopAttrName, rewriter.getUnitAttr());
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.recordLoops[forOp] = {
      records, recordSelection,
      buildLocalRecordInductionValues(*participantPlan)};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = buildConstantIndexTableLookup(
      rewriter, loc, participantPlan->recordIndices, forOp.getInductionVar());
  PipeRecordTables recordTables = buildPipeRecordTables(records.getPipes());
  Value sourceDeviceIndex = buildConstantIndexTableLookup(
      rewriter, loc, recordTables.sourceDeviceIndex, recordIndex);
  Value destinationDeviceIndex = buildConstantIndexTableLookup(
      rewriter, loc, recordTables.destinationDeviceIndex, recordIndex);
  auto selectedPipe = buildSelectedPipe<SelectOp, SelectedPipeType>(
      rewriter, loc, records, recordTables, recordIndex, recordIndex,
      sourceDeviceIndex, destinationDeviceIndex);
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
  FailureOr<std::pair<int64_t, int64_t>> launchGrid = getLaunchGrid(op);
  if (failed(launchGrid)) {
    return false;
  }
  auto [launchGridX, launchGridY] = *launchGrid;
  FailureOr<DevicePipeNetParticipantPlan> tables =
      buildDevicePipeNetParticipantPlan(records, role, launchGridX,
                                        launchGridY);
  if (failed(tables)) {
    return false;
  }
  // Retain direct matching when the dense table is not smaller than the
  // original record list.
  if (tables->recordCountsByDevice.size() >= records.getPipes().size()) {
    return false;
  }
  const DevicePipeNetParticipantPlan &tableData = *tables;
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

// Validate the node relation and prepare immutable lowering data for every
// mapping before any callback IR is changed.
static FailureOr<GraphPipeMappingForeachPlans>
buildGraphPipeMappingForeachPlans(PipeNetRecordsAttr records,
                                  Operation *diagnosticAnchor,
                                  std::pair<int64_t, int64_t> launchGrid) {
  if (failed(validatePipeNetLaunchNodeRelation(
          records, PipeRole::Active, launchGrid.first, launchGrid.second,
          [&]() { return diagnosticAnchor->emitOpError(); }))) {
    return failure();
  }

  GraphPipeMappingForeachPlans plans;
  plans.reserve(records.getMappings().size());
  for (PipeMappingAttr mapping : records.getMappings()) {
    std::unique_ptr<TransferGraph> graph =
        createTransferGraph(mapping.getGraph());
    FailureOr<std::uint64_t> edgeCount = graph->getEdgeCount();
    std::uint64_t nodePipeCount = mapping.getPipes().size();
    std::optional<std::uint64_t> concreteRecordCount =
        succeeded(edgeCount)
            ? llvm::checkedMulUnsigned(*edgeCount, nodePipeCount)
            : std::nullopt;
    if (!concreteRecordCount ||
        *concreteRecordCount >
            static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
      diagnosticAnchor->emitOpError(
          "graph PipeNet record count exceeds the supported index range");
      return failure();
    }

    PipeNetRecordsAttr mappingRecords = PipeNetRecordsAttr::get(
        records.getContext(), records.getPipeNetId(), records.getPipeNetName(),
        ArrayRef<PipeRecordAttr>(), ArrayRef<PipeMappingAttr>{mapping});
    bool usesMatchingNodeCoordinates = hasMatchingPipeForEveryLaunchNode(
        mapping.getPipes(), launchGrid.first, launchGrid.second);
    PipeRecordTables nodePipeTables =
        usesMatchingNodeCoordinates ? PipeRecordTables()
                                    : buildPipeRecordTables(mapping.getPipes());
    plans.push_back(GraphPipeMappingForeachPlan{
        mappingRecords, std::move(nodePipeTables), std::move(graph),
        static_cast<int64_t>(nodePipeCount), launchGrid.first,
        usesMatchingNodeCoordinates});
  }
  return plans;
}

template <typename ForeachOp, typename SelectOp, typename SelectedPipeType>
static LogicalResult
lowerGraphPipeNetForeach(ForeachOp op, RewriterBase &rewriter,
                         PipeForeachLoweringInfo &foreachLoweringInfo,
                         PipeRole role, PipeNetRecordSelection recordSelection,
                         const GraphPipeNetForeachPlans &plansByRecordsAndGrid,
                         SmallVectorImpl<Operation *> &foreachWorklist) {
  auto recordsIt = plansByRecordsAndGrid.find(op.getRecords());
  FailureOr<std::pair<int64_t, int64_t>> launchGrid = getLaunchGrid(op);
  assert(recordsIt != plansByRecordsAndGrid.end() && succeeded(launchGrid) &&
         "preflight must plan every graph PipeNet record relation");
  auto plansIt = recordsIt->second.find(*launchGrid);
  assert(plansIt != recordsIt->second.end() &&
         "preflight must plan every graph PipeNet launch grid");

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  for (const GraphPipeMappingForeachPlan &plan : plansIt->second) {
    Value nodeX =
        ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
    Value nodeY =
        ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
    Value currentDevice = CurrentDeviceIndexOp::create(
        rewriter, loc, rewriter.getIndexType(), plan.graph->getDomain());
    Value incidentEdgeCount =
        plan.graph->buildIncidentEdgeCount(rewriter, loc, currentDevice, role);
    Value nodePipeCount =
        arith::ConstantIndexOp::create(rewriter, loc, plan.nodePipeCount);
    Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper = plan.usesMatchingNodeCoordinates
                      ? incidentEdgeCount
                      : Value(arith::MulIOp::create(
                            rewriter, loc, incidentEdgeCount, nodePipeCount));
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);
    foreachLoweringInfo.recordLoops[forOp] = {
        plan.records, recordSelection,
        plan.usesMatchingNodeCoordinates
            ? buildGraphSameCoordinateRecordInductionValues(
                  plan.records, role,
                  static_cast<std::uint64_t>(plan.nodePipeCount))
            : buildGraphRecordInductionValues(plan.records, role)};

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value localRecordIndex = forOp.getInductionVar();
    Value incidentEdgeIndex;
    Value nodePipeIndex;
    if (plan.usesMatchingNodeCoordinates) {
      incidentEdgeIndex = localRecordIndex;
      Value gridX =
          arith::ConstantIndexOp::create(rewriter, loc, plan.launchGridX);
      Value nodeRowOffset = arith::MulIOp::create(rewriter, loc, nodeY, gridX);
      nodePipeIndex =
          arith::AddIOp::create(rewriter, loc, nodeRowOffset, nodeX);
    } else {
      incidentEdgeIndex = arith::DivSIOp::create(
          rewriter, loc, localRecordIndex, nodePipeCount);
      nodePipeIndex = arith::RemSIOp::create(rewriter, loc, localRecordIndex,
                                             nodePipeCount);
    }
    TransferGraphEdgeIndexValues edgeIndices =
        plan.graph->buildIncidentEdgeIndexValues(rewriter, loc, currentDevice,
                                                 incidentEdgeIndex, role);
    Value edgeRecordBase = arith::MulIOp::create(
        rewriter, loc, edgeIndices.edgeOrdinal, nodePipeCount);
    Value recordIndex =
        arith::AddIOp::create(rewriter, loc, edgeRecordBase, nodePipeIndex);
    auto selectedPipe = [&]() -> SelectOp {
      if (!plan.usesMatchingNodeCoordinates) {
        return buildSelectedPipe<SelectOp, SelectedPipeType>(
            rewriter, loc, plan.records, plan.nodePipeTables, recordIndex,
            nodePipeIndex, edgeIndices.sourceDeviceIndex,
            edgeIndices.destinationDeviceIndex);
      }
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value sourceInDestination =
          arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      return SelectOp::create(
          rewriter, loc, SelectedPipeType::get(rewriter.getContext()),
          recordIndex, nodeX, nodeY, nodeX, nodeY, nodeX, nodeY, one,
          sourceInDestination, edgeIndices.sourceDeviceIndex,
          edgeIndices.destinationDeviceIndex, plan.records);
    }();
    foreachLoweringInfo.controlOps.push_back(forOp);
    if (plan.usesMatchingNodeCoordinates) {
      clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter,
                           foreachWorklist);
      rewriter.setInsertionPointAfter(forOp);
      continue;
    }
    Value roleMatches;
    if (role == PipeRole::Source) {
      roleMatches =
          buildNodePointMatch(rewriter, loc, nodeX, nodeY,
                              selectedPipe.getSrcX(), selectedPipe.getSrcY());
    } else {
      roleMatches = buildNodeRangeMatch(
          rewriter, loc, nodeX, nodeY, selectedPipe.getDstStartX(),
          selectedPipe.getDstStartY(), selectedPipe.getDstEndX(),
          selectedPipe.getDstEndY());
    }
    auto ifOp = scf::IfOp::create(rewriter, loc, roleMatches,
                                  /*withElseRegion=*/false);
    foreachLoweringInfo.controlOps.push_back(ifOp);
    foreachLoweringInfo.ifThenDomains[ifOp] =
        getPipeRecordsRoleLaunchNodeDomain(plan.records, role);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter, foreachWorklist);
    rewriter.setInsertionPointAfter(forOp);
  }
  rewriter.eraseOp(op);
  return success();
}

template <typename ForeachOp, typename SelectOp, typename SelectedPipeType>
static LogicalResult
lowerPipeNetForeach(ForeachOp op, RewriterBase &rewriter,
                    PipeForeachLoweringInfo &foreachLoweringInfo, PipeRole role,
                    PipeNetRecordSelection recordSelection,
                    const GraphPipeNetForeachPlans &plansByRecordsAndGrid,
                    SmallVectorImpl<Operation *> &foreachWorklist) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  PipeNetRecordsAttr records = op.getRecords();
  if (!records.getMappings().empty()) {
    return lowerGraphPipeNetForeach<ForeachOp, SelectOp, SelectedPipeType>(
        op, rewriter, foreachLoweringInfo, role, recordSelection,
        plansByRecordsAndGrid, foreachWorklist);
  }
  if (shouldLowerPipeNetForeachDirect(records)) {
    lowerPipeNetForeachDirect(op, rewriter, role, foreachLoweringInfo,
                              foreachWorklist);
    return success();
  }
  if (tryLowerLocalPipeNetForeach<ForeachOp, SelectOp, SelectedPipeType>(
          op, rewriter, foreachLoweringInfo, role, recordSelection,
          foreachWorklist)) {
    return success();
  }
  if (tryLowerGridMajorPipeNetForeach<ForeachOp, SelectOp, SelectedPipeType>(
          op, rewriter, foreachLoweringInfo, role, recordSelection,
          foreachWorklist)) {
    return success();
  }

  PipeRecordTables tables = buildPipeRecordTables(records.getPipes());
  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper =
      arith::ConstantIndexOp::create(rewriter, loc, records.getPipes().size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  // Keep the fallback loop rolled because it scans the complete PipeNet table;
  // only the bounded per-node table is suitable for unconditional unrolling.
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);
  foreachLoweringInfo.recordLoops[forOp] = {records, recordSelection, {}};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = forOp.getInductionVar();
  Value sourceDeviceIndex = buildConstantIndexTableLookup(
      rewriter, loc, tables.sourceDeviceIndex, recordIndex);
  Value destinationDeviceIndex = buildConstantIndexTableLookup(
      rewriter, loc, tables.destinationDeviceIndex, recordIndex);
  auto selectedPipe = buildSelectedPipe<SelectOp, SelectedPipeType>(
      rewriter, loc, records, tables, recordIndex, recordIndex,
      sourceDeviceIndex, destinationDeviceIndex);
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
  return success();
}

static LogicalResult
lowerPipeNetForeachSrc(PipeNetForeachSrcOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       const GraphPipeNetForeachPlans &plansByRecordsAndGrid,
                       SmallVectorImpl<Operation *> &foreachWorklist) {
  return lowerPipeNetForeach<PipeNetForeachSrcOp, SelectPipeSrcOp,
                             SelectedPipeSrcType>(
      op, rewriter, foreachLoweringInfo, PipeRole::Source,
      PipeNetRecordSelection::Source, plansByRecordsAndGrid, foreachWorklist);
}

static LogicalResult
lowerPipeNetForeachDst(PipeNetForeachDstOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       const GraphPipeNetForeachPlans &plansByRecordsAndGrid,
                       SmallVectorImpl<Operation *> &foreachWorklist) {
  return lowerPipeNetForeach<PipeNetForeachDstOp, SelectPipeDstOp,
                             SelectedPipeDstType>(
      op, rewriter, foreachLoweringInfo, PipeRole::Destination,
      PipeNetRecordSelection::Destination, plansByRecordsAndGrid,
      foreachWorklist);
}

} // namespace

FailureOr<GraphPipeNetForeachPlans>
buildGraphPipeNetForeachPlans(ModuleOp module) {
  GraphPipeNetForeachPlans plansByRecordsAndGrid;
  WalkResult result = module.walk([&](Operation *operation) {
    PipeNetRecordsAttr records;
    if (auto foreachSrc = dyn_cast<PipeNetForeachSrcOp>(operation)) {
      records = foreachSrc.getRecords();
    } else if (auto foreachDst = dyn_cast<PipeNetForeachDstOp>(operation)) {
      records = foreachDst.getRecords();
    } else {
      return WalkResult::advance();
    }
    if (records.getMappings().empty()) {
      return WalkResult::advance();
    }
    FailureOr<std::pair<int64_t, int64_t>> launchGrid =
        getLaunchGrid(operation);
    if (failed(launchGrid)) {
      operation->emitOpError(
          "graph PipeNet callback requires a valid ttl.launch_grid with two "
          "positive integer extents");
      return WalkResult::interrupt();
    }
    auto &plansByGrid = plansByRecordsAndGrid[records];
    if (plansByGrid.count(*launchGrid)) {
      return WalkResult::advance();
    }
    FailureOr<GraphPipeMappingForeachPlans> plans =
        buildGraphPipeMappingForeachPlans(records, operation, *launchGrid);
    if (failed(plans)) {
      return WalkResult::interrupt();
    }
    plansByGrid.try_emplace(*launchGrid, std::move(*plans));
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return plansByRecordsAndGrid;
}

LogicalResult
lowerPipeNetForeachOps(ModuleOp module,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       const GraphPipeNetForeachPlans &plansByRecordsAndGrid) {
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
      if (failed(lowerPipeNetForeachSrc(
              foreachSrcOp, rewriter, foreachLoweringInfo,
              plansByRecordsAndGrid, foreachWorklist))) {
        return failure();
      }
      continue;
    }
    if (failed(lowerPipeNetForeachDst(
            mlir::cast<PipeNetForeachDstOp>(foreachOp), rewriter,
            foreachLoweringInfo, plansByRecordsAndGrid, foreachWorklist))) {
      return failure();
    }
  }
  return success();
}

} // namespace mlir::tt::ttl
