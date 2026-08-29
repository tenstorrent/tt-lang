// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/PipeNetExecutionUtils.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir::tt::ttl {

std::optional<std::uint64_t>
getPipeNetRecordLoopInductionValue(const PipeNetRecordLoop &recordLoop,
                                   const LaunchExecutionLocation &location,
                                   std::uint64_t recordIndex) {
  if (recordLoop.indirectInductionValues.empty()) {
    return recordIndex;
  }
  auto iteration =
      recordLoop.indirectInductionValues.find({location, recordIndex});
  return iteration == recordLoop.indirectInductionValues.end()
             ? std::nullopt
             : std::optional<std::uint64_t>(iteration->second);
}

std::optional<std::uint64_t>
getActivePipeNetRecordIndex(ArrayRef<ActivePipeNetRecord> activeRecords,
                            Operation *loopOp) {
  auto activeIt = llvm::find_if(llvm::reverse(activeRecords),
                                [&](const ActivePipeNetRecord &active) {
                                  return active.loopOp == loopOp;
                                });
  if (activeIt == activeRecords.rend()) {
    return std::nullopt;
  }
  return activeIt->recordIndex;
}

std::optional<llvm::APInt>
evaluateSelectedPipeRecordValue(Value value, PipeRecordAttr record) {
  constexpr unsigned indexBitWidth = IndexType::kInternalStorageBitWidth;
  if (value.getDefiningOp<SelectedPipeSourceDeviceIndexOp>()) {
    DeviceTransferAttr transfer = record.getDeviceTransfer();
    if (!transfer) {
      return std::nullopt;
    }
    return llvm::APInt(indexBitWidth,
                       getLogicalDeviceIndex(transfer.getDomain(),
                                             transfer.getEdge().getSource()));
  }
  if (value.getDefiningOp<SelectedPipeDestinationDeviceIndexOp>()) {
    DeviceTransferAttr transfer = record.getDeviceTransfer();
    if (!transfer || !transfer.getEdge().getDestination()) {
      return std::nullopt;
    }
    return llvm::APInt(indexBitWidth, getLogicalDeviceIndex(
                                          transfer.getDomain(),
                                          transfer.getEdge().getDestination()));
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return std::nullopt;
  }
  if (value.getDefiningOp<SelectedPipeSourceCoordinatesOp>()) {
    assert(result.getResultNumber() < 2 &&
           "source coordinate result must be x or y");
    int64_t coordinate =
        result.getResultNumber() == 0 ? record.getSrcX() : record.getSrcY();
    return llvm::APInt(indexBitWidth, coordinate);
  }
  if (!value.getDefiningOp<SelectedPipeDestinationCoordinatesOp>()) {
    return std::nullopt;
  }
  int64_t coordinate = 0;
  switch (result.getResultNumber()) {
  case 0:
    coordinate = record.getDstStartX();
    break;
  case 1:
    coordinate = record.getDstStartY();
    break;
  case 2:
    coordinate = record.getDstEndX();
    break;
  case 3:
    coordinate = record.getDstEndY();
    break;
  default:
    llvm_unreachable("destination coordinate result must be in bounds");
  }
  return llvm::APInt(indexBitWidth, coordinate);
}

std::optional<llvm::APInt> evaluateActivePipeNetRecordValue(
    Value value, ArrayRef<ActivePipeNetRecord> activeRecords,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveFunctionArgument) {
  Value selectedPipe;
  if (auto sourceDevice =
          value.getDefiningOp<SelectedPipeSourceDeviceIndexOp>()) {
    selectedPipe = sourceDevice.getPipe();
  } else if (auto destinationDevice =
                 value.getDefiningOp<SelectedPipeDestinationDeviceIndexOp>()) {
    selectedPipe = destinationDevice.getPipe();
  } else if (auto sourceCoordinates =
                 value.getDefiningOp<SelectedPipeSourceCoordinatesOp>()) {
    selectedPipe = sourceCoordinates.getPipe();
  } else if (auto destinationCoordinates =
                 value.getDefiningOp<SelectedPipeDestinationCoordinatesOp>()) {
    selectedPipe = destinationCoordinates.getPipe();
  } else {
    return std::nullopt;
  }

  llvm::SmallPtrSet<Value, 4> visited;
  selectedPipe = traceUnrealizedCasts(selectedPipe);
  while (auto argument = dyn_cast<BlockArgument>(selectedPipe)) {
    if (succeeded(getSelectedPipeRecords(selectedPipe))) {
      break;
    }
    if (!visited.insert(selectedPipe).second) {
      return std::nullopt;
    }
    std::optional<Value> operand = resolveFunctionArgument(argument);
    if (!operand) {
      return std::nullopt;
    }
    selectedPipe = traceUnrealizedCasts(*operand);
  }

  FailureOr<SelectedPipeRecords> maybeRecords =
      getSelectedPipeRecords(selectedPipe);
  if (failed(maybeRecords) || !maybeRecords->maybeForeachOp) {
    return std::nullopt;
  }
  std::optional<std::uint64_t> maybeRecordIndex =
      getActivePipeNetRecordIndex(activeRecords, maybeRecords->maybeForeachOp);
  if (!maybeRecordIndex) {
    return std::nullopt;
  }
  ArrayRef<PipeRecordAttr> records = maybeRecords->records.getPipes();
  assert(*maybeRecordIndex < records.size() &&
         "active PipeNet record index must be in bounds");
  return evaluateSelectedPipeRecordValue(value, records[*maybeRecordIndex]);
}

namespace {

static std::optional<PipeNetRecordLoop> getHighLevelRecordLoop(Operation *op) {
  if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(op)) {
    return PipeNetRecordLoop{
        foreachSrc.getRecords(), PipeNetRecordSelection::Source, {}};
  }
  if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(op)) {
    return PipeNetRecordLoop{
        foreachDst.getRecords(), PipeNetRecordSelection::Destination, {}};
  }
  return std::nullopt;
}

static SmallVector<std::uint64_t>
getMatchingRecordIndices(const PipeNetRecordLoop &recordLoop,
                         LaunchNodeCoord coord) {
  SmallVector<std::uint64_t> matchingRecordIndices;
  for (auto [recordIndex, record] :
       llvm::enumerate(recordLoop.records.getPipes())) {
    PipeRole role = recordLoop.selection == PipeNetRecordSelection::Source
                        ? PipeRole::Source
                        : PipeRole::Destination;
    LaunchNodeDomain recordDomain =
        getPipeRecordRoleLaunchNodeDomain(record, role);
    if (knownLaunchNodeDomainContains(recordDomain, coord)) {
      matchingRecordIndices.push_back(recordIndex);
    }
  }
  return matchingRecordIndices;
}

static std::optional<std::uint64_t>
getMatchingRecordCount(const PipeNetRecordLoop &recordLoop,
                       const LaunchExecutionLocation &location) {
  PipeRole role = recordLoop.selection == PipeNetRecordSelection::Source
                      ? PipeRole::Source
                      : PipeRole::Destination;
  std::uint64_t matchingRecordCount = 0;
  for (PipeRecordAttr record : recordLoop.records.getPipes()) {
    std::optional<bool> matches =
        pipeRecordRoleMatchesAtLaunchLocation(record, role, location);
    if (!matches) {
      return std::nullopt;
    }
    matchingRecordCount += *matches;
  }
  return matchingRecordCount;
}

static WalkResult walkPipeNetOpsInProgramOrderImpl(
    Operation *op, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords) {
  std::optional<PipeNetRecordLoop> maybeRecordLoop = getHighLevelRecordLoop(op);
  if (!maybeRecordLoop) {
    maybeRecordLoop = resolveGeneratedRecordLoop(op);
  }
  if (maybeRecordLoop) {
    SmallVector<std::uint64_t> matchingRecordIndices =
        getMatchingRecordIndices(*maybeRecordLoop, coord);
    for (std::uint64_t recordIndex : matchingRecordIndices) {
      activeRecords.push_back({op, recordIndex});
      llvm::scope_exit restoreActiveRecords([&] { activeRecords.pop_back(); });
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            if (walkPipeNetOpsInProgramOrderImpl(&nestedOp, coord,
                                                 resolveGeneratedRecordLoop,
                                                 visitOperation, activeRecords)
                    .wasInterrupted()) {
              return WalkResult::interrupt();
            }
          }
        }
      }
    }
    return WalkResult::advance();
  }

  if (visitOperation(op, activeRecords).wasInterrupted()) {
    return WalkResult::interrupt();
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (walkPipeNetOpsInProgramOrderImpl(&nestedOp, coord,
                                             resolveGeneratedRecordLoop,
                                             visitOperation, activeRecords)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
  }
  return WalkResult::advance();
}

} // namespace

ActivePipeNetExecution evaluateActivePipeNetExecution(
    ArrayRef<ActivePipeNetRecord> activeRecords,
    const LaunchExecutionLocation &location,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop) {
  ActivePipeNetExecution execution;
  for (const ActivePipeNetRecord &activeRecord : activeRecords) {
    std::optional<PipeNetRecordLoop> recordLoop =
        getHighLevelRecordLoop(activeRecord.loopOp);
    if (!recordLoop) {
      recordLoop = resolveGeneratedRecordLoop(activeRecord.loopOp);
    }
    if (!recordLoop) {
      execution.countDivisor = std::nullopt;
      continue;
    }
    assert(activeRecord.recordIndex < recordLoop->records.getPipes().size() &&
           "active PipeNet record index must be in bounds");
    PipeRole role = recordLoop->selection == PipeNetRecordSelection::Source
                        ? PipeRole::Source
                        : PipeRole::Destination;
    std::optional<bool> selectedRecordMatches =
        pipeRecordRoleMatchesAtLaunchLocation(
            recordLoop->records.getPipes()[activeRecord.recordIndex], role,
            location);
    if (selectedRecordMatches && !*selectedRecordMatches) {
      execution.mayExecute = false;
      return execution;
    }
    if (!selectedRecordMatches) {
      execution.countDivisor = std::nullopt;
    }
    std::optional<std::uint64_t> matchingRecordCount =
        getMatchingRecordCount(*recordLoop, location);
    if (!matchingRecordCount) {
      execution.countDivisor = std::nullopt;
      continue;
    }
    if (!execution.countDivisor) {
      continue;
    }
    assert(*matchingRecordCount != 0 && "an active record must have a match");
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned(*execution.countDivisor, *matchingRecordCount);
    if (!product) {
      execution.countDivisor = std::nullopt;
      continue;
    }
    execution.countDivisor = *product;
  }
  return execution;
}

WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>)>
        visitOperation) {
  SmallVector<ActivePipeNetRecord> activeRecords;
  return walkPipeNetOpsInProgramOrderImpl(
      root, coord, resolveGeneratedRecordLoop, visitOperation, activeRecords);
}

WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords) {
  return walkPipeNetOpsInProgramOrderImpl(
      root, coord, resolveGeneratedRecordLoop, visitOperation, activeRecords);
}

} // namespace mlir::tt::ttl
