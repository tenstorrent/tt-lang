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

namespace {

static std::optional<PipeNetRecordLoop> getHighLevelRecordLoop(Operation *op) {
  if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(op)) {
    return PipeNetRecordLoop{foreachSrc.getRecords(),
                             PipeNetRecordSelection::Source};
  }
  if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(op)) {
    return PipeNetRecordLoop{foreachDst.getRecords(),
                             PipeNetRecordSelection::Destination};
  }
  return std::nullopt;
}

static SmallVector<std::uint64_t>
getMatchingRecordIndices(const PipeNetRecordLoop &recordLoop,
                         LaunchNodeCoord coord) {
  SmallVector<std::uint64_t> matchingRecordIndices;
  for (auto [recordIndex, record] :
       llvm::enumerate(recordLoop.records.getPipes())) {
    LaunchNodeDomain recordDomain =
        recordLoop.selection == PipeNetRecordSelection::Source
            ? getPipeRecordSourceLaunchNodeDomain(record)
            : getPipeRecordDestinationLaunchNodeDomain(record);
    if (knownLaunchNodeDomainContains(recordDomain, coord)) {
      matchingRecordIndices.push_back(recordIndex);
    }
  }
  return matchingRecordIndices;
}

static WalkResult walkPipeNetOpsInProgramOrderImpl(
    Operation *op, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<
        WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t> executionCountDivisor)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords,
    std::optional<std::uint64_t> executionCountDivisor) {
  std::optional<PipeNetRecordLoop> maybeRecordLoop = getHighLevelRecordLoop(op);
  if (!maybeRecordLoop) {
    maybeRecordLoop = resolveGeneratedRecordLoop(op);
  }
  if (maybeRecordLoop) {
    SmallVector<std::uint64_t> matchingRecordIndices =
        getMatchingRecordIndices(*maybeRecordLoop, coord);
    std::optional<std::uint64_t> nestedExecutionCountDivisor =
        executionCountDivisor;
    if (executionCountDivisor) {
      nestedExecutionCountDivisor = llvm::checkedMulUnsigned(
          *executionCountDivisor,
          static_cast<std::uint64_t>(matchingRecordIndices.size()));
    }

    for (std::uint64_t recordIndex : matchingRecordIndices) {
      activeRecords.push_back({op, recordIndex});
      llvm::scope_exit restoreActiveRecords([&] { activeRecords.pop_back(); });
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            if (walkPipeNetOpsInProgramOrderImpl(
                    &nestedOp, coord, resolveGeneratedRecordLoop,
                    visitOperation, activeRecords, nestedExecutionCountDivisor)
                    .wasInterrupted()) {
              return WalkResult::interrupt();
            }
          }
        }
      }
    }
    return WalkResult::advance();
  }

  if (visitOperation(op, activeRecords, executionCountDivisor)
          .wasInterrupted()) {
    return WalkResult::interrupt();
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (walkPipeNetOpsInProgramOrderImpl(
                &nestedOp, coord, resolveGeneratedRecordLoop, visitOperation,
                activeRecords, executionCountDivisor)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
  }
  return WalkResult::advance();
}

} // namespace

WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<
        WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t> executionCountDivisor)>
        visitOperation) {
  SmallVector<ActivePipeNetRecord> activeRecords;
  return walkPipeNetOpsInProgramOrderImpl(
      root, coord, resolveGeneratedRecordLoop, visitOperation, activeRecords,
      /*executionCountDivisor=*/1);
}

WalkResult walkPipeNetOpsInProgramOrder(
    Operation *root, LaunchNodeCoord coord,
    llvm::function_ref<std::optional<PipeNetRecordLoop>(Operation *)>
        resolveGeneratedRecordLoop,
    llvm::function_ref<
        WalkResult(Operation *, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t> executionCountDivisor)>
        visitOperation,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords,
    std::optional<std::uint64_t> executionCountDivisor) {
  return walkPipeNetOpsInProgramOrderImpl(
      root, coord, resolveGeneratedRecordLoop, visitOperation, activeRecords,
      executionCountDivisor);
}

} // namespace mlir::tt::ttl
