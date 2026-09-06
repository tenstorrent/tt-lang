// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/PipeNetParticipantPlan.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::tt::ttl {

// Planner table indices are int64_t, so domain products must fit that type.
static FailureOr<int64_t> getRepresentableDeviceCount(DeviceDomainAttr domain) {
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

FailureOr<LocalPipeNetParticipantPlan> buildLocalPipeNetParticipantPlan(
    PipeNetRecordsAttr records, PipeRole role, int64_t gridX, int64_t gridY,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (gridX <= 0 || gridY <= 0) {
    if (emitError) {
      emitError()
          << "local PipeNet launch grid (" << gridX << ", " << gridY
          << ") must have two positive extents; correct the launch grid";
    }
    return failure();
  }
  std::optional<int64_t> maybeGridArea = llvm::checkedMul(gridX, gridY);
  if (!maybeGridArea ||
      records.getPipes().size() >
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    if (emitError) {
      emitError() << "local PipeNet table for launch grid (" << gridX << ", "
                  << gridY << ") and " << records.getPipes().size()
                  << " records exceeds the signed 64-bit indexing limit; "
                     "reduce the launch grid or split the PipeNet";
    }
    return failure();
  }

  SmallVector<SmallVector<int64_t>> recordsByNode(
      static_cast<std::size_t>(*maybeGridArea));
  for (auto [recordIndex, record] : llvm::enumerate(records.getPipes())) {
    if (record.getDeviceTransfer()) {
      if (emitError) {
        emitError()
            << "PipeNet record " << recordIndex
            << " specifies a logical-device transfer but local planning "
               "was requested; use logical-device PipeNet lowering";
      }
      return failure();
    }
    llvm::SmallDenseSet<int64_t, 4> participantIndices;
    for (const PipeRecordRoleFacts &facts :
         getPipeRecordRoleFacts(record, role)) {
      if (facts.device || facts.minX < 0 || facts.minY < 0 ||
          facts.minX > facts.maxX || facts.minY > facts.maxY ||
          facts.maxX >= gridX || facts.maxY >= gridY) {
        if (emitError) {
          emitError() << "PipeNet record " << recordIndex
                      << " has endpoint range core_x=" << facts.minX << ".."
                      << facts.maxX << ", core_y=" << facts.minY << ".."
                      << facts.maxY << " outside the local launch grid ("
                      << gridX << ", " << gridY
                      << "); increase the launch grid or correct the PipeNet "
                         "endpoint coordinates";
        }
        return failure();
      }
      for (int64_t nodeY = facts.minY; nodeY <= facts.maxY; ++nodeY) {
        for (int64_t nodeX = facts.minX; nodeX <= facts.maxX; ++nodeX) {
          participantIndices.insert(nodeY * gridX + nodeX);
        }
      }
    }
    for (int64_t participantIndex : participantIndices) {
      recordsByNode[static_cast<std::size_t>(participantIndex)].push_back(
          static_cast<int64_t>(recordIndex));
    }
  }

  LocalPipeNetParticipantPlan plan;
  plan.gridX = gridX;
  for (ArrayRef<int64_t> nodeRecords : recordsByNode) {
    plan.recordOffsetsByNode.push_back(
        static_cast<int64_t>(plan.recordIndices.size()));
    plan.recordCountsByNode.push_back(static_cast<int64_t>(nodeRecords.size()));
    plan.recordIndices.append(nodeRecords.begin(), nodeRecords.end());
  }
  return plan;
}

FailureOr<DevicePipeNetParticipantPlan>
buildDevicePipeNetParticipantPlan(PipeNetRecordsAttr records, PipeRole role,
                                  int64_t gridX, int64_t gridY) {
  if (records.getPipes().empty() ||
      (role != PipeRole::Source && role != PipeRole::Destination) ||
      gridX <= 0 || gridY <= 0) {
    return failure();
  }
  DeviceTransferAttr firstTransfer =
      records.getPipes().front().getDeviceTransfer();
  if (!firstTransfer) {
    return failure();
  }
  FailureOr<int64_t> deviceCount =
      getRepresentableDeviceCount(firstTransfer.getDomain());
  std::optional<int64_t> maybeGridArea = llvm::checkedMul(gridX, gridY);
  if (failed(deviceCount) || !maybeGridArea ||
      records.getPipes().size() % *maybeGridArea != 0 ||
      records.getPipes().size() >
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return failure();
  }

  int64_t gridArea = *maybeGridArea;
  int64_t recordCount = static_cast<int64_t>(records.getPipes().size());
  int64_t edgeCount = recordCount / gridArea;
  // Dense device tables must remain O(records); a 1x1 N-device gather has N-1
  // records and requires one additional zero-count table entry.
  if (*deviceCount - 1 > recordCount) {
    return failure();
  }
  SmallVector<SmallVector<int64_t>> edgeBlocksByDevice(
      static_cast<std::size_t>(*deviceCount));
  DevicePipeNetParticipantPlan plan;
  plan.gridX = gridX;
  plan.gridArea = gridArea;
  plan.sourceDeviceIndices.reserve(edgeCount);
  plan.destinationDeviceIndices.reserve(edgeCount);
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
    plan.sourceDeviceIndices.push_back(sourceDeviceIndex);
    plan.destinationDeviceIndices.push_back(destinationDeviceIndex);
    int64_t endpointDeviceIndex =
        role == PipeRole::Source ? sourceDeviceIndex : destinationDeviceIndex;
    edgeBlocksByDevice[endpointDeviceIndex].push_back(edgeBlock);
  }

  plan.edgeOffsetsByDevice.push_back(0);
  for (ArrayRef<int64_t> edgeBlocks : edgeBlocksByDevice) {
    plan.edgeBlocks.append(edgeBlocks.begin(), edgeBlocks.end());
    plan.recordCountsByDevice.push_back(
        static_cast<int64_t>(edgeBlocks.size()));
    plan.edgeOffsetsByDevice.push_back(
        static_cast<int64_t>(plan.edgeBlocks.size()));
  }
  return plan;
}

} // namespace mlir::tt::ttl
