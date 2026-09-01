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

FailureOr<LocalPipeNetParticipantPlan>
buildLocalPipeNetParticipantPlan(PipeNetRecordsAttr records, PipeRole role,
                                 int64_t gridX, int64_t gridY) {
  if (gridX <= 0 || gridY <= 0) {
    return failure();
  }
  std::optional<int64_t> maybeGridArea = llvm::checkedMul(gridX, gridY);
  if (!maybeGridArea ||
      records.getPipes().size() >
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return failure();
  }

  SmallVector<SmallVector<int64_t>> recordsByNode(
      static_cast<std::size_t>(*maybeGridArea));
  for (auto [recordIndex, record] : llvm::enumerate(records.getPipes())) {
    if (record.getDeviceTransfer()) {
      return failure();
    }
    llvm::SmallDenseSet<int64_t, 4> participantIndices;
    for (const PipeRecordRoleFacts &facts :
         getPipeRecordRoleFacts(record, role)) {
      if (facts.device || facts.minX < 0 || facts.minY < 0 ||
          facts.minX > facts.maxX || facts.minY > facts.maxY ||
          facts.maxX >= gridX || facts.maxY >= gridY) {
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

} // namespace mlir::tt::ttl
