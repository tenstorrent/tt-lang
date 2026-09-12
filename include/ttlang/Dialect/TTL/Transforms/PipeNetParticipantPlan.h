// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETPARTICIPANTPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETPARTICIPANTPLAN_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttl {

/// Record slices for each row-major launch node in a local PipeNet.
///
/// Records within every node slice retain their order in the source attribute.
/// A record appears once in a slice even when multiple facts for the selected
/// role identify the same node.
struct LocalPipeNetParticipantPlan {
  int64_t gridX = 0;
  SmallVector<int64_t> recordOffsetsByNode;
  SmallVector<int64_t> recordCountsByNode;
  SmallVector<int64_t> recordIndices;
};

/// Check that the node endpoints selected by `records` and `role` are within
/// (`gridX`, `gridY`) and that indexing the relation fits signed 64-bit values.
/// Report invalid inputs through `emitError` when supplied.
LogicalResult validatePipeNetLaunchNodeRelation(
    PipeNetRecordsAttr records, PipeRole role, int64_t gridX, int64_t gridY,
    llvm::function_ref<InFlightDiagnostic()> emitError = {});

/// Group a verified local PipeNet's records by nodes with `role` in the grid.
FailureOr<LocalPipeNetParticipantPlan>
buildLocalPipeNetParticipantPlan(PipeNetRecordsAttr records, PipeRole role,
                                 int64_t gridX, int64_t gridY);

/// Record slices for each logical device in a grid-major device PipeNet.
///
/// Each transfer contributes one edge block containing one record per
/// row-major launch node. Duplicate transfers remain distinct edge blocks.
struct DevicePipeNetParticipantPlan {
  int64_t gridX = 0;
  int64_t gridArea = 0;
  SmallVector<int64_t> edgeOffsetsByDevice;
  SmallVector<int64_t> edgeBlocks;
  SmallVector<int64_t> recordCountsByDevice;
  SmallVector<int64_t> sourceDeviceIndices;
  SmallVector<int64_t> destinationDeviceIndices;
};

/// Build logical-device edge-block slices for `role`.
FailureOr<DevicePipeNetParticipantPlan>
buildDevicePipeNetParticipantPlan(PipeNetRecordsAttr records, PipeRole role,
                                  int64_t gridX, int64_t gridY);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETPARTICIPANTPLAN_H
