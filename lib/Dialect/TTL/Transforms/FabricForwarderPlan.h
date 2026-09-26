//===- FabricForwarderPlan.h - Fabric forwarder planning ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file declares immutable worker-to-forwarder aggregation decisions for
// routing-plane PipeNet transfers.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_FABRICFORWARDERPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_FABRICFORWARDERPLAN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>

namespace mlir::tt::ttl {

class PipeTransferIndex;
class PipeGraph;
struct FabricRoutePlan;
struct PipeForeachLoweringInfo;

/// Record-indexed aggregation decisions for one pipe protocol operation.
/// Runtime record indices select a group and slot. Group-and-slot tables use
/// `groupIndex * maximumGroupSize + slot` and describe each active member.
class FabricForwarderOperationPlan {
public:
  ArrayRef<int64_t> getGroupIndexByRecord() const { return groupIndexByRecord; }
  ArrayRef<int64_t> getForwarderXByGroup() const { return forwarderXByGroup; }
  ArrayRef<int64_t> getForwarderYByGroup() const { return forwarderYByGroup; }
  ArrayRef<int64_t> getSlotByRecord() const { return slotByRecord; }
  ArrayRef<int64_t> getGroupSizeByGroup() const { return groupSizeByGroup; }
  ArrayRef<int64_t> getGroupRecordsByGroupAndSlot() const {
    return groupRecordsByGroupAndSlot;
  }
  ArrayRef<int64_t> getWorkerXByGroupAndSlot() const {
    return workerXByGroupAndSlot;
  }
  ArrayRef<int64_t> getWorkerYByGroupAndSlot() const {
    return workerYByGroupAndSlot;
  }
  ArrayRef<int64_t> getPeerXByGroupAndSlot() const {
    return peerXByGroupAndSlot;
  }
  ArrayRef<int64_t> getPeerYByGroupAndSlot() const {
    return peerYByGroupAndSlot;
  }
  int64_t getMaximumGroupSize() const { return maximumGroupSize; }
  int64_t getPayloadSizeBytes() const { return payloadSizeBytes; }
  int64_t getPayloadStrideBytes() const { return payloadStrideBytes; }
  int64_t getScratchByteOffset() const { return scratchByteOffset; }
  int64_t getArrivalCounterByteOffset() const {
    return arrivalCounterByteOffset;
  }
  int64_t getCompletionCounterByteOffset() const {
    return completionCounterByteOffset;
  }

private:
  friend FailureOr<class FabricForwarderPlan>
  buildFabricForwarderPlan(ModuleOp, const PipeTransferIndex &,
                           const PipeForeachLoweringInfo &, const PipeGraph &,
                           const FabricRoutePlan &);

  /// Forwarder group selected by each original PipeNet record.
  SmallVector<int64_t> groupIndexByRecord;
  /// Logical coordinates of the worker that forwards each group.
  SmallVector<int64_t> forwarderXByGroup;
  SmallVector<int64_t> forwarderYByGroup;
  /// Payload position assigned to each original PipeNet record.
  SmallVector<int64_t> slotByRecord;
  /// Number of active members in each forwarder group.
  SmallVector<int64_t> groupSizeByGroup;
  /// Original PipeNet record selected by each active group member.
  SmallVector<int64_t> groupRecordsByGroupAndSlot;
  /// Local worker coordinates for each active group member.
  SmallVector<int64_t> workerXByGroupAndSlot;
  SmallVector<int64_t> workerYByGroupAndSlot;
  /// Remote endpoint coordinates for each active group member.
  SmallVector<int64_t> peerXByGroupAndSlot;
  SmallVector<int64_t> peerYByGroupAndSlot;
  /// Largest member count; also the stride of each group-and-slot table.
  int64_t maximumGroupSize = 0;
  /// Bytes transferred by one sender record; zero for receiver operations.
  int64_t payloadSizeBytes = 0;
  /// Aligned scratch bytes reserved for each sender payload slot.
  int64_t payloadStrideBytes = 0;
  /// Start of this operation's scratch segment. Sender payload slots precede
  /// the counters.
  int64_t scratchByteOffset = 0;
  /// Operation-specific offsets keep independent cumulative counts separate
  /// and prevent concurrent kernel functions from sharing counter state.
  int64_t arrivalCounterByteOffset = 0;
  int64_t completionCounterByteOffset = 0;
};

/// Immutable aggregation and resource decisions for one module.
class FabricForwarderPlan {
public:
  const FabricForwarderOperationPlan *lookup(Operation *operation) const;
  const llvm::MapVector<Operation *, FabricForwarderOperationPlan> &
  getOperations() const {
    return operations;
  }

  bool empty() const { return operations.empty(); }
  int64_t getSramScratchBytes() const { return sramScratchBytes; }
  int64_t getSramScratchBaseOffset() const { return sramScratchBaseOffset; }

  /// Place the plan's relative scratch offsets in the combined allocation.
  void setSramScratchBaseOffset(int64_t byteOffset) {
    sramScratchBaseOffset = byteOffset;
  }

private:
  friend FailureOr<FabricForwarderPlan>
  buildFabricForwarderPlan(ModuleOp, const PipeTransferIndex &,
                           const PipeForeachLoweringInfo &, const PipeGraph &,
                           const FabricRoutePlan &);
  friend void applyFabricForwarderRoutes(const FabricForwarderPlan &,
                                         FabricRoutePlan &);

  /// Identifies one function-local route whose direct workers are replaced by
  /// the planned connection owners.
  struct RouteSourcePlan {
    /// Kernel containing the route whose direct owners are replaced.
    func::FuncOp function;
    /// Logical device direction for the physical connection.
    DeviceRefAttr localDevice;
    DeviceRefAttr remoteDevice;
    /// Connection index among routes originating on `localDevice`.
    std::size_t routeIndex;
    /// Workers that own the route's physical fabric connections.
    SmallVector<LaunchNodeCoord> sourceNodes;
  };

  /// Aggregation decisions indexed by the protocol operation they replace.
  llvm::MapVector<Operation *, FabricForwarderOperationPlan> operations;
  /// Physical fabric routes whose owners become the planned forwarders.
  SmallVector<RouteSourcePlan> routes;
  /// Total bytes reserved for forwarder payload slots and counters.
  int64_t sramScratchBytes = 0;
  /// Start of this plan's segment in the combined compiler scratch allocation.
  int64_t sramScratchBaseOffset = 0;
};

/// Plan Blackhole worker aggregation for record-selected fabric operations
/// with exact worker domains and fixed execution counts. Ineligible operations
/// retain their direct fabric routes.
FailureOr<FabricForwarderPlan> buildFabricForwarderPlan(
    ModuleOp module, const PipeTransferIndex &transferIndex,
    const PipeForeachLoweringInfo &foreachLoweringInfo,
    const PipeGraph &pipeGraph, const FabricRoutePlan &fabricRoutePlan);

/// Replace direct worker manager locations with the planned forwarders.
void applyFabricForwarderRoutes(const FabricForwarderPlan &forwarderPlan,
                                FabricRoutePlan &fabricRoutePlan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_FABRICFORWARDERPLAN_H
