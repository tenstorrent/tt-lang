// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBPHYSICALALLOCATIONPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBPHYSICALALLOCATIONPLAN_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

class DFBPhysicalConflictModelBuilder;

/// Physical index selected for one logical DFB.
struct DFBPhysicalIndexAssignment {
  int64_t logicalId = 0;
  int32_t physicalIndex = 0;
  Type type;
  TensorBackingAttr tensorBacking;
  DFBAllocationGroupAttr allocationGroup;
  LaunchNodeDomain launchDomain;
  SmallVector<BindCBOp> declarations;
  bool bounded = false;
};

/// Storage used by one physical DFB on an exact launch-node domain.
struct DFBPhysicalStorageSegment {
  LaunchNodeDomain launchDomain;
  TensorBackingAttr tensorBacking;
};

/// Runtime configuration for one physical DFB during one configuration epoch.
struct DFBConfigurationEpochDescriptor {
  std::optional<int64_t> entryReconfigurationOrdinal;
  int32_t numTiles = 0;
  Type elementType;
  int32_t pageSize = 0;
  int32_t blockCount = 0;
  SmallVector<DFBPhysicalStorageSegment> storageSegments;
};

/// Runtime allocation descriptor for one physical DFB.
struct DFBPhysicalAllocationDescriptor {
  int32_t physicalIndex = 0;
  int32_t numTiles = 0;
  Type elementType;
  int32_t pageSize = 0;
  int32_t blockCount = 0;
  /// Exact union of nodes that access this index. An unknown domain requires
  /// conservative whole-grid runtime allocation.
  LaunchNodeDomain allocationDomain = LaunchNodeDomain::unknown();
  SmallVector<DFBPhysicalStorageSegment> storageSegments;
  SmallVector<DFBConfigurationEpochDescriptor> epochConfigurations;
};

using DFBPhysicalAllocationDescriptorList =
    SmallVector<DFBPhysicalAllocationDescriptor, 0>;

/// Final base CTA index for one kernel.
struct DFBKernelBaseIndexAssignment {
  func::FuncOp kernel;
  int32_t baseIndex = 0;
};

/// Static kernel-configuration evidence that forbids one logical DFB alias.
struct DFBStaticConfigurationConflict {
  int64_t lhsLogicalId = 0;
  int64_t rhsLogicalId = 0;
  Operation *lhsOperation = nullptr;
  Operation *rhsOperation = nullptr;
};

/// Semantic reason that two logical DFBs cannot share one physical index.
enum class DFBConflictReason {
  DescriptorMismatch,
  StorageMismatch,
  UnknownLaunchNodeDomain,
  AccessCompletionNotProven,
  TransactionMismatch,
  PointerOwnerMismatch,
  ConcurrentLifetime,
  ResetDomainWrite,
  StaticConfigurationMismatch,
};

/// Returns the stable diagnostic spelling for one conflict reason.
StringRef getDFBConflictReasonName(DFBConflictReason reason);

/// Runtime handoff property accepted without compiler proof for one explicit
/// allocation group.
enum class DFBAllocationGroupAssumptionReason {
  UnknownLaunchNodeDomain,
  AccessCompletionNotProven,
  PointerOwnerMismatch,
  ConcurrentLifetime,
  UnprovenCursorOrder,
  EpochReset,
};

/// Returns the stable diagnostic spelling for an assumed group property.
StringRef getDFBAllocationGroupAssumptionReasonName(
    DFBAllocationGroupAssumptionReason reason);

/// One allocation-group property accepted without compiler proof.
struct DFBAllocationGroupAssumption {
  DFBAllocationGroupAssumptionReason reason =
      DFBAllocationGroupAssumptionReason::ConcurrentLifetime;
  int64_t lhsLogicalId = 0;
  std::optional<int64_t> rhsLogicalId;
};

/// Audit record for one allocation group accepted by unsafe policy.
struct DFBAssumedAllocationGroup {
  DFBAllocationGroupAttr allocationGroup;
  SmallVector<int64_t> logicalIds;
  SmallVector<DFBAllocationGroupAssumption> assumptions;
  Operation *operation = nullptr;
};

/// Source evidence that explains why one logical DFB pair cannot share.
struct DFBConflictEvidence {
  unsigned lhsLogicalIndex = 0;
  unsigned rhsLogicalIndex = 0;
  int64_t lhsLogicalId = 0;
  int64_t rhsLogicalId = 0;
  DFBConflictReason reason = DFBConflictReason::ConcurrentLifetime;
  std::optional<LaunchNodeCoord> node;
  Operation *lhsOperation = nullptr;
  Operation *rhsOperation = nullptr;
};

/// Immutable complete conflict relation used by every allocation policy.
class DFBPhysicalConflictModel {
public:
  bool conflicts(unsigned lhsLogicalIndex, unsigned rhsLogicalIndex) const {
    assert(lhsLogicalIndex < adjacency.size() &&
           rhsLogicalIndex < adjacency.size());
    return adjacency[lhsLogicalIndex].test(rhsLogicalIndex);
  }

  ArrayRef<DFBConflictEvidence> getEvidence() const { return evidence; }

private:
  friend class DFBPhysicalConflictModelBuilder;
  friend class DFBPhysicalAllocationPlanner;

  SmallVector<llvm::BitVector> adjacency;
  SmallVector<DFBConflictEvidence> evidence;
};

/// Immutable assignments, runtime descriptors, and kernel base-index updates.
class DFBPhysicalAllocationPlan {
public:
  /// Returns one physical assignment per logical DFB.
  ArrayRef<DFBPhysicalIndexAssignment> getAssignments() const {
    return assignments;
  }

  /// Returns one runtime descriptor per unique physical index.
  ArrayRef<DFBPhysicalAllocationDescriptor> getDescriptors() const {
    return descriptors;
  }

  /// Returns deferred kernel attribute updates.
  ArrayRef<DFBKernelBaseIndexAssignment> getKernelBaseIndices() const {
    return kernelBaseIndices;
  }

  /// Returns the size of the dense physical-index range.
  int32_t getPhysicalDFBCount() const { return physicalDFBCount; }

  /// Returns boundary identifiers in proved dynamic execution order.
  ArrayRef<int64_t> getReconfigurationBoundaryOrdinals() const {
    return reconfigurationBoundaryOrdinals;
  }

  /// Returns the complete typed conflict relation used for allocation.
  const DFBPhysicalConflictModel &getConflictModel() const {
    return conflictModel;
  }

  /// Returns allocation groups whose runtime handoff was accepted without
  /// compiler proof.
  ArrayRef<DFBAssumedAllocationGroup> getAssumedAllocationGroups() const {
    return assumedAllocationGroups;
  }

private:
  friend class DFBPhysicalAllocationPlanner;

  SmallVector<DFBPhysicalIndexAssignment> assignments;
  DFBPhysicalAllocationDescriptorList descriptors;
  SmallVector<DFBKernelBaseIndexAssignment> kernelBaseIndices;
  SmallVector<DFBAssumedAllocationGroup> assumedAllocationGroups;
  SmallVector<int64_t> reconfigurationBoundaryOrdinals;
  DFBPhysicalConflictModel conflictModel;
  int32_t physicalDFBCount = 0;
};

/// Builds and validates a complete physical DFB allocation without mutation.
class DFBPhysicalAllocationPlanner {
public:
  /// Builds a plan using deterministic first-fit and bounded exhaustive checks.
  ///
  /// `operation` must be a `ModuleOp`. Failures are recorded for the consuming
  /// pass to diagnose before any IR mutation. A successful plan has verified
  /// sharing, dense indices, runtime descriptors, and hardware capacity.
  /// Unknown lifetime facts add conflicts and can cause rejection, but cannot
  /// permit unsafe sharing.
  DFBPhysicalAllocationPlanner(
      Operation *operation, bool reuseUserDFBs,
      bool unsafeAssumeAllocationGroups,
      std::uint64_t exactColoringSearchStateLimit,
      std::optional<uint64_t> l1BudgetOverride,
      ArrayRef<DFBStaticConfigurationConflict> staticConfigurationConflicts,
      AnalysisManager analysisManager);

  /// Returns true when the complete allocation plan is valid.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed plan.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the plan diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns the validated immutable allocation plan.
  const DFBPhysicalAllocationPlan &getPlan() const {
    assert(succeeded() && "failed planning has no valid allocation plan");
    return plan;
  }

private:
  DFBPhysicalAllocationPlan plan;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBPHYSICALALLOCATIONPLAN_H
