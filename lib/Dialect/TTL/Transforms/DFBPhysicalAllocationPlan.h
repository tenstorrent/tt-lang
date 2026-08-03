// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBPHYSICALALLOCATIONPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBPHYSICALALLOCATIONPLAN_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

class InterferenceGraphColoring;

/// Physical index selected for one logical DFB.
struct DFBPhysicalIndexAssignment {
  int64_t logicalId = 0;
  int32_t physicalIndex = 0;
  Type type;
  SmallVector<BindCBOp> declarations;
  bool bounded = false;
};

/// Runtime allocation descriptor for one physical DFB.
struct DFBPhysicalAllocationDescriptor {
  int32_t physicalIndex = 0;
  int32_t numTiles = 0;
  Type elementType;
  int32_t pageSize = 0;
  int32_t blockCount = 0;
};

/// Final base CTA index for one kernel.
struct DFBKernelBaseIndexAssignment {
  func::FuncOp kernel;
  int32_t baseIndex = 0;
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

private:
  friend class DFBPhysicalAllocationPlanner;

  SmallVector<DFBPhysicalIndexAssignment> assignments;
  SmallVector<DFBPhysicalAllocationDescriptor> descriptors;
  SmallVector<DFBKernelBaseIndexAssignment> kernelBaseIndices;
  int32_t physicalDFBCount = 0;
};

/// Builds and validates a complete physical DFB allocation without mutation.
class DFBPhysicalAllocationPlanner {
public:
  /// Builds a plan using deterministic greedy first-fit coloring.
  ///
  /// `operation` must be a `ModuleOp`. Failures are recorded for the consuming
  /// pass to diagnose before any IR mutation.
  DFBPhysicalAllocationPlanner(Operation *operation, bool reuseUserDFBs,
                               AnalysisManager analysisManager);

  /// Builds a plan with an injected coloring strategy.
  ///
  /// This overload permits allocation-policy tests without changing liveness
  /// or interference construction.
  DFBPhysicalAllocationPlanner(Operation *operation, bool reuseUserDFBs,
                               AnalysisManager analysisManager,
                               const InterferenceGraphColoring &coloring);

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
