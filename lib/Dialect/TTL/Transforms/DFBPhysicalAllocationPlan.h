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
  ArrayRef<DFBPhysicalIndexAssignment> getAssignments() const {
    return assignments;
  }

  ArrayRef<DFBPhysicalAllocationDescriptor> getDescriptors() const {
    return descriptors;
  }

  ArrayRef<DFBKernelBaseIndexAssignment> getKernelBaseIndices() const {
    return kernelBaseIndices;
  }

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
  DFBPhysicalAllocationPlanner(Operation *operation, bool reuseUserDFBs,
                               AnalysisManager analysisManager);

  DFBPhysicalAllocationPlanner(Operation *operation, bool reuseUserDFBs,
                               AnalysisManager analysisManager,
                               const InterferenceGraphColoring &coloring);

  bool succeeded() const { return errorMessage.empty(); }

  StringRef getErrorMessage() const { return errorMessage; }

  Operation *getErrorOperation() const { return errorOperation; }

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
