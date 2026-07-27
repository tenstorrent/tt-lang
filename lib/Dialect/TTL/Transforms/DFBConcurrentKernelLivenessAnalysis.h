// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H

//===----------------------------------------------------------------------===//
// Concurrent Kernel DFB Liveness Analysis
//===----------------------------------------------------------------------===//
//
// These analyses resolve module-wide logical DFB identity and assign one
// physical index to every logical DFB without modifying IR. The finalization
// pass materializes the returned identities and indices only after analysis
// succeeds, so an error cannot leave a partially rewritten module.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

class InterferenceGraphColoring;

/// Resolved logical identity for one `ttl.bind_cb` declaration.
struct DFBLogicalIdentityAssignment {
  /// DFB declaration associated with `logicalId`.
  BindCBOp declaration;

  /// Module-wide identity shared by declarations of the same logical DFB.
  int64_t logicalId = 0;
};

/// Read-only module analysis that assigns a module-wide logical ID to every DFB
/// declaration. User declarations must carry `dfb_id`. Compiler-created
/// declarations receive deterministic IDs after all user IDs. A failed
/// analysis records the operation that violates the identity contract.
class DFBLogicalIdentityAnalysis {
public:
  explicit DFBLogicalIdentityAnalysis(Operation *operation);

  /// Returns true when every declaration has a valid logical identity.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed analysis.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the analysis diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns each DFB declaration and its resolved logical ID.
  ArrayRef<DFBLogicalIdentityAssignment> getAssignments() const {
    return assignments;
  }

  /// Returns the resolved logical identity for `bindOp`.
  int64_t getLogicalId(BindCBOp bindOp) const;

private:
  SmallVector<DFBLogicalIdentityAssignment> assignments;
  DenseMap<Operation *, int64_t> identities;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

/// Physical allocation selected for one logical DFB.
struct DFBPhysicalIndexAssignment {
  /// Module-wide identity of the logical DFB.
  int64_t logicalId = 0;

  /// Hardware DFB index assigned to this logical DFB.
  int32_t physicalIndex = 0;

  /// Common type of every declaration in `declarations`.
  Type type;

  /// All declarations that refer to this logical DFB.
  SmallVector<BindCBOp> declarations;

  /// Whether the analysis proved a finite lifetime for this logical DFB.
  bool bounded = false;
};

/// Read-only module analysis that assigns a physical index to every logical DFB
/// from happens-before relations across concurrent kernels. DFBs share an index
/// only when their storage and kernel participants match and one proven
/// lifetime ends before the other begins.
class DFBConcurrentKernelLivenessAnalysis {
public:
  /// Assigns physical DFB indices using deterministic greedy first-fit
  /// coloring.
  explicit DFBConcurrentKernelLivenessAnalysis(Operation *operation);

  /// Assigns one physical DFB index per logical DFB using `coloring`.
  DFBConcurrentKernelLivenessAnalysis(
      Operation *operation, const InterferenceGraphColoring &coloring);

  /// Returns true when every logical DFB was assigned a physical index.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed analysis.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the analysis diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns the physical index and allocation metadata for each logical DFB.
  ArrayRef<DFBPhysicalIndexAssignment> getAssignments() const {
    return assignments;
  }

  /// Returns the number of distinct physical DFB indices assigned.
  int32_t getPhysicalSlotCount() const { return physicalSlotCount; }

private:
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalSlotCount = 0;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBCONCURRENTKERNELLIVENESSANALYSIS_H
