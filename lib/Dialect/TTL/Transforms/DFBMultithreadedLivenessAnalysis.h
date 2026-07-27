// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H

//===----------------------------------------------------------------------===//
// Multithreaded DFB Liveness Analysis
//===----------------------------------------------------------------------===//
//
// These analyses resolve module-wide logical DFB identity and compute a
// physical index assignment without modifying IR. The finalization pass
// materializes the returned identities and indices only after analysis
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

/// Resolved logical identity for one `ttl.bind_cb` declaration.
struct DFBLogicalIdentityAssignment {
  /// The declaration to annotate after analysis succeeds.
  BindCBOp declaration;

  /// Module-wide identity shared by declarations of the same logical DFB.
  int64_t logicalId = 0;
};

/// Read-only module analysis that resolves every DFB declaration to a stable
/// logical identity. User declarations must carry `dfb_id`. Compiler-created
/// declarations receive deterministic identities after all explicit IDs. A
/// failed analysis records the operation that violates the identity contract.
class DFBLogicalIdentityAnalysis {
public:
  explicit DFBLogicalIdentityAnalysis(Operation *operation);

  /// Returns true when every declaration has a valid logical identity.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed analysis.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the analysis diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns one resolved identity assignment per DFB declaration.
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

  /// Whether the analysis proved finite earliest and terminal frontiers.
  bool bounded = false;
};

/// Read-only module analysis that computes a conservative physical DFB index
/// assignment from multithreaded happens-before relations. DFBs share an index
/// only when their storage and thread participants match and one proven
/// lifetime ends before the other begins.
class DFBMultithreadedLivenessAnalysis {
public:
  explicit DFBMultithreadedLivenessAnalysis(Operation *operation);

  /// Returns true when a complete physical assignment was computed.
  bool succeeded() const { return errorMessage.empty(); }

  /// Returns the diagnostic message recorded for a failed analysis.
  StringRef getErrorMessage() const { return errorMessage; }

  /// Returns the operation to which the analysis diagnostic should attach.
  Operation *getErrorOperation() const { return errorOperation; }

  /// Returns one physical assignment per logical DFB.
  ArrayRef<DFBPhysicalIndexAssignment> getAssignments() const {
    return assignments;
  }

  /// Returns the number of physical indices used by `getAssignments()`.
  int32_t getPhysicalSlotCount() const { return physicalSlotCount; }

private:
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalSlotCount = 0;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H
