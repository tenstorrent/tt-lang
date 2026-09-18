// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBLOGICALIDENTITYANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBLOGICALIDENTITYANALYSIS_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

/// Resolved logical identity for one `ttl.bind_cb` declaration.
struct DFBLogicalIdentityAssignment {
  /// DFB declaration associated with `logicalId`.
  BindCBOp declaration;

  /// Module-wide identity shared by declarations of the same logical DFB.
  int64_t logicalId = 0;

  /// Optional compiler-verified physical-allocation identity.
  DFBAllocationGroupAttr allocationGroup;
};

/// Resolves the module-wide logical identity of every DFB declaration.
///
/// User declarations must carry `dfb_id`. Compiler-created declarations
/// receive IDs in module traversal order after all explicit IDs. The analysis
/// does not modify IR, so callers can validate the complete identity assignment
/// before materializing generated IDs.
class DFBLogicalIdentityAnalysis {
public:
  /// Resolves every DFB declaration nested under the given `ModuleOp`.
  ///
  /// `operation` must be a `ModuleOp`. Invalid declarations are recorded as a
  /// diagnostic operation and message rather than emitted by the analysis.
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

  /// Returns the logical ID resolved for `declaration`.
  ///
  /// The declaration must belong to a successful analysis result.
  int64_t getLogicalId(BindCBOp declaration) const;

  /// Returns the logical ID for the declaration reached from `dfb`.
  ///
  /// Returns failure when `dfb` does not resolve to an analyzed declaration.
  FailureOr<int64_t> getLogicalId(Value dfb) const;

private:
  SmallVector<DFBLogicalIdentityAssignment> assignments;
  DenseMap<Operation *, int64_t> logicalIds;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBLOGICALIDENTITYANALYSIS_H
