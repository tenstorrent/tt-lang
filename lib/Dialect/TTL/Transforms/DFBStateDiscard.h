// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBSTATEDISCARD_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBSTATEDISCARD_H

#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <set>

namespace mlir::tt::ttl {

/// Returns the finalized physical descriptor index of `dfb`.
///
/// Emits an error on `op` and fails when the index is unresolved or outside
/// the target's physical DFB index range.
FailureOr<int32_t> getValidatedDFBIndex(Value dfb, Operation *op);

/// Returns one bit per physical DFB index that a finalized `ttl.bind_cb` in
/// `module` declares.
///
/// Emits an error on the declaration and fails when an index is unresolved or
/// outside the target's physical DFB index range.
FailureOr<uint64_t> getAllocatedDFBMask(ModuleOp module);

/// Returns the physical DFB interfaces whose protocol state the synchronized
/// reset `reset` (`ttl.reset_dfbs` or `ttl.reset_all_dfbs`) restores.
///
/// `allocatedMask` is the result of `getAllocatedDFBMask`. Reset lowering and
/// DFB lifecycle verification both derive reset targets from this function,
/// so the verified reset intervals match the interfaces the runtime resets.
FailureOr<uint64_t> getSynchronizedResetDFBMask(Operation *reset,
                                                uint64_t allocatedMask);

/// Physical DFB descriptors that each reconfiguration boundary installs, read
/// from `ttl.dfb_reconfiguration_plan`.
///
/// The runtime installs a descriptor at a boundary only when the finalized
/// plan gives it a configuration entering there, and installation resets the
/// descriptor's pointers and counters on the nodes that configuration covers.
/// A descriptor the plan does not install keeps its protocol state across the
/// boundary, including a boundary that declares `discard_dfb_state`.
class DFBReconfigurationInstalls {
public:
  /// Parses the finalized plan of `module`. A module without the plan
  /// installs nothing. Emits an error on `module` and fails for malformed
  /// metadata.
  static FailureOr<DFBReconfigurationInstalls> build(ModuleOp module);

  /// Returns the physical DFB indices installed at boundary `ordinal` on
  /// `node`.
  uint64_t getInstalledDFBMask(int64_t ordinal, LaunchNodeCoord node) const;

  /// Returns the physical DFB indices installed at boundary `ordinal` on at
  /// least one node.
  uint64_t getInstalledDFBMask(int64_t ordinal) const;

private:
  struct Install {
    int32_t physicalIndex = 0;
    /// Nodes the configuration covers; absent when it covers every node.
    std::optional<std::set<LaunchNodeCoord>> nodes;
  };

  llvm::DenseMap<int64_t, SmallVector<Install>> installsByOrdinal;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBSTATEDISCARD_H
