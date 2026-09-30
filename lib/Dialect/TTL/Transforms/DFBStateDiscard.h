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

namespace mlir::tt::ttl {

/// Returns the finalized physical descriptor index of `dfb`.
///
/// Emits an error on `op` and fails when the index is unresolved or outside
/// the target's physical DFB index range.
FailureOr<int32_t> getValidatedDFBIndex(Value dfb, Operation *op);

/// Returns one bit per physical DFB index that a finalized `ttl.bind_cb` in
/// `module` declares; fails as `getValidatedDFBIndex` does.
FailureOr<uint64_t> getAllocatedDFBMask(ModuleOp module);

/// Returns the physical DFB interfaces whose protocol state the synchronized
/// reset `reset` (`ttl.reset_dfbs` or `ttl.reset_all_dfbs`) restores: the
/// listed DFBs, or every allocated DFB except the preserved ones. A listed or
/// preserved DFB stands for every member of its allocation group, which shares
/// one L1 allocation.
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
  /// Reads the plan `ttl-finalize-dfb-indices` attaches to a module with
  /// reconfigurations; a module without reconfigurations installs nothing. A
  /// configuration without storage segments covers every node of
  /// `launchDomain`.
  static DFBReconfigurationInstalls build(ModuleOp module,
                                          const LaunchNodeDomain &launchDomain);

  /// Returns the physical DFB indices installed at boundary `ordinal` on
  /// `node`, or on at least one node when `node` is absent.
  uint64_t getInstalledDFBMask(int64_t ordinal,
                               std::optional<LaunchNodeCoord> node) const;

private:
  struct Install {
    int32_t physicalIndex = 0;
    LaunchNodeDomain nodes;
  };

  llvm::DenseMap<int64_t, SmallVector<Install>> installsByOrdinal;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBSTATEDISCARD_H
