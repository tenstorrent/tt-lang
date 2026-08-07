// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TARGETINFO_H
#define TTLANG_TARGET_TARGETINFO_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <cstdint>

namespace mlir::tt {

/// Module attribute identifying the compilation target architecture.
constexpr llvm::StringLiteral kTargetArchAttrName("ttl.target_arch");

/// tt-metal `NOC_MAX_BURST_SIZE` values, measured in bytes.
///
/// Architecture headers define these as
/// `NOC_MAX_BURST_WORDS * NOC_WORD_BYTES`: 8192 on Wormhole B0 and 16384 on
/// Blackhole.
inline constexpr int64_t kWormholeB0NocMaxBurstBytes = 8192;
inline constexpr int64_t kBlackholeNocMaxBurstBytes = 16384;
inline constexpr int64_t kDefaultNocMaxBurstBytes = kWormholeB0NocMaxBurstBytes;

/// Return whether `op` belongs to a module targeting `arch`.
inline bool hasTargetArch(Operation *op, ttcore::Arch arch) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return false;
  }

  auto targetArch =
      moduleOp->getAttrOfType<ttcore::ArchAttr>(kTargetArchAttrName);
  return targetArch && targetArch.getValue() == arch;
}

/// Return whether `op` targets Blackhole.
inline bool isBlackholeTarget(Operation *op) {
  return hasTargetArch(op, ttcore::Arch::Blackhole);
}

/// Return whether `op` targets Wormhole B0.
inline bool isWormholeB0Target(Operation *op) {
  return hasTargetArch(op, ttcore::Arch::WormholeB0);
}

/// Return the maximum one-packet NoC transfer size for the module target.
///
/// Compile-only IR may omit `ttl.target_arch`; in that case this uses the
/// minimum supported Wormhole B0/Blackhole value.
inline int64_t getTargetNocMaxBurstBytes(Operation *op) {
  if (isBlackholeTarget(op)) {
    return kBlackholeNocMaxBurstBytes;
  }
  return kDefaultNocMaxBurstBytes;
}

} // namespace mlir::tt

#endif
