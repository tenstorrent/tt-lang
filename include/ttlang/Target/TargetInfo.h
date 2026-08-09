// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TARGETINFO_H
#define TTLANG_TARGET_TARGETINFO_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
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

/// Return the maximum one-packet NoC transfer size for the module target.
///
/// Compile-only IR may omit `ttl.target_arch`; in that case this uses the
/// minimum supported Wormhole B0/Blackhole value.
inline int64_t getTargetNocMaxBurstBytes(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  auto targetArch =
      moduleOp ? moduleOp->getAttrOfType<ttcore::ArchAttr>(kTargetArchAttrName)
               : nullptr;
  if (targetArch && targetArch.getValue() == ttcore::Arch::Blackhole) {
    return kBlackholeNocMaxBurstBytes;
  }
  return kDefaultNocMaxBurstBytes;
}

} // namespace mlir::tt

#endif
