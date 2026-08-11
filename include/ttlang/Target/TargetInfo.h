// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TARGETINFO_H
#define TTLANG_TARGET_TARGETINFO_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

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

/// Physical DFB-index capacities by compilation target.
inline constexpr int32_t kWormholeB0DFBIndexCapacity = 32;
inline constexpr int32_t kBlackholeDFBIndexCapacity = 64;
inline constexpr int32_t kQuasarDFBIndexCapacity = 32;
inline constexpr int32_t kMissingTargetDFBIndexCapacity = 32;

namespace target_info_detail {

inline ttcore::ArchAttr getTargetArchAttr(Operation *op) {
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  return moduleOp
             ? moduleOp->getAttrOfType<ttcore::ArchAttr>(kTargetArchAttrName)
             : nullptr;
}

struct TargetDFBIndexCapacity {
  int32_t indexCount;
  llvm::StringRef targetName;
};

inline TargetDFBIndexCapacity getTargetDFBIndexCapacity(Operation *op) {
  ttcore::ArchAttr targetArch = getTargetArchAttr(op);
  if (!targetArch) {
    return {kMissingTargetDFBIndexCapacity, {}};
  }
  switch (targetArch.getValue()) {
  case ttcore::Arch::WormholeB0:
    return {kWormholeB0DFBIndexCapacity, "Wormhole B0"};
  case ttcore::Arch::Blackhole:
    return {kBlackholeDFBIndexCapacity, "Blackhole"};
  case ttcore::Arch::Quasar:
    return {kQuasarDFBIndexCapacity, "Quasar"};
  }
  llvm_unreachable("unhandled ttcore::Arch");
}

} // namespace target_info_detail

/// Return the physical DFB-index capacity for the module target.
inline int32_t getTargetMaxDFBIndices(Operation *op) {
  return target_info_detail::getTargetDFBIndexCapacity(op).indexCount;
}

/// Return target and limit text for DFB-capacity diagnostics.
inline std::string getTargetDFBIndexCapacityDescription(Operation *op) {
  target_info_detail::TargetDFBIndexCapacity capacity =
      target_info_detail::getTargetDFBIndexCapacity(op);
  if (capacity.targetName.empty()) {
    return "the conservative " + std::to_string(capacity.indexCount) +
           "-DFB-index capacity used when target metadata is absent";
  }
  return "the " + std::to_string(capacity.indexCount) + "-DFB-index " +
         capacity.targetName.str() + " target capacity";
}

/// Return the maximum one-packet NoC transfer size for the module target.
///
/// Compile-only IR may omit `ttl.target_arch`; in that case this uses the
/// minimum supported Wormhole B0/Blackhole value.
inline int64_t getTargetNocMaxBurstBytes(Operation *op) {
  ttcore::ArchAttr targetArch = target_info_detail::getTargetArchAttr(op);
  if (targetArch && targetArch.getValue() == ttcore::Arch::Blackhole) {
    return kBlackholeNocMaxBurstBytes;
  }
  return kDefaultNocMaxBurstBytes;
}

} // namespace mlir::tt

#endif
