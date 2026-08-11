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
inline constexpr int32_t kWormholeB0MaxDFBIndices = 32;
inline constexpr int32_t kBlackholeMaxDFBIndices = 64;
inline constexpr int32_t kQuasarMaxDFBIndices = 32;
inline constexpr int32_t kMissingTargetMaxDFBIndices = 32;

namespace detail {

inline ttcore::ArchAttr getTargetArch(Operation *op) {
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  return moduleOp
             ? moduleOp->getAttrOfType<ttcore::ArchAttr>(kTargetArchAttrName)
             : nullptr;
}

struct TargetDFBCapacity {
  int32_t maxIndices;
  llvm::StringRef targetName;
};

inline TargetDFBCapacity getTargetDFBCapacity(Operation *op) {
  ttcore::ArchAttr targetArch = getTargetArch(op);
  if (!targetArch) {
    return {kMissingTargetMaxDFBIndices, {}};
  }
  switch (targetArch.getValue()) {
  case ttcore::Arch::WormholeB0:
    return {kWormholeB0MaxDFBIndices, "Wormhole B0"};
  case ttcore::Arch::Blackhole:
    return {kBlackholeMaxDFBIndices, "Blackhole"};
  case ttcore::Arch::Quasar:
    return {kQuasarMaxDFBIndices, "Quasar"};
  }
}

} // namespace detail

/// Return the physical DFB-index capacity for the module target.
inline int32_t getTargetMaxDFBIndices(Operation *op) {
  return detail::getTargetDFBCapacity(op).maxIndices;
}

/// Return target and limit text for DFB-capacity diagnostics.
inline std::string getTargetDFBCapacityDescription(Operation *op) {
  detail::TargetDFBCapacity capacity = detail::getTargetDFBCapacity(op);
  if (capacity.targetName.empty()) {
    return "the conservative " + std::to_string(capacity.maxIndices) +
           "-DFB-index capacity used when target metadata is absent";
  }
  return "the " + std::to_string(capacity.maxIndices) + "-DFB-index " +
         capacity.targetName.str() + " target capacity";
}

/// Return the maximum one-packet NoC transfer size for the module target.
///
/// Compile-only IR may omit `ttl.target_arch`; in that case this uses the
/// minimum supported Wormhole B0/Blackhole value.
inline int64_t getTargetNocMaxBurstBytes(Operation *op) {
  ttcore::ArchAttr targetArch = detail::getTargetArch(op);
  if (targetArch && targetArch.getValue() == ttcore::Arch::Blackhole) {
    return kBlackholeNocMaxBurstBytes;
  }
  return kDefaultNocMaxBurstBytes;
}

} // namespace mlir::tt

#endif
