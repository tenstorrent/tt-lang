// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TARGETINFO_H
#define TTLANG_TARGET_TARGETINFO_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <optional>
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

/// Physical DFB-index capacity selected for a compilation target.
struct TargetDFBIndexCapacity {
  int32_t indexCount;
  llvm::StringRef targetName;

  bool contains(int64_t index) const {
    return index >= 0 && index < indexCount;
  }

  std::string getDescription() const {
    if (targetName.empty()) {
      return "the conservative " + std::to_string(indexCount) +
             "-DFB-index capacity used when target metadata is absent";
    }
    return "the " + std::to_string(indexCount) + "-DFB-index " +
           targetName.str() + " target capacity";
  }
};

namespace target_info_detail {

inline FailureOr<std::optional<ttcore::Arch>>
getDeviceArch(ModuleOp module, std::string &failureReason) {
  Attribute rawSystemDesc = module->getAttr(ttcore::SystemDescAttr::name);
  auto systemDesc = dyn_cast_or_null<ttcore::SystemDescAttr>(rawSystemDesc);
  if (rawSystemDesc && !systemDesc) {
    failureReason =
        "ttcore.system_desc must be a #ttcore.system_desc attribute";
    return failure();
  }
  auto device = ttcore::lookupDeviceOp(module, ttcore::getDefaultDeviceName());
  if (!systemDesc || !device) {
    return std::optional<ttcore::Arch>();
  }

  ArrayRef<unsigned> chipIds = device.getDeviceAttr().getChipIds();
  if (chipIds.empty()) {
    failureReason = "default device has no selected chip";
    return failure();
  }
  auto invalidChip = llvm::find_if(chipIds, [&](unsigned chipId) {
    return chipId >= systemDesc.getChipDescIndices().size();
  });
  if (invalidChip != chipIds.end()) {
    failureReason = "default device selects chip " +
                    std::to_string(*invalidChip) +
                    " outside the system description";
    return failure();
  }
  ttcore::Arch arch =
      systemDesc.getChipDesc(chipIds.front()).getArch().getValue();
  if (llvm::any_of(llvm::drop_begin(chipIds), [&](unsigned chipId) {
        return systemDesc.getChipDesc(chipId).getArch().getValue() != arch;
      })) {
    failureReason = "default device selects chips with different architectures";
    return failure();
  }
  return std::optional<ttcore::Arch>(arch);
}

} // namespace target_info_detail

/// Resolve the optional architecture selected by the module attribute or
/// default device. Both sources must agree when present.
inline FailureOr<std::optional<ttcore::Arch>>
resolveTargetArch(Operation *operation, std::string &failureReason) {
  failureReason.clear();
  ModuleOp module = dyn_cast<ModuleOp>(operation);
  if (!module) {
    module = operation->getParentOfType<ModuleOp>();
  }
  if (!module) {
    failureReason = "operation is not nested in a module";
    return failure();
  }

  std::optional<ttcore::Arch> attributeArch;
  Attribute rawTargetArch = module->getAttr(kTargetArchAttrName);
  auto targetArch = dyn_cast_or_null<ttcore::ArchAttr>(rawTargetArch);
  if (rawTargetArch && !targetArch) {
    failureReason =
        (kTargetArchAttrName + " must be a #ttcore.arch attribute").str();
    return failure();
  }
  if (targetArch) {
    attributeArch = targetArch.getValue();
  }

  FailureOr<std::optional<ttcore::Arch>> deviceArch =
      target_info_detail::getDeviceArch(module, failureReason);
  if (failed(deviceArch)) {
    return failure();
  }
  if (attributeArch && *deviceArch && *attributeArch != **deviceArch) {
    failureReason =
        (kTargetArchAttrName + " does not match the selected device arch")
            .str();
    return failure();
  }
  return attributeArch ? attributeArch : *deviceArch;
}

/// Return the DFB-index capacity for a resolved target architecture.
inline TargetDFBIndexCapacity
getTargetDFBIndexCapacity(ttcore::Arch targetArch) {
  switch (targetArch) {
  case ttcore::Arch::WormholeB0:
    return {32, "Wormhole B0"};
  case ttcore::Arch::Blackhole:
    return {64, "Blackhole"};
  case ttcore::Arch::Quasar:
    return {32, "Quasar"};
  }
  llvm_unreachable("unhandled ttcore::Arch");
}

/// Return the conservative capacity used without target metadata.
inline TargetDFBIndexCapacity getConservativeDFBIndexCapacity() {
  return {32, {}};
}

/// Resolve and return the physical DFB-index capacity for an operation.
inline FailureOr<TargetDFBIndexCapacity>
resolveTargetDFBIndexCapacity(Operation *operation,
                              std::string &failureReason) {
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(operation, failureReason);
  if (failed(targetArch)) {
    return failure();
  }
  return *targetArch ? getTargetDFBIndexCapacity(**targetArch)
                     : getConservativeDFBIndexCapacity();
}

/// Return the physical DFB-index capacity after target metadata validation.
inline int32_t getTargetMaxDFBIndices(Operation *operation) {
  std::string failureReason;
  FailureOr<TargetDFBIndexCapacity> capacity =
      resolveTargetDFBIndexCapacity(operation, failureReason);
  assert(succeeded(capacity) && "target metadata must be validated");
  return capacity->indexCount;
}

/// Return target and limit text after target metadata validation.
inline std::string getTargetDFBIndexCapacityDescription(Operation *operation) {
  std::string failureReason;
  FailureOr<TargetDFBIndexCapacity> capacity =
      resolveTargetDFBIndexCapacity(operation, failureReason);
  assert(succeeded(capacity) && "target metadata must be validated");
  return capacity->getDescription();
}

/// Return the maximum one-packet NoC transfer size for the module target.
///
/// Compile-only IR may omit `ttl.target_arch`; in that case this uses the
/// minimum supported Wormhole B0/Blackhole value.
inline int64_t getTargetNocMaxBurstBytes(Operation *operation) {
  std::string failureReason;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(operation, failureReason);
  assert(succeeded(targetArch) && "target metadata must be validated");
  if (*targetArch && **targetArch == ttcore::Arch::Blackhole) {
    return kBlackholeNocMaxBurstBytes;
  }
  return kDefaultNocMaxBurstBytes;
}

} // namespace mlir::tt

#endif
