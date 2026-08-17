// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "FabricManagerLifetimeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::tt::ttl {
namespace {

struct ClaimEffects {
  func::FuncOp function;
  SmallVector<std::pair<OpaqueCallOp, FabricManagerEffectKind>> effects;
};

static LogicalResult
appendClaimEffect(OpaqueCallOp call, FabricManagerEffectAttr effect,
                  llvm::MapVector<StringAttr, ClaimEffects> &effectsByClaim) {
  StringAttr claim = effect.getClaim();
  ClaimEffects &claimEffects = effectsByClaim[claim];
  func::FuncOp function = call->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr(kLogicalKernelAttrName)) {
    return call.emitError("fabric manager effects require an enclosing logical "
                          "kernel function");
  }
  if (call->getBlock() != &function.getBody().front() &&
      effect.getKind() != FabricManagerEffectKind::Scoped) {
    return call.emitError("fabric manager effects must be in the logical "
                          "kernel's straight-line entry block unless the "
                          "effect is scoped");
  }
  if (claimEffects.function && claimEffects.function != function) {
    return call.emitError("fabric manager claim '")
           << claim.getValue() << "' is used by multiple logical kernels";
  }
  if (llvm::any_of(claimEffects.effects, [&](const auto &existing) {
        return existing.first == call;
      })) {
    return call.emitError("fabric manager claim '")
           << claim.getValue() << "' has multiple effects on one call";
  }
  claimEffects.function = function;
  claimEffects.effects.emplace_back(call, effect.getKind());
  return success();
}

static FailureOr<ExternalFabricManagerInterval>
validateClaimEffects(StringAttr claim, const ClaimEffects &claimEffects) {
  SmallVector<OpaqueCallOp> acquires;
  SmallVector<OpaqueCallOp> releases;
  SmallVector<OpaqueCallOp> scoped;
  SmallVector<OpaqueCallOp> uses;
  for (auto [call, kind] : claimEffects.effects) {
    switch (kind) {
    case FabricManagerEffectKind::Acquire:
      acquires.push_back(call);
      break;
    case FabricManagerEffectKind::Use:
      uses.push_back(call);
      break;
    case FabricManagerEffectKind::Release:
      releases.push_back(call);
      break;
    case FabricManagerEffectKind::Scoped:
      scoped.push_back(call);
      break;
    }
  }

  Operation *diagnosticOperation = claimEffects.effects.front().first;
  auto emitClaimError = [&](const Twine &message) -> LogicalResult {
    diagnosticOperation->emitError("fabric manager claim '")
        << claim.getValue() << "' " << message;
    return failure();
  };
  if (!scoped.empty()) {
    if (scoped.size() != 1 || claimEffects.effects.size() != 1) {
      return emitClaimError("must use one scoped effect or one acquire/release "
                            "interval, not both");
    }
    return ExternalFabricManagerInterval{claim, claimEffects.function,
                                         scoped.front(), scoped.front()};
  }
  if (acquires.size() != 1) {
    return emitClaimError(acquires.empty() ? "has no acquire effect"
                                           : "has multiple acquire effects");
  }
  if (releases.size() != 1) {
    return emitClaimError(releases.empty() ? "has no release effect"
                                           : "has multiple release effects");
  }

  OpaqueCallOp acquire = acquires.front();
  OpaqueCallOp release = releases.front();
  if (!acquire->isBeforeInBlock(release)) {
    release.emitError("fabric manager claim '")
        << claim.getValue() << "' release must follow its acquire";
    return failure();
  }
  for (OpaqueCallOp use : uses) {
    if (!acquire->isBeforeInBlock(use) || !use->isBeforeInBlock(release)) {
      use.emitError("fabric manager claim '")
          << claim.getValue()
          << "' use must be between its acquire and release";
      return failure();
    }
  }
  return ExternalFabricManagerInterval{claim, claimEffects.function, acquire,
                                       release};
}

} // namespace

FailureOr<SmallVector<ExternalFabricManagerInterval>>
analyzeExternalFabricManagerLifetimes(ModuleOp module) {
  llvm::MapVector<StringAttr, ClaimEffects> effectsByClaim;
  WalkResult walkResult = module.walk([&](OpaqueCallOp call) {
    ArrayAttr effects = call.getFabricManagerEffectsAttr();
    if (!effects) {
      return WalkResult::advance();
    }
    for (Attribute attribute : effects) {
      auto effect = cast<FabricManagerEffectAttr>(attribute);
      if (failed(appendClaimEffect(call, effect, effectsByClaim))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  SmallVector<ExternalFabricManagerInterval> intervals;
  intervals.reserve(effectsByClaim.size());
  for (const auto &[claim, effects] : effectsByClaim) {
    FailureOr<ExternalFabricManagerInterval> interval =
        validateClaimEffects(claim, effects);
    if (failed(interval)) {
      return failure();
    }
    intervals.push_back(std::move(*interval));
  }
  return intervals;
}

} // namespace mlir::tt::ttl
