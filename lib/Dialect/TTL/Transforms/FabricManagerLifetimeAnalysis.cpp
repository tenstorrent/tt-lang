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
  DenseI64ArrayAttr workerNodes;
  SmallVector<FabricManagerExecutionLocationAttr> executionLocations;
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
  if (claimEffects.function && claimEffects.function != function) {
    return call.emitError("fabric manager claim '")
           << claim.getValue() << "' is used by multiple logical kernels";
  }
  DenseI64ArrayAttr workerNodes = effect.getWorkerNodes();
  ArrayRef<FabricManagerExecutionLocationAttr> executionLocations =
      effect.getExecutionLocations();
  ArrayRef<int64_t> coordinates = workerNodes.asArrayRef();
  if (coordinates.size() % 2 != 0) {
    return call.emitError("fabric manager claim '")
           << claim.getValue()
           << "' worker-node domain must contain coordinate pairs";
  }
  for (std::size_t coordinateIndex = 0; coordinateIndex < coordinates.size();
       coordinateIndex += 2) {
    if (coordinates[coordinateIndex] < 0 ||
        coordinates[coordinateIndex + 1] < 0) {
      return call.emitError("fabric manager claim '")
             << claim.getValue()
             << "' worker-node coordinates must be non-negative";
    }
    for (std::size_t priorIndex = 0; priorIndex < coordinateIndex;
         priorIndex += 2) {
      if (coordinates[priorIndex] == coordinates[coordinateIndex] &&
          coordinates[priorIndex + 1] == coordinates[coordinateIndex + 1]) {
        return call.emitError("fabric manager claim '")
               << claim.getValue()
               << "' worker-node domain contains duplicate coordinates";
      }
    }
  }
  if (!coordinates.empty() && !executionLocations.empty()) {
    return call.emitError("fabric manager claim '")
           << claim.getValue()
           << "' declares both uniform and device-qualified worker domains";
  }
  for (auto indexedLocation : llvm::enumerate(executionLocations)) {
    FabricManagerExecutionLocationAttr location = indexedLocation.value();
    ArrayRef<int64_t> workerNode = location.getWorkerNode().asArrayRef();
    if (workerNode.size() != 2 || workerNode[0] < 0 || workerNode[1] < 0) {
      return call.emitError("fabric manager claim '")
             << claim.getValue()
             << "' execution locations require non-negative worker-coordinate "
                "pairs";
    }
    for (std::size_t priorIndex = 0; priorIndex < indexedLocation.index();
         ++priorIndex) {
      FabricManagerExecutionLocationAttr prior = executionLocations[priorIndex];
      if (prior.getDevice() == location.getDevice() &&
          prior.getWorkerNode() == location.getWorkerNode()) {
        return call.emitError("fabric manager claim '")
               << claim.getValue()
               << "' execution domain contains duplicate locations";
      }
    }
  }
  if (claimEffects.workerNodes && claimEffects.workerNodes != workerNodes) {
    return call.emitError("fabric manager claim '")
           << claim.getValue()
           << "' effects declare inconsistent worker-node domains";
  }
  if (!claimEffects.effects.empty() &&
      ArrayRef<FabricManagerExecutionLocationAttr>(
          claimEffects.executionLocations) != executionLocations) {
    return call.emitError("fabric manager claim '")
           << claim.getValue()
           << "' effects declare inconsistent device-qualified execution "
              "domains";
  }
  if (llvm::any_of(claimEffects.effects, [&](const auto &existing) {
        return existing.first == call;
      })) {
    return call.emitError("fabric manager claim '")
           << claim.getValue() << "' has multiple effects on one call";
  }
  claimEffects.function = function;
  claimEffects.workerNodes = workerNodes;
  claimEffects.executionLocations.assign(executionLocations.begin(),
                                         executionLocations.end());
  claimEffects.effects.emplace_back(call, effect.getKind());
  return success();
}

static bool intervalContainsBlock(const ExternalFabricManagerInterval &interval,
                                  Block *block) {
  Block *outerBlock = interval.acquire->getBlock();
  if (block == outerBlock) {
    return false;
  }

  Operation *ancestor = block->getParentOp();
  while (ancestor && ancestor->getBlock() != outerBlock) {
    ancestor = ancestor->getParentOp();
  }
  return ancestor && interval.acquire->isBeforeInBlock(ancestor) &&
         ancestor->isBeforeInBlock(interval.release);
}

static FailureOr<ExternalFabricManagerClaimLifetime>
validateClaimEffects(StringAttr claim, const ClaimEffects &claimEffects) {
  llvm::MapVector<Block *,
                  SmallVector<std::pair<OpaqueCallOp, FabricManagerEffectKind>>>
      effectsByBlock;
  for (auto effect : claimEffects.effects) {
    effectsByBlock[effect.first->getBlock()].push_back(effect);
  }

  SmallVector<ExternalFabricManagerInterval> intervals;
  for (auto &blockEntry : effectsByBlock) {
    auto &blockEffects = blockEntry.second;
    llvm::sort(blockEffects, [](const auto &lhs, const auto &rhs) {
      return lhs.first->isBeforeInBlock(rhs.first);
    });

    OpaqueCallOp acquire;
    for (auto [call, kind] : blockEffects) {
      switch (kind) {
      case FabricManagerEffectKind::Acquire:
        if (acquire) {
          return call.emitError("fabric manager claim '")
                 << claim.getValue()
                 << "' acquires ownership while it is already live";
        }
        acquire = call;
        break;
      case FabricManagerEffectKind::Use:
        if (!acquire) {
          return call.emitError("fabric manager claim '")
                 << claim.getValue()
                 << "' use is outside an acquire/release interval";
        }
        break;
      case FabricManagerEffectKind::Release:
        if (!acquire) {
          return call.emitError("fabric manager claim '")
                 << claim.getValue()
                 << "' release has no preceding acquire in its block";
        }
        intervals.push_back({acquire, call});
        acquire = {};
        break;
      case FabricManagerEffectKind::Scoped:
        if (acquire) {
          return call.emitError("fabric manager claim '")
                 << claim.getValue()
                 << "' scoped ownership overlaps a live interval";
        }
        intervals.push_back({call, call});
        break;
      }
    }
    if (acquire) {
      return acquire.emitError("fabric manager claim '")
             << claim.getValue()
             << "' acquire has no following release in its block";
    }
  }

  for (auto indexedOuter : llvm::enumerate(intervals)) {
    for (auto indexedInner : llvm::enumerate(intervals)) {
      if (indexedOuter.index() == indexedInner.index()) {
        continue;
      }
      if (intervalContainsBlock(indexedOuter.value(),
                                indexedInner.value().acquire->getBlock())) {
        return indexedInner.value().acquire.emitError("fabric manager claim '")
               << claim.getValue()
               << "' has a nested interval while ownership is already live";
      }
    }
  }

  return ExternalFabricManagerClaimLifetime{claim, claimEffects.function,
                                            claimEffects.workerNodes,
                                            claimEffects.executionLocations,
                                            std::move(intervals)};
}

} // namespace

FailureOr<SmallVector<ExternalFabricManagerClaimLifetime>>
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

  SmallVector<ExternalFabricManagerClaimLifetime> lifetimes;
  lifetimes.reserve(effectsByClaim.size());
  for (const auto &[claim, effects] : effectsByClaim) {
    FailureOr<ExternalFabricManagerClaimLifetime> lifetime =
        validateClaimEffects(claim, effects);
    if (failed(lifetime)) {
      return failure();
    }
    lifetimes.push_back(std::move(*lifetime));
  }
  return lifetimes;
}

} // namespace mlir::tt::ttl
