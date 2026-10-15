// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBNETWORKPLANNING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBNETWORKPLANNING_H

//===----------------------------------------------------------------------===//
// DFB Network Planning
//===----------------------------------------------------------------------===//
//
// DFB network expansion rewrites network records and the kernels that use
// them into ordinary DFBs. Every route, count and binding it needs must be
// decided from the unchanged module, so a network that cannot be expanded is
// reported before any IR changes. This file provides the read-only planner and
// the immutable plan it produces.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "ttlang/Analysis/PlanningResult.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tt::ttl {

/// A valid network that this compiler cannot expand yet.
struct DFBNetworkRejection {
  /// Operation to which the pass driver attaches the diagnostic.
  Operation *operation = nullptr;

  /// Diagnostic text without an operation-name prefix.
  std::string reason;
};

/// Immutable expansion decisions for every network in one module.
///
/// Operation and value handles remain valid only while the analyzed module is
/// unchanged. The expansion pass applies the complete plan after planning
/// succeeds for every network.
class DFBNetworkPlan {
private:
  friend class DFBNetworkPlanner;
  DFBNetworkPlan() = default;
};

using DFBNetworkPlanningResult =
    PlanningResult<DFBNetworkPlan, SmallVector<DFBNetworkRejection>>;

/// Computes DFB network expansion decisions without modifying IR.
///
/// A rejected result lists every network that cannot be expanded yet, so one
/// compilation reports all of them.
class DFBNetworkPlanner {
public:
  explicit DFBNetworkPlanner(ModuleOp module) : module(module) {}

  DFBNetworkPlanningResult build() const;

private:
  ModuleOp module;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBNETWORKPLANNING_H
