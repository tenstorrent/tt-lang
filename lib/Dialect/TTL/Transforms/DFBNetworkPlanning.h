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
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

/// A valid network that this compiler cannot expand yet.
struct DFBNetworkRejection {
  /// Operation to which the pass driver attaches the diagnostic.
  Operation *operation = nullptr;

  /// Diagnostic text without an operation-name prefix.
  std::string reason;
};

/// The end of a DFB that a network record takes over from kernels.
enum class DFBNetworkRole {
  /// Fork or split source: the record consumes it.
  Source,
  /// Merge input: the record consumes it.
  Input,
  /// Fork or split output: the record produces it.
  Output,
};

/// Returns the name of `role` used in debug output, e.g. "source".
StringRef stringifyDFBNetworkRole(DFBNetworkRole role);

/// One DFB that a record names, with the kernel side of it.
struct DFBNetworkEndpoint {
  int64_t dfbId = 0;

  /// End of the DFB that the record holds.
  DFBNetworkRole role = DFBNetworkRole::Source;

  /// `ttl.bind_cb` declarations of the DFB, in module order.
  SmallVector<BindCBOp> bindings;

  /// Kernel ops on the end the kernels keep, in module order: the reserves
  /// and pushes of a source or input, or the waits and pops of an output.
  SmallVector<Operation *> kernelOps;
};

/// Resolved endpoints of one record.
struct DFBNetworkRecordPlan {
  Operation *record = nullptr;

  /// DFBs the record names, in operand order.
  SmallVector<DFBNetworkEndpoint> endpoints;
};

/// Resolved endpoints of one network that planning admitted.
struct DFBNetworkExpansion {
  DFBNetworkOp network;

  /// Records in network body order.
  SmallVector<DFBNetworkRecordPlan> records;
};

/// Immutable expansion decisions for every network in one module.
///
/// Operation and value handles remain valid only while the analyzed module is
/// unchanged. The expansion pass applies the complete plan after planning
/// succeeds for every network.
class DFBNetworkPlan {
public:
  /// Admitted networks in module order.
  ArrayRef<DFBNetworkExpansion> getNetworks() const { return networks; }

private:
  friend class DFBNetworkPlanner;
  DFBNetworkPlan() = default;

  SmallVector<DFBNetworkExpansion> networks;
};

using DFBNetworkPlanningResult =
    PlanningResult<DFBNetworkPlan, SmallVector<DFBNetworkRejection>>;

/// Computes DFB network expansion decisions without modifying IR.
///
/// Planning resolves the bindings, producers, and consumers of every DFB that
/// a record names. A network that breaks a module-level rule is invalid IR. A
/// rejected result lists every feature of a valid network that cannot be
/// expanded yet, so one compilation reports all of them.
class DFBNetworkPlanner {
public:
  explicit DFBNetworkPlanner(ModuleOp module) : module(module) {}

  DFBNetworkPlanningResult build() const;

private:
  ModuleOp module;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBNETWORKPLANNING_H
