// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Launch Node Domain Analysis
//===----------------------------------------------------------------------===//
//
// Utilities for computing the launch-node domain that reaches each operation.
// A domain is the set of `(core_x, core_y)` coordinates from `ttl.launch_grid`
// that can execute a program point after applying structured predicates.
// Verifiers use the domain facts from this helper while keeping their own
// policy checks and diagnostics local to each pass.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_LAUNCHNODEDOMAINANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_LAUNCHNODEDOMAINANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>
#include <set>
#include <string>

namespace mlir::tt::ttl {

/// Module attribute containing the two-dimensional launch grid extent.
inline constexpr llvm::StringLiteral kLaunchGridAttrName = "ttl.launch_grid";

/// `ttl.pipenet_scope` attribute listing the PipeNet ids selected by a scope.
inline constexpr llvm::StringLiteral kPipeNetIdsAttrName = "ttl.pipe_net_ids";

/// `ttl.pipenet_scope` attribute listing the PipeNet roles selected by a scope.
inline constexpr llvm::StringLiteral kPipeNetRolesAttrName =
    "ttl.pipe_net_roles";

/// Coordinate of one node in the module launch grid.
struct LaunchNodeCoord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator<(const LaunchNodeCoord &rhs) const;
  bool operator==(const LaunchNodeCoord &rhs) const;
};

/// Set of launch nodes that may execute a program point.
///
/// A known domain contains an exact finite set of launch-grid coordinates. An
/// unknown domain means a coordinate-dependent predicate could not be evaluated
/// statically, so clients must not treat the node set as complete. Domain
/// algebra preserves unknown inputs because verifiers cannot prove disjointness
/// from incomplete launch-node facts.
struct LaunchNodeDomain {
  bool known = true;
  std::set<LaunchNodeCoord> nodes;

  /// Return a domain whose node set is not statically known.
  static LaunchNodeDomain unknown();

  /// Return true when both domains are known and this domain is contained in
  /// `rhs`.
  bool isSubsetOf(const LaunchNodeDomain &rhs) const;

  /// Return the launch domain reached through either domain.
  LaunchNodeDomain unionWith(const LaunchNodeDomain &rhs) const;

  /// Return the launch domain common to both domains.
  LaunchNodeDomain intersectWith(const LaunchNodeDomain &rhs) const;

  /// Return the launch domain reached through this domain but not `rhs`.
  LaunchNodeDomain subtract(const LaunchNodeDomain &rhs) const;

  bool operator==(const LaunchNodeDomain &rhs) const;
};

/// Return every launch node in the rectangular `[0, gridX) x [0, gridY)` grid.
LaunchNodeDomain getFullLaunchNodeDomain(int64_t gridX, int64_t gridY);

/// Return the launch node containing the source endpoint of `pipeType`.
LaunchNodeDomain getPipeSourceLaunchNodeDomain(PipeType pipeType);

/// Return the launch nodes containing the destination endpoint range of
/// `pipeType`.
LaunchNodeDomain getPipeDestinationLaunchNodeDomain(PipeType pipeType);

/// Read the PipeNet ids selected by a `ttl.pipenet_scope`.
bool readPipeNetScopeIds(PipeNetScopeOp scopeOp, SmallVectorImpl<int64_t> &ids);

/// Domain and role metadata represented by a `ttl.pipenet_scope`.
struct PipeNetScopeLaunchNodeDomains {
  LaunchNodeDomain domain;
  SmallVector<std::pair<int64_t, PipeRole>> roles;
};

/// Module-wide launch-grid and PipeNet role domains used by the analysis.
///
/// `initialize` records the PipeNet source and destination domains from
/// `ttl.create_pipe` ops. It also parses `ttl.launch_grid`; malformed or
/// missing launch-grid attributes leave `hasLaunchGrid` false so each verifier
/// can emit diagnostics with pass-specific context.
struct LaunchNodeDomainState {
  LaunchNodeDomain baseDomain;
  llvm::DenseMap<int64_t, LaunchNodeDomain> netSourceDomains;
  llvm::DenseMap<int64_t, LaunchNodeDomain> netDestinationDomains;
  llvm::DenseMap<int64_t, SmallVector<Location>> pipeNetLocs;
  llvm::DenseMap<int64_t, std::string> pipeNetNames;
  bool sawError = false;
  bool hasLaunchGrid = false;

  /// Return true if the module contains at least one declared PipeNet.
  bool hasPipes() const;

  /// Return the recorded PipeNet name, or a deterministic id-based fallback.
  std::string netName(int64_t netId) const;

  /// Return the launch nodes that have `role` for the requested PipeNet.
  LaunchNodeDomain getRoleDomain(int64_t netId, PipeRole role) const;

  /// Populate launch-grid and PipeNet role domains from the module.
  void initialize(ModuleOp module);
};

/// Return the operation with the earlier source location.
///
/// Diagnostics use this to attach a single note for combined unknown domains
/// in a deterministic location.
Operation *pickEarlierBySourceLoc(Operation *lhs, Operation *rhs);

/// Dataflow lattice value storing the active launch-node domain before or
/// after an operation.
///
/// The optional operation records the first coordinate-dependent predicate
/// that made the domain unknown, which lets verifiers point at the predicate
/// that prevented a static proof.
class LaunchNodeDomainLattice : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LaunchNodeDomainLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  /// Join control-flow predecessors by unioning their launch-node domains.
  ChangeResult join(const AbstractDenseLattice &rhs) override;

  /// Replace the lattice value with `domain` and the optional predicate that
  /// made the domain unknown.
  ChangeResult setDomain(LaunchNodeDomain domain,
                         Operation *unanalyzableOp = nullptr);

  /// Print the domain for MLIR dataflow debugging.
  void print(raw_ostream &os) const override;

  /// Return the active launch-node domain.
  const LaunchNodeDomain &getDomain() const;

  /// Return the predicate operation that made the active domain unknown.
  Operation *getUnanalyzableOp() const;

private:
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

/// Optional callbacks and analysis behavior requested by a verifier.
struct LaunchNodeDomainAnalysisOptions {
  /// Intersect the active domain with each `ttl.pipenet_scope` role domain when
  /// entering the scope body.
  bool narrowPipeNetScopes = false;

  /// Called with the domain that reaches each operation.
  std::function<void(Operation *, const LaunchNodeDomain &, Operation *)>
      operationCallback;

  /// Called when the analysis resolves the role domain for a
  /// `ttl.pipenet_scope`.
  std::function<void(PipeNetScopeOp, const LaunchNodeDomain &, Operation *,
                     const PipeNetScopeLaunchNodeDomains &)>
      pipeNetScopeCallback;
};

/// Forward dataflow analysis that propagates launch-node domains through
/// structured control flow.
///
/// Region branch transfers narrow domains for `scf.if`, `affine.if`,
/// `ttl.if_src`, `ttl.if_dst`, and `ttl.pipenet_scope`. Other operations carry
/// the incoming domain through unchanged.
class LaunchNodeDomainAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<LaunchNodeDomainLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LaunchNodeDomainAnalysis)

  LaunchNodeDomainAnalysis(DataFlowSolver &solver, LaunchNodeDomainState &state,
                           LaunchNodeDomainAnalysisOptions options = {});

  /// Initialize entry blocks to the full launch-grid domain.
  void setToEntryState(LaunchNodeDomainLattice *lattice) override;

  /// Propagate the incoming domain through a non-branching operation.
  LogicalResult visitOperation(Operation *op,
                               const LaunchNodeDomainLattice &before,
                               LaunchNodeDomainLattice *after) override;

  /// Narrow domains when entering predicate-bearing region ops.
  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const LaunchNodeDomainLattice &before,
      LaunchNodeDomainLattice *after) override;

private:
  LaunchNodeDomainState &state;
  LaunchNodeDomainAnalysisOptions options;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_LAUNCHNODEDOMAINANALYSIS_H
