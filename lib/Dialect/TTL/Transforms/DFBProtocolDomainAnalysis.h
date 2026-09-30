// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBPROTOCOLDOMAINANALYSIS_H
#define TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBPROTOCOLDOMAINANALYSIS_H

//===----------------------------------------------------------------------===//
// DFB Protocol Domain Analysis
//===----------------------------------------------------------------------===//
//
// Shared inputs of the finalized-DFB verifiers (`ttl-verify-dfb-spsc`,
// `ttl-verify-dfb-lifecycle`): the launch-node domain of every operation with
// DFB protocol effects, the domain of every opaque call, the declaration of
// every logical DFB, and the opaque calls whose protocol actions on a DFB the
// IR does not represent.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::tt::ttl {

/// Launch-node domain of one operation that performs DFB protocol effects.
struct DFBProtocolActionDomain {
  LaunchNodeDomain domain = LaunchNodeDomain::unknown();
  /// Operation that prevented a precise domain, or null when precise.
  Operation *unanalyzableOp = nullptr;
};

/// Launch-node domains recorded while `LaunchNodeDomainAnalysis` runs.
struct DFBProtocolDomainState : LaunchNodeDomainState {
  llvm::DenseMap<Operation *, DFBProtocolActionDomain> protocolActionDomains;
  llvm::DenseMap<Operation *, LaunchNodeDomain> externalCallDomains;

  /// Domain of an operation with protocol effects; unknown when unrecorded,
  /// with `op` as the unanalyzable evidence.
  DFBProtocolActionDomain getProtocolActionDomain(Operation *op) const;

  /// Domain of an opaque call; unknown when unrecorded.
  LaunchNodeDomain getExternalCallDomain(Operation *op) const;
};

/// Reads launch-grid and PipeNet role domains from `module`. Emits a
/// diagnostic naming `passArgument` when the module lacks a valid
/// `ttl.launch_grid`.
LogicalResult initializeDFBProtocolDomainState(ModuleOp module,
                                               StringRef passArgument,
                                               DFBProtocolDomainState &state);

/// Runs `LaunchNodeDomainAnalysis` over `module` with PipeNet scope narrowing
/// and records protocol-action and opaque-call domains into an initialized
/// `state`. Analysis errors are forwarded.
LogicalResult analyzeDFBProtocolDomains(ModuleOp module,
                                        DFBProtocolDomainState &state);

/// Returns whether a kernel thread performs a DFB protocol effect, or with
/// `acquisitionsOnly` a reserve or wait effect.
bool hasDFBProtocolEffect(ModuleOp module, bool acquisitionsOnly = false);

/// Returns the declaration of every logical DFB. Fails with a diagnostic when
/// one logical DFB carries different finalized `cb_index` values.
FailureOr<llvm::DenseMap<int64_t, BindCBOp>>
collectFinalizedDFBBindSites(ModuleOp module);

/// Opaque calls whose protocol actions on a logical DFB the IR does not
/// represent: a dependency occurrence without an effect or access contract, and
/// every user-managed DFB for a call with `unknown_dfb_access`.
llvm::DenseMap<int64_t, SmallVector<OpaqueCallOp>>
collectDFBsWithOpaqueProtocolActions(
    ModuleOp module, const llvm::DenseMap<int64_t, BindCBOp> &bindSites);

} // namespace mlir::tt::ttl

#endif // TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBPROTOCOLDOMAINANALYSIS_H
