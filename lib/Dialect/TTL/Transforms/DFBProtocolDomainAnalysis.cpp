// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBProtocolDomainAnalysis.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

namespace mlir::tt::ttl {

DFBProtocolActionDomain
DFBProtocolDomainState::getProtocolActionDomain(Operation *op) const {
  auto domainIt = protocolActionDomains.find(op);
  if (domainIt == protocolActionDomains.end()) {
    return DFBProtocolActionDomain{LaunchNodeDomain::unknown(), op};
  }
  return domainIt->second;
}

LaunchNodeDomain
DFBProtocolDomainState::getExternalCallDomain(Operation *op) const {
  auto domainIt = externalCallDomains.find(op);
  return domainIt == externalCallDomains.end() ? LaunchNodeDomain::unknown()
                                               : domainIt->second;
}

LogicalResult initializeDFBProtocolDomainState(ModuleOp module,
                                               StringRef passArgument,
                                               DFBProtocolDomainState &state) {
  state.initialize(module);
  if (!state.hasLaunchGrid) {
    module.emitError()
        << passArgument
        << " requires a `ttl.launch_grid` module attribute (an i64 array of "
           "length 2 with positive entries) when verifying DFB protocol "
           "actions";
    return failure();
  }
  return success();
}

LogicalResult analyzeDFBProtocolDomains(ModuleOp module,
                                        DFBProtocolDomainState &state) {
  assert(state.hasLaunchGrid && "state must be initialized");
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  LaunchNodeDomainAnalysisOptions options;
  options.narrowPipeNetScopes = true;
  options.operationCallback = [&](Operation *op, const LaunchNodeDomain &domain,
                                  Operation *unanalyzableOp) {
    if (isa<OpaqueCallOp>(op)) {
      state.externalCallDomains[op] = domain;
    }
    auto access = dyn_cast<DFBAccessOpInterface>(op);
    if (access && !access.getDFBProtocolEffects().empty()) {
      state.protocolActionDomains[op] = {domain, unanalyzableOp};
    }
  };
  solver.load<LaunchNodeDomainAnalysis>(state, options);
  if (failed(solver.initializeAndRun(module)) || state.sawError) {
    return failure();
  }
  return success();
}

bool hasDFBProtocolEffect(ModuleOp module, bool acquisitionsOnly) {
  bool hasEffect = false;
  module.walk([&](DFBAccessOpInterface access) {
    if (!getEnclosingKernelThread(access)) {
      return;
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      hasEffect |= !acquisitionsOnly ||
                   effect.kind == DFBProtocolEffectKind::Reserve ||
                   effect.kind == DFBProtocolEffectKind::Wait;
    }
  });
  return hasEffect;
}

FailureOr<llvm::DenseMap<int64_t, BindCBOp>>
collectFinalizedDFBBindSites(ModuleOp module) {
  llvm::DenseMap<int64_t, BindCBOp> bindSites;
  bool hasInconsistentIndex = false;
  module.walk([&](BindCBOp bindOp) {
    FailureOr<int64_t> dfbId = getDFBId(bindOp.getResult());
    assert(succeeded(dfbId) && "DFB identities were verified");
    auto [siteIt, inserted] = bindSites.try_emplace(*dfbId, bindOp);
    if (inserted) {
      return;
    }
    int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
    int64_t firstIndex = siteIt->second.getCbIndex().getSExtValue();
    if (firstIndex != cbIndex) {
      bindOp.emitOpError() << "logical DFB " << *dfbId
                           << " has inconsistent finalized cb_index values "
                           << firstIndex << " and " << cbIndex;
      hasInconsistentIndex = true;
    }
  });
  if (hasInconsistentIndex) {
    return failure();
  }
  return bindSites;
}

llvm::DenseMap<int64_t, SmallVector<OpaqueCallOp>>
collectDFBsWithOpaqueProtocolActions(
    ModuleOp module, const llvm::DenseMap<int64_t, BindCBOp> &bindSites) {
  llvm::DenseMap<int64_t, SmallVector<OpaqueCallOp>> callsByDFB;
  module.walk([&](OpaqueCallOp call) {
    if (!getEnclosingKernelThread(call)) {
      return;
    }
    SmallVector<Value> dependencies = call.getDFBDependencyOperands();
    for (unsigned dependencyIndex : getOpaqueDFBDependencyIndices(call)) {
      FailureOr<int64_t> dfbId = getDFBId(dependencies[dependencyIndex]);
      assert(succeeded(dfbId) && "DFB identities were verified");
      callsByDFB[*dfbId].push_back(call);
    }
    if (!call.hasUnknownDFBAccess()) {
      return;
    }
    for (auto [dfbId, bindSite] : bindSites) {
      if (isUserManagedDFB(bindSite.getResult())) {
        callsByDFB[dfbId].push_back(call);
      }
    }
  });
  return callsByDFB;
}

} // namespace mlir::tt::ttl
