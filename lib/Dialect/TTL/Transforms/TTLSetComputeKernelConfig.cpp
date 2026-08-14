// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Set Compute Kernel Config Pass
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSETCOMPUTEKERNELCONFIG
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLSetComputeKernelConfigPass
    : public impl::TTLSetComputeKernelConfigBase<
          TTLSetComputeKernelConfigPass> {
  using Base =
      impl::TTLSetComputeKernelConfigBase<TTLSetComputeKernelConfigPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LaunchNodeDomainState launchDomains;
    launchDomains.initialize(module);

    SmallVector<std::pair<func::FuncOp, KernelConfigPlan>> plans;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      FailureOr<std::unique_ptr<KernelTargetEnvironment>> target =
          KernelTargetEnvironment::get(function);
      if (failed(target)) {
        signalPassFailure();
        return;
      }
      FailureOr<KernelConfigPolicy> policy = KernelConfigPolicy::get(
          function, fp32DestAccEn, dstFullSyncEn, reduceFullFp32,
          matmulFullFp32, enableFPUBinaryOps);
      if (failed(policy)) {
        signalPassFailure();
        return;
      }
      FailureOr<KernelRequirements> requirements =
          collectKernelRequirements(function, launchDomains);
      if (failed(requirements)) {
        signalPassFailure();
        return;
      }
      FailureOr<KernelConfigPlan> plan =
          resolveKernelConfig(function, **target, *policy, *requirements);
      if (failed(plan)) {
        signalPassFailure();
        return;
      }
      plans.emplace_back(function, std::move(*plan));
    }

    for (auto &[function, plan] : plans) {
      applyKernelConfigPlan(function, plan);
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
