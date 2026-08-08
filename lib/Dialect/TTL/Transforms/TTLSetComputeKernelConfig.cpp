// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Set Compute Kernel Config Pass
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"

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
    func::FuncOp function = getOperation();
    FailureOr<std::unique_ptr<KernelTargetEnvironment>> target =
        KernelTargetEnvironment::get(function);
    FailureOr<KernelConfigPolicy> policy = KernelConfigPolicy::get(
        function, fp32DestAccEn, dstFullSyncEn, reduceFullFp32, matmulFullFp32,
        enableFPUBinaryOps);
    if (failed(target) || failed(policy)) {
      signalPassFailure();
      return;
    }

    FailureOr<KernelRequirements> requirements =
        collectKernelRequirements(function);
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
    applyKernelConfigPlan(function, *plan);
  }
};

} // namespace
} // namespace mlir::tt::ttl
