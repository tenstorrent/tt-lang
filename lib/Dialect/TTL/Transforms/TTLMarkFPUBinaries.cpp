// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLMARKFPUBINARIES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLMarkFPUBinariesPass
    : public impl::TTLMarkFPUBinariesBase<TTLMarkFPUBinariesPass> {
  using Base = impl::TTLMarkFPUBinariesBase<TTLMarkFPUBinariesPass>;
  using Base::Base;

  void runOnOperation() override {
    if (!enableFPUBinaryOps) {
      return;
    }
    UnitAttr marker = UnitAttr::get(&getContext());
    getOperation().walk([&](ComputeOp computeOp) {
      for (Operation &op : computeOp.getRegion().front()) {
        if (isFpuBinaryEligible(&op, computeOp, /*enableFPUBinaryOps=*/true)) {
          op.setAttr(kFPUBinaryAttrName, marker);
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttl
