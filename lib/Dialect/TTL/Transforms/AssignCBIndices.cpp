// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include <optional>

using namespace mlir;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLASSIGNCBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Assign CB indices to ttl.bind_cb ops when not provided. Ensures uniqueness
/// per function and preserves user-specified indices. Limits indices to the
/// hardware slot budget (0..31).
class AssignCBIndicesPass
    : public impl::TTLAssignCBIndicesBase<AssignCBIndicesPass> {
public:
  using impl::TTLAssignCBIndicesBase<
      AssignCBIndicesPass>::TTLAssignCBIndicesBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    constexpr int64_t maxCbs = mlir::tt::ttl::kMaxCircularBuffers;
    SmallVector<bool, 32> used(maxCbs, false);
    SmallVector<Type, 32> boundTypes(maxCbs, Type());
    SmallVector<BindCBOp> unassigned;

    auto diagnose = [&](BindCBOp op) -> InFlightDiagnostic {
      return op.emitOpError();
    };

    for (BindCBOp op : func.getOps<BindCBOp>()) {
      if (IntegerAttr idxAttr = op.getBufferIndexAttr()) {
        int64_t idx = idxAttr.getInt();
        if (idx < 0 || idx >= maxCbs) {
          diagnose(op) << "buffer_index must be in [0,31]";
          return;
        }
        if (used[idx]) {
          // Require identical type when sharing an index.
          if (boundTypes[idx] != op.getResult().getType()) {
            diagnose(op) << "buffer_index " << idx
                         << " already bound with type " << boundTypes[idx];
            return;
          }
        } else {
          used[idx] = true;
          boundTypes[idx] = op.getResult().getType();
        }
      } else {
        unassigned.push_back(op);
      }
    }

    int64_t nextIdx = 0;
    auto nextFreeIndex = [&]() -> std::optional<int64_t> {
      while (nextIdx < maxCbs && used[nextIdx]) {
        ++nextIdx;
      }
      if (nextIdx >= maxCbs) {
        return std::nullopt;
      }
      return nextIdx++;
    };

    Builder b(func.getContext());
    for (BindCBOp op : unassigned) {
      auto maybeIdx = nextFreeIndex();
      if (!maybeIdx) {
        diagnose(op) << "exceeded circular buffer slot budget (32)";
        return;
      }
      used[*maybeIdx] = true;
      boundTypes[*maybeIdx] = op.getResult().getType();
      op.setBufferIndexAttr(b.getI32IntegerAttr(*maybeIdx));
    }
  }
};

} // namespace

namespace mlir::tt::ttl::impl {
std::unique_ptr<Pass> createTTLAssignCBIndices() {
  return std::make_unique<AssignCBIndicesPass>();
}
} // namespace mlir::tt::ttl::impl

} // namespace mlir::tt::ttl
