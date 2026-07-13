// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTLSpecializeCores (Phase B of per-core specialization)
//
// Materializes per-core kernel clones from the `ttl.specialize_plan` attribute
// produced by `ttl-specialize-plan` (Phase A). This pass runs at the TTKernel
// level, right before EmitC conversion, so PipeNet lowering and its queue-depth
// validation have already completed on the single-function form.
//
// For each function carrying a plan, one clone is emitted per coordinate group.
// In each clone, every `scf.if` marked with `ttl.specialize_branch = k` has its
// condition replaced by an `arith.constant` equal to that group's `taken[k]`.
// The clone is tagged with `ttl.core_coord` (the coordinates it serves) and the
// planning markers are dropped. The following `canonicalize` / `cse` then fold
// the constant conditions and delete the dead branches.
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/SpecializeCoresAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/ADT/Twine.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSPECIALIZECORES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Emit one clone of `func` serving `coords`, forcing every marked branch to
/// the group's recorded outcome in `taken`.
static LogicalResult emitSpecializedClone(func::FuncOp func,
                                          OpBuilder &moduleBuilder,
                                          Builder &builder,
                                          DenseI64ArrayAttr coords,
                                          DenseBoolArrayAttr taken) {
  if (coords.empty() || coords.size() % 2 != 0) {
    return func.emitOpError()
           << "malformed " << kSpecializePlanAttrName << " coords entry";
  }
  int64_t repX = coords[0];
  int64_t repY = coords[1];

  func::FuncOp clone = func.clone();
  clone.setSymName(
      (func.getSymName() + "_c" + Twine(repX) + "_" + Twine(repY)).str());
  clone->removeAttr(kOperationGridAttrName);
  clone->removeAttr(kSpecializePlanAttrName);

  // Force each marked branch to this group's outcome.
  WalkResult walk = clone.walk([&](scf::IfOp ifOp) -> WalkResult {
    auto idAttr = ifOp->getAttrOfType<IntegerAttr>(kSpecializeBranchAttrName);
    if (!idAttr) {
      return WalkResult::advance();
    }
    int64_t id = idAttr.getInt();
    if (id < 0 || id >= taken.size()) {
      ifOp->emitOpError() << kSpecializeBranchAttrName << " id " << id
                          << " out of range for plan";
      return WalkResult::interrupt();
    }
    OpBuilder condBuilder(ifOp);
    Value cst = arith::ConstantOp::create(condBuilder, ifOp.getLoc(),
                                          builder.getBoolAttr(taken[id]));
    ifOp.getConditionMutable().assign(cst);
    ifOp->removeAttr(kSpecializeBranchAttrName);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    clone.erase();
    return failure();
  }

  SmallVector<Attribute> coordAttrs;
  coordAttrs.reserve(coords.size() / 2);
  for (int64_t i = 0; i < coords.size(); i += 2) {
    coordAttrs.push_back(builder.getI64ArrayAttr({coords[i], coords[i + 1]}));
  }
  clone->setAttr(kCoreCoordAttrName, builder.getArrayAttr(coordAttrs));

  moduleBuilder.insert(clone);
  return success();
}

struct TTLSpecializeCoresPass
    : impl::TTLSpecializeCoresBase<TTLSpecializeCoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> templates;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(kSpecializePlanAttrName)) {
        templates.push_back(func);
      }
    }

    Builder builder(&getContext());
    for (func::FuncOp func : templates) {
      auto plan = func->getAttrOfType<ArrayAttr>(kSpecializePlanAttrName);
      if (!plan) {
        continue;
      }

      OpBuilder moduleBuilder(func);
      for (Attribute groupAttr : plan) {
        auto group = llvm::dyn_cast<DictionaryAttr>(groupAttr);
        auto coords =
            group
                ? llvm::dyn_cast_or_null<DenseI64ArrayAttr>(group.get("coords"))
                : nullptr;
        auto taken =
            group
                ? llvm::dyn_cast_or_null<DenseBoolArrayAttr>(group.get("taken"))
                : nullptr;
        if (!coords || !taken) {
          func->emitOpError()
              << "malformed " << kSpecializePlanAttrName << " group entry";
          signalPassFailure();
          return;
        }
        if (failed(emitSpecializedClone(func, moduleBuilder, builder, coords,
                                        taken))) {
          signalPassFailure();
          return;
        }
      }

      func.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
