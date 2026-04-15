// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that runs after all DFB-creating passes. Computes the
// true DFB count, updates ttl.base_cta_index on every function, and
// collects compiler-allocated DFBs into the ttl.compiler_allocated_dfbs
// module attribute for the Python runtime.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    // Reuse the shared utility to find the DFB count.
    int32_t numDFBs = getNextAvailableDFBIndex(moduleOp);
    if (numDFBs <= 0) {
      return;
    }

    // Collect compiler-allocated BindCBOps.
    SmallVector<BindCBOp> compilerAllocatedOps;
    moduleOp->walk([&](BindCBOp bindOp) {
      if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
        compilerAllocatedOps.push_back(bindOp);
      }
    });

    // Update ttl.base_cta_index on every function that has it.
    moduleOp->walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr(kBaseCTAIndexAttrName)) {
        funcOp->setAttr(kBaseCTAIndexAttrName,
                        builder.getI32IntegerAttr(numDFBs));
      }
    });

    if (compilerAllocatedOps.empty()) {
      return;
    }

    MLIRContext *ctx = moduleOp.getContext();
    SmallVector<Attribute> entries;

    for (BindCBOp bindOp : compilerAllocatedOps) {
      auto cbType =
          mlir::cast<CircularBufferType>(bindOp.getResult().getType());

      int32_t dfbIndex =
          static_cast<int32_t>(bindOp.getCbIndex().getSExtValue());
      int32_t numTiles = static_cast<int32_t>(cbType.getElementsPerBlock());
      int32_t blockCount = static_cast<int32_t>(cbType.getBlockCount());
      Type elementType = cbType.getElementType();

      SmallVector<NamedAttribute> entryAttrs;
      entryAttrs.push_back(builder.getNamedAttr(
          "dfb_index", builder.getI32IntegerAttr(dfbIndex)));
      entryAttrs.push_back(builder.getNamedAttr(
          "num_tiles", builder.getI32IntegerAttr(numTiles)));
      entryAttrs.push_back(
          builder.getNamedAttr("element_type", TypeAttr::get(elementType)));
      entryAttrs.push_back(builder.getNamedAttr(
          "block_count", builder.getI32IntegerAttr(blockCount)));
      entries.push_back(DictionaryAttr::get(ctx, entryAttrs));
    }

    moduleOp->setAttr(kCompilerAllocatedDFBsAttrName,
                      ArrayAttr::get(ctx, entries));
  }
};

} // namespace

} // namespace mlir::tt::ttl
