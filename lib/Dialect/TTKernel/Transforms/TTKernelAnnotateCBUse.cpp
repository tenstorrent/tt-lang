// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELANNOTATECBUSE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using CBSet = llvm::SmallDenseSet<int32_t, 8>;

static int64_t getCBCount(func::FuncOp func, int64_t fallback) {
  if (auto attr =
          func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
    return attr.getInt();
  }
  return fallback;
}

struct TTKernelAnnotateCBUsePass
    : impl::TTKernelAnnotateCBUseBase<TTKernelAnnotateCBUsePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<Operation *> symbols;
    llvm::DenseMap<Operation *, CBSet> usedCBs;
    llvm::DenseMap<Operation *, SmallVector<StringRef>> callees;
    llvm::SmallDenseSet<Operation *> conservative;
    int64_t moduleCBCount = 0;

    // Populate symbol map and get max base cta index
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      symbols[func.getSymName()] = func.getOperation();
      if (auto attr =
              func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
        moduleCBCount = std::max(moduleCBCount, attr.getInt());
      }
    }

    // Populate usedCBs set and callees map
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      Operation *key = func.getOperation();
      int64_t cbCount = getCBCount(func, moduleCBCount);
      func.walk([&](ttk::GetCompileArgValOp op) {
        int64_t index = static_cast<int64_t>(op.getArgIndex());
        if (index >= 0 && index < cbCount) {
          usedCBs[key].insert(static_cast<int32_t>(index));
        }
      });
      func.walk([&](func::CallOp call) {
        callees[key].push_back(call.getCallee());
      });
    }

    // Iterate through all functions and update their usedCBs set.
    // Here we define a function to be "conservative" if it calls a function that is
    // "conservative" or if it calls a function that is not in the symbol map.
    // A "conservative" function is one that uses all CBs up to the max base cta index.
    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp func : module.getOps<func::FuncOp>()) {
        Operation *key = func.getOperation();
        for (StringRef calleeName : callees[key]) {
          auto calleeIt = symbols.find(calleeName);
          if (calleeIt == symbols.end()) {
            changed |= conservative.insert(key).second;
            continue;
          }
          Operation *callee = calleeIt->second;
          auto calleeFunc = cast<func::FuncOp>(callee);
          if (calleeFunc.isDeclaration() || conservative.contains(callee)) {
            changed |= conservative.insert(key).second;
          }
          for (int32_t index : usedCBs[callee]) {
            changed |= usedCBs[key].insert(index).second;
          }
        }
      }
    }

    // Annotate the functions with the usedCBs set.
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      Operation *key = func.getOperation();
      if (conservative.contains(key)) {
        int64_t cbCount = getCBCount(func, moduleCBCount);
        for (int64_t index = 0; index < cbCount; ++index) {
          usedCBs[key].insert(static_cast<int32_t>(index));
        }
      }
      SmallVector<int32_t> sorted(usedCBs[key].begin(), usedCBs[key].end());
      llvm::sort(sorted);
      func->setAttr(kUsedDFBIndicesAttrName,
                    DenseI32ArrayAttr::get(module.getContext(), sorted));
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
