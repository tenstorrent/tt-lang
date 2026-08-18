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

#define GEN_PASS_DEF_TTKERNELANNOTATEDFBUSE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using DFBIndexSet = llvm::SmallDenseSet<int32_t, 8>;

static int64_t getDFBCount(func::FuncOp function, int64_t moduleDFBCount) {
  if (auto baseCTAIndex =
          function->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
    return baseCTAIndex.getInt();
  }
  return moduleDFBCount;
}

struct TTKernelAnnotateDFBUsePass
    : impl::TTKernelAnnotateDFBUseBase<TTKernelAnnotateDFBUsePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<Operation *> symbols;
    llvm::DenseMap<Operation *, DFBIndexSet> usedDFBIndices;
    llvm::DenseMap<Operation *, SmallVector<StringRef>> directCallees;
    llvm::SmallDenseSet<Operation *> conservativelyAnnotated;
    int64_t moduleDFBCount = 0;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      symbols[function.getSymName()] = function.getOperation();
      if (auto baseCTAIndex =
              function->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
        moduleDFBCount = std::max(moduleDFBCount, baseCTAIndex.getInt());
      }
    }

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      Operation *functionOperation = function.getOperation();
      int64_t dfbCount = getDFBCount(function, moduleDFBCount);
      function.walk([&](ttk::GetCompileArgValOp compileArgument) {
        int64_t argumentIndex =
            static_cast<int64_t>(compileArgument.getArgIndex());
        if (argumentIndex >= 0 && argumentIndex < dfbCount) {
          usedDFBIndices[functionOperation].insert(
              static_cast<int32_t>(argumentIndex));
        }
      });
      function.walk([&](func::CallOp call) {
        directCallees[functionOperation].push_back(call.getCallee());
      });
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        Operation *functionOperation = function.getOperation();
        for (StringRef calleeName : directCallees[functionOperation]) {
          auto calleeIterator = symbols.find(calleeName);
          if (calleeIterator == symbols.end()) {
            changed |= conservativelyAnnotated.insert(functionOperation).second;
            continue;
          }

          Operation *calleeOperation = calleeIterator->second;
          auto callee = cast<func::FuncOp>(calleeOperation);
          if (callee.isDeclaration() ||
              conservativelyAnnotated.contains(calleeOperation)) {
            changed |= conservativelyAnnotated.insert(functionOperation).second;
          }
          for (int32_t dfbIndex : usedDFBIndices[calleeOperation]) {
            changed |=
                usedDFBIndices[functionOperation].insert(dfbIndex).second;
          }
        }
      }
    }

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      Operation *functionOperation = function.getOperation();
      if (conservativelyAnnotated.contains(functionOperation)) {
        int64_t dfbCount = getDFBCount(function, moduleDFBCount);
        for (int64_t dfbIndex = 0; dfbIndex < dfbCount; ++dfbIndex) {
          usedDFBIndices[functionOperation].insert(
              static_cast<int32_t>(dfbIndex));
        }
      }

      SmallVector<int32_t> sortedIndices(
          usedDFBIndices[functionOperation].begin(),
          usedDFBIndices[functionOperation].end());
      llvm::sort(sortedIndices);
      function->setAttr(
          kUsedDFBIndicesAttrName,
          DenseI32ArrayAttr::get(module.getContext(), sortedIndices));
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
