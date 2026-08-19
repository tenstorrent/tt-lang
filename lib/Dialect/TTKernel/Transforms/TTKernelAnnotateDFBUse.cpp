// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELANNOTATEDFBUSE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using DFBSet = llvm::SmallDenseSet<int32_t, 8>;

static int64_t getDFBCount(func::FuncOp func, int64_t fallback) {
  if (auto attr = func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
    return attr.getInt();
  }
  return fallback;
}

static func::FuncOp getCallableFunc(CallGraphNode *node) {
  if (node->isExternal()) {
    return {};
  }
  return dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
}

static void collectDirectDFBUses(func::FuncOp func, int64_t dfbCount,
                                 DFBSet &used) {
  func.walk([&](ttk::GetCompileArgValOp op) {
    int64_t index = static_cast<int64_t>(op.getArgIndex());
    if (index >= 0 && index < dfbCount) {
      used.insert(static_cast<int32_t>(index));
    }
  });
}

static void recordAllDFBs(func::FuncOp func, int64_t moduleDFBCount,
                          DFBSet &used) {
  int64_t dfbCount = getDFBCount(func, moduleDFBCount);
  for (int64_t index = 0; index < dfbCount; ++index) {
    used.insert(static_cast<int32_t>(index));
  }
}

// Callees are visited before callers. Nodes in the same SCC, including
// recursive and mutually recursive functions, share one unioned use set.
static void propagateSCC(ArrayRef<CallGraphNode *> scc,
                         llvm::DenseMap<Operation *, DFBSet> &usedDFBs,
                         llvm::SmallDenseSet<Operation *> &conservative,
                         int64_t moduleDFBCount) {
  SmallVector<func::FuncOp> funcs;
  DFBSet sccUses;
  for (CallGraphNode *node : scc) {
    func::FuncOp func = getCallableFunc(node);
    if (!func) {
      continue;
    }
    funcs.push_back(func);
    for (int32_t index : usedDFBs[func.getOperation()]) {
      sccUses.insert(index);
    }
  }
  if (funcs.empty()) {
    return;
  }

  llvm::SmallDenseSet<CallGraphNode *, 4> sccNodes(scc.begin(), scc.end());
  bool sccConservative = false;
  for (CallGraphNode *node : scc) {
    if (node->isExternal()) {
      continue;
    }
    for (const CallGraphNode::Edge &edge : *node) {
      if (!edge.isCall()) {
        continue;
      }
      CallGraphNode *target = edge.getTarget();
      if (sccNodes.contains(target)) {
        continue;
      }
      func::FuncOp callee = getCallableFunc(target);
      if (!callee || callee.isDeclaration() ||
          conservative.contains(callee.getOperation())) {
        sccConservative = true;
        continue;
      }
      for (int32_t index : usedDFBs[callee.getOperation()]) {
        sccUses.insert(index);
      }
    }
  }

  for (func::FuncOp func : funcs) {
    Operation *key = func.getOperation();
    if (sccConservative) {
      conservative.insert(key);
      recordAllDFBs(func, moduleDFBCount, usedDFBs[key]);
    } else {
      usedDFBs[key] = sccUses;
    }
  }
}

struct TTKernelAnnotateDFBUsePass
    : impl::TTKernelAnnotateDFBUseBase<TTKernelAnnotateDFBUsePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::DenseMap<Operation *, DFBSet> usedDFBs;
    llvm::SmallDenseSet<Operation *> conservative;
    int64_t moduleDFBCount = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (auto attr =
              func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
        moduleDFBCount = std::max(moduleDFBCount, attr.getInt());
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      collectDirectDFBUses(func, getDFBCount(func, moduleDFBCount),
                           usedDFBs[func.getOperation()]);
    }

    CallGraph callgraph(module);
    const CallGraph *graph = &callgraph;
    for (auto sccIt = llvm::scc_begin(graph); !sccIt.isAtEnd(); ++sccIt) {
      propagateSCC(*sccIt, usedDFBs, conservative, moduleDFBCount);
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!getKernelThreadType(func)) {
        continue;
      }
      Operation *key = func.getOperation();
      SmallVector<int32_t> sorted(usedDFBs[key].begin(), usedDFBs[key].end());
      llvm::sort(sorted);
      func->setAttr(kUsedDFBIndicesAttrName,
                    DenseI32ArrayAttr::get(module.getContext(), sorted));
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
