// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELANNOTATEDFBUSE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using DFBSet = llvm::SmallDenseSet<int32_t, 8>;

static void collectPrintUsers(Value value, SmallVectorImpl<Operation *> &users,
                              SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second) {
      continue;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
      for (Value result : cast.getResults()) {
        collectPrintUsers(result, users, visited);
      }
      continue;
    }
    users.push_back(user);
  }
}

// True when every use (through unrealized casts) is an emitc.verbatim dprint.
// An i32 get used as a DFB id for wait/reserve/extern is not print-only.
static bool isPrintOnly(ttk::GetCompileArgValOp op) {
  SmallVector<Operation *> users;
  SmallPtrSet<Operation *, 8> visited;
  collectPrintUsers(op.getResult(), users, visited);
  return !users.empty() && llvm::all_of(users, [](Operation *user) {
    return isa<emitc::VerbatimOp>(user);
  });
}

static void warnDroppedPrint(func::FuncOp func, int32_t dfbIndex) {
  InFlightDiagnostic diag = func.emitWarning()
                            << "eliminating debug print of unused DFB "
                            << dfbIndex;
  if (auto coord = func->getAttr("ttl.core_coord")) {
    diag << " on specialized core " << coord;
  }
}

static int64_t getFuncDFBCount(func::FuncOp func, int64_t maxDFBCount) {
  if (auto attr = func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
    return attr.getInt();
  }
  return maxDFBCount;
}

// Erase dprint-only gets whose DFB index is absent from the function's
// recorded uses. Prints of a DFB remain on functions that still have a
// non-print use of that index (including uses inherited from callees).
static void
dropUnusedPrintOnlyDFBGets(ModuleOp module,
                           const llvm::DenseMap<Operation *, DFBSet> &usedDFBs,
                           int64_t maxDFBCount) {
  SmallVector<ttk::GetCompileArgValOp> gets;
  module.walk([&](ttk::GetCompileArgValOp op) { gets.push_back(op); });

  for (ttk::GetCompileArgValOp op : gets) {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func || !isPrintOnly(op)) {
      continue;
    }
    int64_t index = static_cast<int64_t>(op.getArgIndex());
    int64_t dfbCount = getFuncDFBCount(func, maxDFBCount);
    if (index < 0 || index >= dfbCount) {
      continue;
    }
    auto usedIt = usedDFBs.find(func.getOperation());
    if (usedIt != usedDFBs.end() &&
        usedIt->second.contains(static_cast<int32_t>(index))) {
      continue;
    }

    warnDroppedPrint(func, static_cast<int32_t>(index));
    SmallVector<Operation *> users;
    SmallPtrSet<Operation *, 8> visited;
    collectPrintUsers(op.getResult(), users, visited);
    for (Operation *user : users) {
      user->erase();
    }
    SmallVector<Operation *> casts;
    for (Operation *user : op->getUsers()) {
      if (isa<UnrealizedConversionCastOp>(user) && user->use_empty()) {
        casts.push_back(user);
      }
    }
    for (Operation *cast : casts) {
      cast->erase();
    }
    if (op->use_empty()) {
      op->erase();
    }
  }
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
    if (isPrintOnly(op)) {
      return;
    }
    int64_t index = static_cast<int64_t>(op.getArgIndex());
    if (index >= 0 && index < dfbCount) {
      used.insert(static_cast<int32_t>(index));
    }
  });
}

static void recordAllDFBs(func::FuncOp func, int64_t maxDFBCount,
                          DFBSet &used) {
  int64_t dfbCount = getFuncDFBCount(func, maxDFBCount);
  for (int64_t index = 0; index < dfbCount; ++index) {
    used.insert(static_cast<int32_t>(index));
  }
}

// Callees are visited before callers. Nodes in the same SCC, including
// recursive and mutually recursive functions, share one unioned use set.
static void propagateSCC(ArrayRef<CallGraphNode *> scc,
                         llvm::DenseMap<Operation *, DFBSet> &usedDFBs,
                         llvm::SmallDenseSet<Operation *> &conservative,
                         int64_t maxDFBCount) {
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
      recordAllDFBs(func, maxDFBCount, usedDFBs[key]);
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
    // Maximum DFB index in the module.
    int64_t maxDFBCount = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (auto attr = func->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName)) {
        maxDFBCount = std::max(maxDFBCount, attr.getInt());
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      int64_t dfbCount = getFuncDFBCount(func, maxDFBCount);
      collectDirectDFBUses(func, dfbCount, usedDFBs[func.getOperation()]);
    }

    CallGraph callgraph(module);
    const CallGraph *graph = &callgraph;
    for (auto sccIt = llvm::scc_begin(graph); !sccIt.isAtEnd(); ++sccIt) {
      propagateSCC(*sccIt, usedDFBs, conservative, maxDFBCount);
    }

    dropUnusedPrintOnlyDFBGets(module, usedDFBs, maxDFBCount);

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
