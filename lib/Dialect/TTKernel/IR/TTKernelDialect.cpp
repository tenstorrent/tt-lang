// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttkernel;

NocCommandEffects NocCommandEffectsAnalysis::getEffects(Operation *operation) {
  NocCommandEffects effects{mayReprogramNocCommand(operation, commandClass),
                            usesNocCommandState(operation, commandClass)};
  auto call = dyn_cast<CallOpInterface>(operation);
  if (!call) {
    return effects;
  }
  Operation *callee = call.resolveCallable();
  auto callable = dyn_cast_or_null<CallableOpInterface>(callee);
  Region *body = callable ? callable.getCallableRegion() : nullptr;
  if (!body) {
    return {true, true};
  }
  auto cached = callableEffects.find(callee);
  if (cached != callableEffects.end()) {
    effects.mayReprogram |= cached->second.mayReprogram;
    effects.mayUseState |= cached->second.mayUseState;
    return effects;
  }

  // A recursive call encounters this conservative entry before all operations
  // in its strongly connected component have been inspected.
  callableEffects[callee] = {true, true};
  body->walk([&](Operation *nested) {
    NocCommandEffects nestedEffects = getEffects(nested);
    effects.mayReprogram |= nestedEffects.mayReprogram;
    effects.mayUseState |= nestedEffects.mayUseState;
  });
  callableEffects[callee] = effects;
  return effects;
}

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.cpp.inc"

namespace {

// Collect core-range restrictions on ancestors of `op`, stopping before
// `limit`. The operation executes only on cores allowed by every collected
// restriction.
SmallVector<ArrayAttr> getEnclosingExecutionCoreRanges(Operation *op,
                                                       Operation *limit) {
  SmallVector<ArrayAttr> domains;
  for (Operation *ancestor = op->getParentOp(); ancestor && ancestor != limit;
       ancestor = ancestor->getParentOp()) {
    if (auto ranges =
            ancestor->getAttrOfType<ArrayAttr>(kExecutionCoreRangesAttrName)) {
      domains.push_back(ranges);
    }
  }
  return domains;
}

// Prove that no core belongs to both range lists. Empty or malformed lists
// provide insufficient information and return false.
bool haveDisjointCoreRanges(ArrayAttr lhs, ArrayAttr rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  for (Attribute lhsAttr : lhs) {
    auto lhsRange = dyn_cast<tt::ttcore::CoreRangeAttr>(lhsAttr);
    if (!lhsRange) {
      return false;
    }
    for (Attribute rhsAttr : rhs) {
      auto rhsRange = dyn_cast<tt::ttcore::CoreRangeAttr>(rhsAttr);
      if (!rhsRange || lhsRange.intersects(rhsRange)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

bool mlir::tt::ttkernel::haveDisjointExecutionCoreRanges(Operation *lhs,
                                                         Operation *rhs,
                                                         Operation *limit) {
  SmallVector<ArrayAttr> lhsDomains =
      getEnclosingExecutionCoreRanges(lhs, limit);
  SmallVector<ArrayAttr> rhsDomains =
      getEnclosingExecutionCoreRanges(rhs, limit);
  return llvm::any_of(lhsDomains, [&](ArrayAttr lhsDomain) {
    return llvm::any_of(rhsDomains, [&](ArrayAttr rhsDomain) {
      return haveDisjointCoreRanges(lhsDomain, rhsDomain);
    });
  });
}

//===----------------------------------------------------------------------===//
// TTKernel dialect.
//===----------------------------------------------------------------------===//

void TTKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
}
