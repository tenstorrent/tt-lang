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
#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttkernel;

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.cpp.inc"

namespace {

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
