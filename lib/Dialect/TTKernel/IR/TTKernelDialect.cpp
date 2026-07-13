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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttkernel;

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.cpp.inc"

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
