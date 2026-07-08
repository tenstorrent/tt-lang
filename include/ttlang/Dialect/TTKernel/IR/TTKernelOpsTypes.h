// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_IR_TTKERNELOPSTYPES_H
#define TTLANG_DIALECT_TTKERNEL_IR_TTKERNELOPSTYPES_H

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsAttrDefs.h.inc"

#endif
