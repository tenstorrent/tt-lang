// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H
#define TTLANG_CONVERSION_TTKERNELTOEMITC_TTKERNELTOEMITC_H

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {
#define GEN_PASS_DECL_CONVERTTTKERNELTOEMITC
#include "ttlang/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveDeadEmitCExpressionsPass();
} // namespace mlir::tt

#endif
