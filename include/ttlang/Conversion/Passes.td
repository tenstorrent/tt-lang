// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_CONVERSION_PASSES
#define TTLANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTTKernelToEmitC : Pass<"convert-ttkernel-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTKernel dialect to EmitC dialect.";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::func::FuncDialect",
                           "mlir::tt::ttkernel::TTKernelDialect"];
}

def RemoveDeadEmitCExpressions : Pass<"remove-dead-emitc-expressions", "::mlir::func::FuncOp"> {
  let summary = "Remove unused side-effect-free EmitC operations.";
  let constructor = "createRemoveDeadEmitCExpressionsPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
}

#endif // TTLANG_CONVERSION_PASSES
