// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_IR_TTKERNEL_H
#define TTLANG_DIALECT_TTKERNEL_IR_TTKERNEL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsDialect.h.inc"

namespace mlir::tt::ttkernel {

/// Core ranges on which an internally lowered control-flow region executes.
constexpr llvm::StringLiteral
    kExecutionCoreRangesAttrName("ttkernel.execution_core_ranges");

} // namespace mlir::tt::ttkernel

#endif
