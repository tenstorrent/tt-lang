// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_PASSES_H
#define TTLANG_DIALECT_TTKERNEL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttkernel {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "ttlang/Dialect/TTKernel/Passes.h.inc"

} // namespace mlir::tt::ttkernel

#endif // TTLANG_DIALECT_TTKERNEL_PASSES_H
