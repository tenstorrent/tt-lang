// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TRANSFORMS_PASSES_H
#define TTLANG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "ttlang/Transforms/Passes.h.inc"

} // namespace mlir::tt::d2m

#endif // TTLANG_TRANSFORMS_PASSES_H
