// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// C API for the minimal set of tt-mlir dialects used by tt-lang.

#ifndef TTLANG_TTMLIR_MINIMAL_CAPI_DIALECTS_H
#define TTLANG_TTMLIR_MINIMAL_CAPI_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TT, tt);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTMetal, ttmetal);

/// Register the minimal set of tt-mlir dialects into a registry.
MLIR_CAPI_EXPORTED void
ttmlirMinimalRegisterAllDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_TTMLIR_MINIMAL_CAPI_DIALECTS_H
