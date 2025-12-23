// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_C_DIALECTS_H
#define TTLANG_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// TTL Dialect Registration
//===----------------------------------------------------------------------===//

/// Register all tt-lang dialects with the given MlirContext.
MLIR_CAPI_EXPORTED void ttlangRegisterAllDialects(MlirContext context);

/// Register the TTL dialect with the given MlirDialectRegistry.
MLIR_CAPI_EXPORTED void ttlangRegisterTTLDialect(MlirDialectRegistry registry);

/// Load the TTL dialect into the given MlirContext.
MLIR_CAPI_EXPORTED MlirDialectHandle ttlangGetTTLDialectHandle(void);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_C_DIALECTS_H
