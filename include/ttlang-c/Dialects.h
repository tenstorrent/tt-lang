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

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTL, ttl);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTCore, ttcore);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel);

/// Register all tt-lang dialects with the given MlirContext.
MLIR_CAPI_EXPORTED void ttlangRegisterAllDialects(MlirContext context);

/// Register the TTL dialect with the given MlirDialectRegistry.
MLIR_CAPI_EXPORTED void ttlangRegisterTTLDialect(MlirDialectRegistry registry);

/// Register the TTCore dialect with the given MlirDialectRegistry.
MLIR_CAPI_EXPORTED void
ttlangRegisterTTCoreDialect(MlirDialectRegistry registry);

/// Register the TTKernel dialect with the given MlirDialectRegistry.
MLIR_CAPI_EXPORTED void
ttlangRegisterTTKernelDialect(MlirDialectRegistry registry);

/// Register the minimal set of upstream MLIR dialects the tt-lang pipeline uses
/// with the given MlirDialectRegistry. Replaces MLIR's RegisterEverything so
/// the Python CAPI library does not statically link every upstream dialect.
MLIR_CAPI_EXPORTED void
ttlangRegisterUpstreamDialects(MlirDialectRegistry registry);

/// Register tt-lang passes.
MLIR_CAPI_EXPORTED void ttlangRegisterPasses(void);

/// Run TTKernelToEmitC conversion pass on a module.
/// Returns true on success, false on failure.
MLIR_CAPI_EXPORTED bool ttlangRunTTKernelToEmitC(MlirModule module);

/// Translate a named TTKernel function to C++.
/// Caller must free the returned string with free().
/// Returns NULL on failure.
MLIR_CAPI_EXPORTED char *ttlangTranslateKernelToCpp(MlirModule module,
                                                    const char *kernelName);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_C_DIALECTS_H
