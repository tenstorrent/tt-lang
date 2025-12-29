// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_C_TTLATTRS_H
#define TTLANG_C_TTLATTRS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// TTL SliceAttr
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a TTL SliceAttr.
MLIR_CAPI_EXPORTED bool ttlangMlirAttributeIsATTLSliceAttr(MlirAttribute attr);

/// Creates a TTL SliceAttr with the given start, stop, and step values.
MLIR_CAPI_EXPORTED MlirAttribute ttlangTTLSliceAttrGet(MlirContext ctx,
                                                       int64_t start,
                                                       int64_t stop,
                                                       int64_t step);

/// Gets the start value from a TTL SliceAttr.
MLIR_CAPI_EXPORTED int64_t ttlangTTLSliceAttrGetStart(MlirAttribute attr);

/// Gets the stop value from a TTL SliceAttr.
MLIR_CAPI_EXPORTED int64_t ttlangTTLSliceAttrGetStop(MlirAttribute attr);

/// Gets the step value from a TTL SliceAttr.
MLIR_CAPI_EXPORTED int64_t ttlangTTLSliceAttrGetStep(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_C_TTLATTRS_H
