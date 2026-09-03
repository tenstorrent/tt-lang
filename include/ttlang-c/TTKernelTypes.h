// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_C_TTKERNELTYPES_H
#define TTLANG_C_TTKERNELTYPES_H

#include "ttlang-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirType ttlangTTKernelCBTypeGet(MlirContext ctx,
                                                    MlirType memrefType);

MLIR_CAPI_EXPORTED MlirType
ttlangTTKernelLocalSemaphoreTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttlangTTKernelNocAddrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTKernelThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTKernelReduceTypeAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTKernelReduceDimAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirType ttlangTTKernelL1AddrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttlangTTKernelL1AddrPtrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttlangTTKernelDataFormatTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType
ttlangTTKernelTensorAccessorArgsTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType
ttlangTTKernelTensorAccessorTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType
ttlangTTKernelLocalTensorAccessorTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType
ttlangTTKernelTensorAccessorPageMappingTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTKernelArgAttrGet(MlirContext ctx,
                                                          uint32_t argTypeValue,
                                                          size_t operandIndex,
                                                          bool isUniform);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTKernelArgSpecAttrGet(
    MlirContext ctx, MlirAttribute *rtArgs, size_t rtArgsSize,
    MlirAttribute *ctArgs, size_t ctArgsSize);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_C_TTKERNELTYPES_H
