// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang-c/TTKernelTypes.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir::tt::ttkernel;

MlirType ttlangTTKernelCBTypeGet(MlirContext ctx, MlirType memrefType) {
  return wrap(CBType::get(mlir::cast<mlir::MemRefType>(unwrap(memrefType))));
}

MlirType ttlangTTKernelLocalSemaphoreTypeGet(MlirContext ctx) {
  return wrap(LocalSemaphoreType::get(unwrap(ctx)));
}

MlirType ttlangTTKernelNocAddrTypeGet(MlirContext ctx) {
  return wrap(NocAddrType::get(unwrap(ctx)));
}

MlirAttribute ttlangTTKernelThreadTypeAttrGet(MlirContext ctx,
                                              uint32_t enumValue) {
  return wrap(
      ThreadTypeAttr::get(unwrap(ctx), static_cast<ThreadType>(enumValue)));
}

MlirAttribute ttlangTTKernelReduceTypeAttrGet(MlirContext ctx,
                                              uint32_t enumValue) {
  return wrap(
      ReduceTypeAttr::get(unwrap(ctx), static_cast<ReduceType>(enumValue)));
}

MlirAttribute ttlangTTKernelReduceDimAttrGet(MlirContext ctx,
                                             uint32_t enumValue) {
  return wrap(
      ReduceDimAttr::get(unwrap(ctx), static_cast<ReduceDim>(enumValue)));
}

MlirType ttlangTTKernelL1AddrTypeGet(MlirContext ctx) {
  return wrap(L1AddrType::get(unwrap(ctx)));
}

MlirType ttlangTTKernelL1AddrPtrTypeGet(MlirContext ctx) {
  return wrap(L1AddrPtrType::get(unwrap(ctx), /*elementWidth=*/32));
}

MlirType ttlangTTKernelDataFormatTypeGet(MlirContext ctx) {
  return wrap(DataFormatType::get(unwrap(ctx)));
}

MlirType ttlangTTKernelTensorAccessorArgsTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorArgsType::get(unwrap(ctx)));
}

MlirType ttlangTTKernelTensorAccessorTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorType::get(unwrap(ctx)));
}

MlirType ttlangTTKernelTensorAccessorPageMappingTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorPageMappingType::get(unwrap(ctx)));
}

MlirAttribute ttlangTTKernelArgAttrGet(MlirContext ctx, uint32_t argTypeValue,
                                       size_t operandIndex, bool isUniform) {
  return wrap(ArgAttr::get(unwrap(ctx), static_cast<ArgType>(argTypeValue),
                           operandIndex));
}

MlirAttribute ttlangTTKernelArgSpecAttrGet(MlirContext ctx,
                                           MlirAttribute *rtArgs,
                                           size_t rtArgsSize,
                                           MlirAttribute *ctArgs,
                                           size_t ctArgsSize) {
  std::vector<ArgAttr> _rt_args, _ct_args;

  for (size_t i = 0; i < rtArgsSize; i++) {
    _rt_args.emplace_back(mlir::cast<ArgAttr>(unwrap(rtArgs[i])));
  }

  for (size_t i = 0; i < ctArgsSize; i++) {
    _ct_args.emplace_back(mlir::cast<ArgAttr>(unwrap(ctArgs[i])));
  }

  return wrap(ArgSpecAttr::get(unwrap(ctx), _rt_args, _ct_args));
}
