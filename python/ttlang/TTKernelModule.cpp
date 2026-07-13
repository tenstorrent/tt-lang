// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "mlir/CAPI/IR.h"
#include "ttlang-c/TTKernelTypes.h"

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir;

void populateTTKernelModule(nb::module_ &m) {
  tt_type_class<tt::ttkernel::CBType>(m, "CBType")
      .def_static("get",
                  [](MlirContext ctx, MlirType memrefType) {
                    return ttlangTTKernelCBTypeGet(ctx, memrefType);
                  })
      .def_static("cast", [](MlirType &ty) {
        return mlir::cast<tt::ttkernel::CBType>(unwrap(ty));
      });

  tt_type_class<tt::ttkernel::LocalSemaphoreType>(m, "LocalSemaphoreType")
      .def_static("get", &ttlangTTKernelLocalSemaphoreTypeGet);

  tt_type_class<tt::ttkernel::NocAddrType>(m, "NocAddrType")
      .def_static("get", &ttlangTTKernelNocAddrTypeGet);

  tt_attribute_class<tt::ttkernel::ThreadTypeAttr>(m, "ThreadTypeAttr")
      .def_prop_ro_static("name",
                          [](nb::handle /*unused*/) {
                            return std::string(
                                tt::ttkernel::ThreadTypeAttr::name);
                          })
      .def_static("get", [](MlirContext ctx, std::string threadTypeStr) {
        tt::ttkernel::ThreadType threadType;
        if (threadTypeStr == "compute") {
          threadType = tt::ttkernel::ThreadType::Compute;
        } else if (threadTypeStr == "noc") {
          threadType = tt::ttkernel::ThreadType::Noc;
        } else {
          throw std::runtime_error("Unknown thread type " + threadTypeStr);
        }

        return ttlangTTKernelThreadTypeAttrGet(
            ctx, static_cast<std::underlying_type_t<tt::ttkernel::ThreadType>>(
                     threadType));
      });

  tt_attribute_class<tt::ttkernel::ReduceTypeAttr>(m, "ReduceTypeAttr")
      .def_static("get", &ttlangTTKernelReduceTypeAttrGet)
      .def_prop_ro("value", &tt::ttkernel::ReduceTypeAttr::getValue);

  tt_attribute_class<tt::ttkernel::ReduceDimAttr>(m, "ReduceDimAttr")
      .def_static("get", &ttlangTTKernelReduceDimAttrGet)
      .def_prop_ro("value", &tt::ttkernel::ReduceDimAttr::getValue);

  tt_type_class<tt::ttkernel::L1AddrType>(m, "L1AddrType")
      .def_static("get", &ttlangTTKernelL1AddrTypeGet);

  tt_type_class<tt::ttkernel::L1AddrPtrType>(m, "L1AddrPtrType")
      .def_static("get", &ttlangTTKernelL1AddrPtrTypeGet);

  tt_type_class<tt::ttkernel::DataFormatType>(m, "DataFormatType")
      .def_static("get", &ttlangTTKernelDataFormatTypeGet);

  tt_type_class<tt::ttkernel::TensorAccessorArgsType>(m,
                                                      "TensorAccessorArgsType")
      .def_static("get", &ttlangTTKernelTensorAccessorArgsTypeGet);

  tt_type_class<tt::ttkernel::TensorAccessorType>(m, "TensorAccessorType")
      .def_static("get", &ttlangTTKernelTensorAccessorTypeGet);

  tt_type_class<tt::ttkernel::TensorAccessorPageMappingType>(
      m, "TensorAccessorPageMappingType")
      .def_static("get", &ttlangTTKernelTensorAccessorPageMappingTypeGet);

  tt_attribute_class<tt::ttkernel::ArgAttr>(m, "ArgAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t argTypeValue, size_t operandIndex,
             bool isUniform = true) {
            return wrap(tt::ttkernel::ArgAttr::get(
                unwrap(ctx), static_cast<tt::ttkernel::ArgType>(argTypeValue),
                operandIndex, isUniform));
          },
          nb::arg("ctx"), nb::arg("argTypeValue"), nb::arg("operandIndex"),
          nb::arg("isUniform") = true)
      .def_prop_ro("is_uniform", &tt::ttkernel::ArgAttr::getIsUniform)
      .def_prop_ro("operand_index", &tt::ttkernel::ArgAttr::getOperandIndex)
      .def_prop_ro("arg_type_as_value", [](tt::ttkernel::ArgAttr &self) {
        return static_cast<uint32_t>(self.getArgType());
      });

  tt_attribute_class<tt::ttkernel::ArgSpecAttr>(m, "ArgSpecAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<MlirAttribute> rtArgs,
                     std::vector<MlirAttribute> ctArgs) {
                    std::vector<tt::ttkernel::ArgAttr> _rt_args, _ct_args;

                    for (const auto &x : rtArgs) {
                      _rt_args.emplace_back(
                          mlir::cast<tt::ttkernel::ArgAttr>(unwrap(x)));
                    }

                    for (const auto &x : ctArgs) {
                      _ct_args.emplace_back(
                          mlir::cast<tt::ttkernel::ArgAttr>(unwrap(x)));
                    }

                    return wrap(tt::ttkernel::ArgSpecAttr::get(
                        unwrap(ctx), _rt_args, _ct_args));
                  })
      .def_prop_ro_static("name",
                          [](nb::handle) {
                            return std::string(tt::ttkernel::ArgSpecAttr::name);
                          })
      .def_prop_ro("rt_args",
                   [](tt::ttkernel::ArgSpecAttr &self) {
                     std::vector<tt::ttkernel::ArgAttr> result;
                     for (const auto &arg : self.getRtArgs()) {
                       result.push_back(arg);
                     }
                     return result;
                   })
      .def_prop_ro("ct_args", [](tt::ttkernel::ArgSpecAttr &self) {
        std::vector<tt::ttkernel::ArgAttr> result;
        for (const auto &arg : self.getCtArgs()) {
          result.push_back(arg);
        }
        return result;
      });
}
