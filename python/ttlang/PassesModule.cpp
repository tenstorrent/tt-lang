// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"

#include "ttlang-c/Dialects.h"
#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdlib>

using namespace mlir;
namespace mlirPython = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

static void lowerTTKernelModuleToEmitC(MlirModule module) {
  MlirContext context = mlirOperationGetContext(mlirModuleGetOperation(module));
  mlirPython::PyMlirContext::ErrorCapture errors(
      mlirPython::PyMlirContext::forContext(context));
  if (!ttlangRunTTKernelToEmitC(module)) {
    throw mlirPython::MLIRError(
        "TTKernel-to-EmitC conversion failed; correct the reported IR error",
        errors.take());
  }
}

static std::string translateTTKernelFunction(MlirModule module,
                                             const std::string &kernelName) {
  char *result = ttlangTranslateKernelToCpp(module, kernelName.c_str());
  if (!result) {
    throw std::runtime_error("Failed to translate kernel '" + kernelName +
                             "' to C++");
  }
  std::string output(result);
  std::free(result);
  return output;
}

void populatePassesModule(nb::module_ &m) {

  m.def(
      "ttkernels_to_cpp",
      [](MlirModule module, const std::vector<std::string> &kernelNames) {
        lowerTTKernelModuleToEmitC(module);
        std::vector<std::string> outputs;
        outputs.reserve(kernelNames.size());
        for (const std::string &kernelName : kernelNames) {
          outputs.push_back(translateTTKernelFunction(module, kernelName));
        }
        return outputs;
      },
      nb::arg("module"), nb::arg("kernel_names"),
      "Lower TTKernel to EmitC once and translate the requested kernels.");

  m.def(
      "ttkernel_to_cpp_by_name",
      [](MlirModule module, const std::string &kernelName) -> std::string {
        lowerTTKernelModuleToEmitC(module);
        return translateTTKernelFunction(module, kernelName);
      },
      nb::arg("module"), nb::arg("kernel_name"),
      "Translate a named TTKernel function to C++ string.");

  m.def(
      "get_ttkernel_names",
      [](MlirModule module)
          -> std::vector<std::pair<std::string, std::string>> {
        mlir::ModuleOp mod = llvm::cast<mlir::ModuleOp>(unwrap(module));
        std::vector<std::pair<std::string, std::string>> result;
        mod.walk([&](mlir::func::FuncOp funcOp) {
          auto threadAttr =
              funcOp->getAttrOfType<mlir::tt::ttkernel::ThreadTypeAttr>(
                  mlir::tt::ttkernel::ThreadTypeAttr::name);
          if (threadAttr) {
            result.emplace_back(
                funcOp.getName().str(),
                mlir::tt::ttkernel::stringifyThreadType(threadAttr.getValue())
                    .str());
          }
        });
        return result;
      },
      nb::arg("module"), "Get names of all TTKernel functions in a module.");

  m.def(
      "get_ttkernel_arg_spec",
      [](MlirModule module, const std::string &kernelName) -> nb::object {
        mlir::ModuleOp mod = llvm::cast<mlir::ModuleOp>(unwrap(module));
        mlir::func::FuncOp func =
            mod.lookupSymbol<mlir::func::FuncOp>(kernelName);
        if (!func) {
          return nb::none();
        }
        auto argSpecAttr = func->getAttrOfType<mlir::tt::ttkernel::ArgSpecAttr>(
            mlir::tt::ttkernel::ArgSpecAttr::name);
        if (!argSpecAttr) {
          return nb::none();
        }
        return nb::cast(argSpecAttr);
      },
      nb::arg("module"), nb::arg("kernel_name"),
      "Get the ArgSpecAttr for a named TTKernel function.");
}
