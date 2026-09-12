// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

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

static void lowerTTKernelModuleToEmitC(MlirModule module) {
  if (!ttlangRunTTKernelToEmitC(module)) {
    throw std::runtime_error("Failed to run TTKernelToEmitC pass");
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
        // Per-function conversion would repeatedly rewrite the complete module.
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
                  "ttkernel.thread");
          if (threadAttr) {
            auto threadType = threadAttr.getValue();
            std::string threadStr;
            switch (threadType) {
            case mlir::tt::ttkernel::ThreadType::Noc:
              threadStr = "noc";
              break;
            case mlir::tt::ttkernel::ThreadType::Compute:
              threadStr = "compute";
              break;
            default:
              threadStr = "unknown";
              break;
            }
            result.emplace_back(funcOp.getName().str(), threadStr);
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
        auto argSpecAttr =
            func->getAttrOfType<mlir::tt::ttkernel::ArgSpecAttr>("arg_spec");
        if (!argSpecAttr) {
          return nb::none();
        }
        return nb::cast(argSpecAttr);
      },
      nb::arg("module"), nb::arg("kernel_name"),
      "Get the ArgSpecAttr for a named TTKernel function.");
}
