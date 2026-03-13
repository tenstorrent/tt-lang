// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Minimal pass functions for tt-lang Python API.
// Provides ttkernel_to_cpp_by_name and related helpers.

#include "Dialects.h"
#include "TTMLIRMinimalModule.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir::ttmlir::python {

void populatePassesModule(nb::module_ &m) {

  m.def(
      "ttkernel_to_cpp_by_name",
      [](MlirModule module, const std::string &kernelName) -> std::string {
        // Convert to EmitC first
        if (!ttmlirMinimalRunTTKernelToEmitC(module)) {
          throw std::runtime_error("Failed to run TTKernelToEmitC pass");
        }

        // Translate single kernel to C++
        char *result =
            ttmlirMinimalTranslateKernelToCpp(module, kernelName.c_str());
        if (!result) {
          throw std::runtime_error("Failed to translate kernel '" + kernelName +
                                   "' to C++");
        }
        std::string output(result);
        free(result);
        return output;
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

} // namespace mlir::ttmlir::python
