// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Minimal pass functions for tt-lang Python API.
// Provides ttkernel_to_cpp_by_name and related helpers.

#include "TTMLIRMinimalModule.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir::ttmlir::python {

void populatePassesModule(nb::module_ &m) {

  m.def(
      "ttkernel_to_cpp_by_name",
      [](MlirModule module, const std::string &kernelName) -> std::string {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        // Convert to EmitC first (matching upstream behavior)
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run TTKernelToEmitC pass");
        }

        // Translate single kernel to C++
        std::string output;
        llvm::raw_string_ostream os(output);
        if (mlir::failed(mlir::tt::ttkernel::translateTopLevelKernelToCpp(
                mlir::cast<mlir::ModuleOp>(moduleOp), os, kernelName))) {
          throw std::runtime_error("Failed to translate kernel '" + kernelName +
                                   "' to C++");
        }
        return output;
      },
      nb::arg("module"), nb::arg("kernel_name"),
      "Translate a named TTKernel function to C++ string.");

  m.def(
      "get_ttkernel_names",
      [](MlirModule module) -> std::vector<std::string> {
        mlir::ModuleOp mod = llvm::cast<mlir::ModuleOp>(unwrap(module));
        std::vector<std::string> names;
        mod.walk([&](mlir::func::FuncOp funcOp) {
          if (funcOp->hasAttr("ttkernel.thread_type")) {
            names.push_back(funcOp.getName().str());
          }
        });
        return names;
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
