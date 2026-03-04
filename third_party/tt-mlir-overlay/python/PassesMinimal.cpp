// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal Passes module for tt-lang
// Contains only the pass functions that tt-lang actually uses:
// - ttkernel_to_cpp / ttkernel_to_cpp_by_name / ttkernel_to_cpp_file
// - get_ttkernel_names
// - get_ttkernel_arg_spec
// - pykernel_compile_pipeline
// - translate_to_cpp

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace mlir::ttmlir::python {

void populatePassesModuleMinimal(nb::module_ &m) {
  // Register the passes we need
  mlir::tt::ttkernel::registerTTKernelPasses();
  mlir::tt::registerConversionPasses();

  m.def(
      "ttkernel_to_cpp",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        // Convert to EmitC
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }

        // Translate to C++
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::tt::ttkernel::translateTopLevelKernelsToCpp(
                mlir::cast<ModuleOp>(moduleOp), output_stream))) {
          throw std::runtime_error("Failed to generate cpp");
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"));

  m.def(
      "ttkernel_to_cpp_file",
      [](MlirModule module, const std::string &filepath) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        // Convert to EmitC
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }

        // Translate each kernel to C++ and dump to file
        moduleOp->walk([&](func::FuncOp entry) {
          if (!entry->hasAttr((mlir::tt::ttkernel::ThreadTypeAttr::name))) {
            return;
          }

          std::string out_path =
              filepath + "/" + std::string(entry.getName()) + ".cpp";
          std::error_code fileError;
          llvm::raw_fd_ostream out_file(out_path, fileError);
          if (fileError || failed(mlir::tt::ttkernel::translateKernelFuncToCpp(
                               entry, out_file))) {
            throw std::runtime_error("Failed to generate cpp files");
          }
        });
      },
      nb::arg("module"), nb::arg("filepath"));

  m.def(
      "ttkernel_to_cpp_by_name",
      [](MlirModule module, std::string symbolName) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        // Convert to EmitC
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }

        // Translate single kernel to C++
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::tt::ttkernel::translateTopLevelKernelToCpp(
                mlir::cast<ModuleOp>(moduleOp), output_stream, symbolName))) {
          throw std::runtime_error("Failed to generate cpp for kernel: " +
                                   symbolName);
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"), nb::arg("symbol_name"));

  m.def(
      "get_ttkernel_names",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        auto mod = mlir::cast<ModuleOp>(moduleOp);

        // Vector of (kernel_name, thread_type) tuples for each kernel function
        std::vector<std::tuple<std::string, std::string>> kernels;
        mod.walk([&](func::FuncOp funcOp) {
          if (auto threadTypeAttr =
                  funcOp->getAttrOfType<mlir::tt::ttkernel::ThreadTypeAttr>(
                      mlir::tt::ttkernel::ThreadTypeAttr::name)) {
            kernels.emplace_back(funcOp.getName().str(),
                                 mlir::tt::ttkernel::stringifyThreadType(
                                     threadTypeAttr.getValue())
                                     .str());
          }
        });
        return kernels;
      },
      nb::arg("module"));

  m.def(
      "get_ttkernel_arg_spec",
      [](MlirModule module,
         std::string kernelName) -> std::optional<MlirAttribute> {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        auto mod = mlir::cast<ModuleOp>(moduleOp);

        std::optional<MlirAttribute> result;
        mod.walk([&](func::FuncOp funcOp) {
          if (funcOp.getName() == kernelName) {
            if (auto argSpecAttr =
                    funcOp->getAttrOfType<mlir::tt::ttkernel::ArgSpecAttr>(
                        mlir::tt::ttkernel::ArgSpecAttr::name)) {
              result = wrap(argSpecAttr);
            }
          }
        });
        return result;
      },
      nb::arg("module"), nb::arg("kernel_name"));

  m.def(
      "pykernel_compile_pipeline",
      [](MlirModule module, std::string options) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("pykernel-compile-pipeline");

        if (!pipeline) {
          throw std::runtime_error(
              "pykernel-compile-pipeline not found - is it registered?");
        }

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "translate_to_cpp",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        // Translate to C++
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::emitc::translateToCpp(
                mlir::cast<ModuleOp>(moduleOp), output_stream))) {
          throw std::runtime_error("Failed to generate cpp");
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"));
}

} // namespace mlir::ttmlir::python
