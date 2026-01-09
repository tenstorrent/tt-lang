// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal Python bindings for tt-lang
// This is a minimal version of TTMLIRModule.cpp that only registers
// the dialects needed for tt-lang (TTCore, TTKernel, TTMetal, TTIR, D2M)

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <cstdlib>

namespace nb = nanobind;

// Forward declarations for populateXXXModule functions
namespace mlir::ttmlir::python {
void populateTTModule(nb::module_ &m);
void populateTTIRModule(nb::module_ &m);
void populateTTKernelModule(nb::module_ &m);
void populatePassesModuleMinimal(nb::module_ &m);
void populateUtilModule(nb::module_ &m);
} // namespace mlir::ttmlir::python

// Custom signal handler that exits cleanly after stack trace is printed
static void cleanExitSignalHandler(void *cookie) {
  _exit(1);
}

// Minimal dialect registration - only registers what tt-lang needs
static void registerMinimalDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
  registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
  registry.insert<mlir::tt::ttir::TTIRDialect>();
}

NB_MODULE(_ttmlir, m) {
  m.doc() = "ttmlir minimal python extension for tt-lang";

  // Enable PrettyStackTrace infrastructure
  static llvm::PrettyStackTraceProgram prettyStackTraceProgram(0, nullptr);

  // Install LLVM signal handlers for stack traces and clean exit
  llvm::sys::PrintStackTraceOnErrorSignal("");
  llvm::sys::AddSignalHandler(cleanExitSignalHandler, nullptr);

  // Create specialized register_dialects function for site initialize
  m.def(
      "register_dialects",
      [](MlirDialectRegistry _registry) {
        mlir::DialectRegistry *registry = unwrap(_registry);
        registerMinimalDialects(*registry);
      },
      nb::arg("dialectRegistry"));

  // Register dialect function (legacy API)
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        mlir::DialectRegistry registry;
        registerMinimalDialects(registry);

        mlir::MLIRContext *mlirContext = unwrap(context);
        mlirContext->appendDialectRegistry(registry);

        if (load) {
          mlirContext->loadAllAvailableDialects();
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  // Create submodules for dialect bindings
  auto tt_ir = m.def_submodule("tt_ir", "TT IR Bindings");
  mlir::ttmlir::python::populateTTModule(tt_ir);

  auto ttir_ir = m.def_submodule("ttir_ir", "TTIR IR Bindings");
  mlir::ttmlir::python::populateTTIRModule(ttir_ir);

  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);

  auto passes = m.def_submodule("passes", "Python-Bound Passes & Transformations");
  mlir::ttmlir::python::populatePassesModuleMinimal(passes);

  auto util = m.def_submodule("util", "Python-Bound Utilities & Helpers");
  mlir::ttmlir::python::populateUtilModule(util);
}
