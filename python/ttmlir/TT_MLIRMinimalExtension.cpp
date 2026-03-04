// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Nanobind extension for the minimal set of tt-mlir dialects.
// This replaces tt-mlir's _ttmlir module with one that only includes
// TTCore, TTKernel, and TTMetal.

// TTCore headers must come before TTMLIRMinimalModule.h because Utils.h uses
// ttmlir::utils which is ambiguous after mlir::ttmlir namespace is declared.
#include "TTMLIRMinimalModule.h"
#include "mlir-c/Pass.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <cstdlib>

// Custom signal handler for clean exit after stack trace
static void cleanExitSignalHandler(void *cookie) { _exit(1); }

NB_MODULE(_ttmlir, m) {
  m.doc() = "Minimal tt-mlir Python bindings (TTCore + TTKernel + TTMetal)";

  // Enable PrettyStackTrace infrastructure
  static llvm::PrettyStackTraceProgram prettyStackTraceProgram(0, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal("");
  llvm::sys::AddSignalHandler(cleanExitSignalHandler, nullptr);

  // Register TTKernel passes (TTCore passes not needed in tt-lang pipelines)
  mlir::tt::ttkernel::registerPasses();

  m.def(
      "enable_pretty_stack_traces",
      [](nb::object pmObj) {
        MlirPassManager pm = mlirPythonCapsuleToPassManager(pmObj.ptr());
        if (mlirPassManagerIsNull(pm)) {
          throw std::runtime_error("Invalid PassManager capsule");
        }
        // Note: pass tracking requires PassTracker which we don't include
        // in the minimal build. This is a no-op placeholder.
      },
      nb::arg("pass_manager"),
      "Enable pass tracking for crash diagnostics (minimal build).");

  // Register dialects into a dialect registry (for site initialization)
  m.def(
      "register_dialects",
      [](MlirDialectRegistry _registry) {
        mlir::DialectRegistry *registry = unwrap(_registry);
        registry->insert<mlir::tt::ttcore::TTCoreDialect>();
        registry->insert<mlir::tt::ttkernel::TTKernelDialect>();
        registry->insert<mlir::tt::ttmetal::TTMetalDialect>();
      },
      nb::arg("dialectRegistry"),
      "Register minimal tt-mlir dialects into a registry.");

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        mlir::DialectRegistry registry;
        registry.insert<mlir::tt::ttcore::TTCoreDialect>();
        registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
        registry.insert<mlir::tt::ttmetal::TTMetalDialect>();

        mlir::MLIRContext *mlirContext = unwrap(context);
        mlirContext->appendDialectRegistry(registry);
        if (load) {
          mlirContext->loadAllAvailableDialects();
        }
      },
      nb::arg("context"), nb::arg("load") = true,
      "Register and optionally load minimal tt-mlir dialects.");

  // TTCore dialect submodule
  auto tt_ir = m.def_submodule("tt_ir", "TTCore IR Bindings");
  mlir::ttmlir::python::populateTTModule(tt_ir);

  // TTKernel dialect submodule
  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR Bindings");
  mlir::ttmlir::python::populateTTKernelModule(ttkernel_ir);

  // Passes submodule
  auto passes =
      m.def_submodule("passes", "Python-Bound Passes & Transformations");
  mlir::ttmlir::python::populatePassesModule(passes);
}
