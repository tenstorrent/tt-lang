// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Nanobind extension for the minimal set of tt-mlir dialects.
// This replaces tt-mlir's _ttmlir module with one that only includes
// TTCore, TTKernel, and TTMetal.

#include "Dialects.h"
#include "TTMLIRMinimalModule.h"
#include "mlir-c/Pass.h"

NB_MODULE(_ttmlir, m) {
  m.doc() = "Minimal tt-mlir Python bindings (TTCore + TTKernel + TTMetal)";

  // Register TTKernel transform passes and TTKernelToEmitC via CAPI
  ttmlirMinimalRegisterPasses();

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
      [](MlirDialectRegistry registry) {
        ttmlirMinimalRegisterAllDialects(registry);
      },
      nb::arg("dialectRegistry"),
      "Register minimal tt-mlir dialects into a registry.");

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectRegistry registry = mlirDialectRegistryCreate();
        ttmlirMinimalRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(context, registry);
        mlirDialectRegistryDestroy(registry);
        if (load) {
          mlirContextLoadAllAvailableDialects(context);
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
