// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang-c/Dialects.h"

#include "mlir-c/Pass.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings (ttl, ttcore, ttkernel dialects)";

  ttlangRegisterPasses();

  // Register every tt-lang dialect (ttl, ttcore, ttkernel) plus the minimal
  // upstream MLIR dialects the pipeline uses.
  m.def(
      "register_dialects",
      [](MlirDialectRegistry registry) {
        ttlangRegisterTTLDialect(registry);
        ttlangRegisterTTCoreDialect(registry);
        ttlangRegisterTTKernelDialect(registry);
        ttlangRegisterUpstreamDialects(registry);
      },
      nb::arg("dialectRegistry"),
      "Register all tt-lang dialects into the given dialect registry");

  m.def(
      "enable_pretty_stack_traces",
      [](nb::object pmObj) {
        MlirPassManager pm = mlirPythonCapsuleToPassManager(pmObj.ptr());
        if (mlirPassManagerIsNull(pm)) {
          throw std::runtime_error("Invalid PassManager capsule");
        }
      },
      nb::arg("pass_manager"),
      "Enable pass tracking for crash diagnostics (no-op placeholder).");

  // TTL dialect submodule.
  auto ttlIrModule = m.def_submodule("ttl_ir", "TTL dialect bindings");
  populateTTLModule(ttlIrModule);

  // TTCore dialect submodule.
  auto tt_ir = m.def_submodule("tt_ir", "TTCore IR bindings");
  populateTTModule(tt_ir);

  // TTKernel dialect submodule.
  auto ttkernel_ir = m.def_submodule("ttkernel_ir", "TTKernel IR bindings");
  populateTTKernelModule(ttkernel_ir);

  // Passes submodule (ttkernel-to-cpp helpers).
  auto passes = m.def_submodule("passes", "Python-bound passes and transforms");
  populatePassesModule(passes);
}
