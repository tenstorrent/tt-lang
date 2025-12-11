// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Transforms/Passes.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings for TTL dialect";

  // Register tt-lang passes and TTL dialect when the module is loaded.
  mlir::tt::d2m::registerD2MPasses();

  // Register TTL dialect with any Context that loads this module
  m.def(
      "register_ttl_dialect",
      [](MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        ctx->loadDialect<mlir::tt::ttl::TTLDialect>();
      },
      nb::arg("context"),
      "Register and load the TTL dialect into the given context");

  // Register dialects into a dialect registry (for site initialization)
  m.def(
      "register_dialects",
      [](MlirDialectRegistry _registry) {
        mlir::DialectRegistry *registry = unwrap(_registry);
        registry->insert<mlir::tt::ttl::TTLDialect>();
      },
      nb::arg("dialectRegistry"),
      "Register all tt-lang dialects into the given dialect registry");

  // Create TTL dialect submodule matching tt-mlir naming.
  auto ttlIrModule = m.def_submodule("ttl_ir", "TTL dialect bindings");
  populateTTLModule(ttlIrModule);

  // Keep `ttl` alias for compatibility with any early callers.
  m.attr("ttl") = ttlIrModule;
}
