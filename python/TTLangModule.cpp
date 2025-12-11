// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "ttlang/Transforms/Passes.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings";

  // Register passes
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

  // Create TTL dialect submodule
  auto ttl_m = m.def_submodule("ttl", "TTL dialect bindings");
  populateTTLModule(ttl_m);
}
