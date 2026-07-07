// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Minimal version of ttmlir/Bindings/Python/TTMLIRModule.h.
// Provides tt_attribute_class and tt_type_class helpers without pulling in
// the full tt-mlir registration machinery (TTNN, StableHLO, etc.).

#ifndef TTLANG_PYTHON_TTMLIR_MINIMAL_MODULE_H
#define TTLANG_PYTHON_TTMLIR_MINIMAL_MODULE_H

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include <nanobind/stl/variant.h>
#include <variant>

namespace nb = nanobind;

namespace mlir::ttmlir::python {

template <typename T>
nb::class_<T> tt_attribute_class(nb::module_ &m, const char *class_name) {
  nb::class_<T> cls(m, class_name);
  cls.def_static("maybe_downcast",
                 [](MlirAttribute attr) -> std::variant<T, nb::object> {
                   auto res = mlir::dyn_cast<T>(unwrap(attr));
                   if (res) {
                     return res;
                   }
                   return nb::none();
                 });
  return cls;
}

template <typename T>
nb::class_<T> tt_type_class(nb::module_ &m, const char *class_name) {
  nb::class_<T> cls(m, class_name);
  cls.def_static("maybe_downcast",
                 [](MlirType type) -> std::variant<T, nb::object> {
                   auto res = mlir::dyn_cast<T>(unwrap(type));
                   if (res) {
                     return res;
                   }
                   return nb::none();
                 });
  return cls;
}

void populateTTModule(nb::module_ &m);
void populateTTKernelModule(nb::module_ &m);
void populatePassesModule(nb::module_ &m);

} // namespace mlir::ttmlir::python

#endif // TTLANG_PYTHON_TTMLIR_MINIMAL_MODULE_H
