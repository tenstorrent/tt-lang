// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal header for tt-mlir Python bindings
// This replaces ttmlir/Bindings/Python/TTMLIRModule.h with only the
// essential templates and forward declarations needed for minimal builds

#ifndef TTMLIR_MINIMAL_MODULE_H
#define TTMLIR_MINIMAL_MODULE_H

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include <nanobind/stl/variant.h>
#include <variant>

namespace nb = nanobind;

namespace mlir::ttmlir::python {

// Helper template for creating attribute classes with maybe_downcast support
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

// Helper template for creating type classes with maybe_downcast support
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

// Forward declarations for populate functions
void populateTTModule(nb::module_ &m);
void populateTTIRModule(nb::module_ &m);
void populateTTKernelModule(nb::module_ &m);
void populatePassesModuleMinimal(nb::module_ &m);
void populateUtilModule(nb::module_ &m);

} // namespace mlir::ttmlir::python

#endif // TTMLIR_MINIMAL_MODULE_H
