// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H
#define TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir-c/IR.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <variant>

namespace nb = nanobind;

//===----------------------------------------------------------------------===//
// Type/Attribute Class Helper Functions
//===----------------------------------------------------------------------===//
// These functions provide standardized Python bindings for MLIR types and
// attributes, following the pattern used in tt-mlir.

/// Helper function to create a nanobind class for MLIR attributes with
/// automatic downcasting support.
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

/// Helper function to create a nanobind class for MLIR types with
/// automatic downcasting support.
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

//===----------------------------------------------------------------------===//
// Dialect Module Population Functions
//===----------------------------------------------------------------------===//

/// Populates the TTL dialect Python bindings.
void populateTTLModule(nb::module_ &m);

#endif // TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H
