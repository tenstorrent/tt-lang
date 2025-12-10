// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H
#define TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H

#include "mlir-c/IR.h"
#include <nanobind/nanobind.h>
#include <optional>

namespace nb = nanobind;

//===----------------------------------------------------------------------===//
// Type/Attribute Class Templates
//===----------------------------------------------------------------------===//
// These templates provide standardized Python bindings for MLIR types and
// attributes, following the pattern used in tt-mlir.

/// Template for binding MLIR attribute classes to Python.
/// Provides automatic downcasting from MlirAttribute to the specific type.
template <typename T>
class tt_attribute_class : public nb::class_<T> {
public:
  using nb::class_<T>::def;
  using nb::class_<T>::def_prop_ro;
  using nb::class_<T>::def_static;

  tt_attribute_class(nb::handle scope, const char *name,
                     const char *descr = nullptr)
      : nb::class_<T>(scope, name, descr) {
    // Provide a maybe_downcast method for type checking and casting
    this->def_static(
        "maybe_downcast",
        [](MlirAttribute attr) -> std::optional<T> {
          if (T::isaFunction(attr)) {
            return T{attr};
          }
          return std::nullopt;
        },
        nb::arg("attr"),
        "Attempts to downcast the given attribute to this type. Returns None "
        "if the attribute is not of this type.");
  }
};

/// Template for binding MLIR type classes to Python.
/// Provides automatic downcasting from MlirType to the specific type.
template <typename T>
class tt_type_class : public nb::class_<T> {
public:
  using nb::class_<T>::def;
  using nb::class_<T>::def_prop_ro;
  using nb::class_<T>::def_static;

  tt_type_class(nb::handle scope, const char *name, const char *descr = nullptr)
      : nb::class_<T>(scope, name, descr) {
    // Provide a maybe_downcast method for type checking and casting
    this->def_static(
        "maybe_downcast",
        [](MlirType type) -> std::optional<T> {
          if (T::isaFunction(type)) {
            return T{type};
          }
          return std::nullopt;
        },
        nb::arg("type"),
        "Attempts to downcast the given type to this type. Returns None if "
        "the type is not of this type.");
  }
};

//===----------------------------------------------------------------------===//
// Dialect Module Population Functions
//===----------------------------------------------------------------------===//

/// Populates the TTL dialect Python bindings.
void populateTTLModule(nb::module_ &m);

#endif // TTLANG_BINDINGS_PYTHON_TTLANGMODULE_H
