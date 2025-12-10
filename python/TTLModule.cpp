// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang-c/TTLAttrs.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::tt::ttl;

//===----------------------------------------------------------------------===//
// SliceAttr C++ wrapper for Python bindings
//===----------------------------------------------------------------------===//

namespace {
/// Wrapper struct that holds an MlirAttribute and provides type-safe access
/// to SliceAttr methods through the C-API.
struct PySliceAttr {
  MlirAttribute attr;

  /// Check if an attribute is a SliceAttr
  static bool isaFunction(MlirAttribute attr) {
    return ttlangMlirAttributeIsATTLSliceAttr(attr);
  }

  /// Get the start value
  int64_t getStart() const { return ttlangTTLSliceAttrGetStart(attr); }

  /// Get the stop value
  int64_t getStop() const { return ttlangTTLSliceAttrGetStop(attr); }

  /// Get the step value
  int64_t getStep() const { return ttlangTTLSliceAttrGetStep(attr); }
};
} // namespace

//===----------------------------------------------------------------------===//
// TTL Module Population
//===----------------------------------------------------------------------===//

void populateTTLModule(nb::module_ &m) {
  m.doc() = "TTL (TT-Lang) dialect Python bindings";

  //===--------------------------------------------------------------------===//
  // SliceAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<PySliceAttr>(m, "SliceAttr",
                                   "Slice specification for core ranges")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t start, int64_t stop, int64_t step) {
            return PySliceAttr{ttlangTTLSliceAttrGet(ctx, start, stop, step)};
          },
          nb::arg("ctx"), nb::arg("start"), nb::arg("stop"), nb::arg("step"),
          "Create a SliceAttr with the given start, stop, and step values.\n"
          "Represents a half-open interval [start, stop) with the given step.")
      .def_prop_ro("start", &PySliceAttr::getStart,
                   "The start index (inclusive)")
      .def_prop_ro("stop", &PySliceAttr::getStop, "The stop index (exclusive)")
      .def_prop_ro("step", &PySliceAttr::getStep, "The step value")
      .def("__repr__",
           [](const PySliceAttr &self) {
             return "<SliceAttr start=" + std::to_string(self.getStart()) +
                    " stop=" + std::to_string(self.getStop()) +
                    " step=" + std::to_string(self.getStep()) + ">";
           });
}
