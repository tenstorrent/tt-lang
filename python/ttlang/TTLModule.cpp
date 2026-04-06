// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/CAPI/IR.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::tt::ttl;

//===----------------------------------------------------------------------===//
// TTL Module Population
//===----------------------------------------------------------------------===//

void populateTTLModule(nb::module_ &m) {
  m.doc() = "TTL (TT-Lang) dialect Python bindings";

  //===--------------------------------------------------------------------===//
  // SliceAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<SliceAttr>(m, "SliceAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t start, int64_t stop, int64_t step) {
            return wrap(SliceAttr::get(unwrap(ctx), start, stop, step));
          },
          nb::arg("context"), nb::arg("start"), nb::arg("stop"),
          nb::arg("step"))
      .def_prop_ro("start", &SliceAttr::getStart)
      .def_prop_ro("stop", &SliceAttr::getStop)
      .def_prop_ro("step", &SliceAttr::getStep);

  //===--------------------------------------------------------------------===//
  // CircularBufferType
  //===--------------------------------------------------------------------===//

  tt_type_class<CircularBufferType>(m, "CircularBufferType")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> shape, MlirType elementType,
             int64_t blockCount) {
            return wrap(CircularBufferType::get(
                unwrap(ctx), shape, unwrap(elementType), blockCount));
          },
          nb::arg("context"), nb::arg("shape"), nb::arg("element_type"),
          nb::arg("block_count"))
      .def_prop_ro("shape",
                   [](CircularBufferType &self) {
                     return std::vector<int64_t>(self.getShape().begin(),
                                                 self.getShape().end());
                   })
      .def_prop_ro(
          "element_type",
          [](CircularBufferType &self) { return wrap(self.getElementType()); })
      .def_prop_ro("block_count", &CircularBufferType::getBlockCount);

  //===--------------------------------------------------------------------===//
  // LayoutAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<LayoutAttr>(m, "LayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> shape, MlirType elementType,
             uint32_t bufferType, std::vector<int64_t> grid,
             std::optional<uint32_t> memLayout) {
            auto memoryLayout =
                memLayout.has_value()
                    ? static_cast<TensorMemoryLayout>(*memLayout)
                    : TensorMemoryLayout::Interleaved;
            return wrap(LayoutAttr::get(unwrap(ctx), shape, unwrap(elementType),
                                        static_cast<BufferType>(bufferType),
                                        grid, memoryLayout));
          },
          nb::arg("ctx"), nb::arg("shape"), nb::arg("element_type"),
          nb::arg("buffer_type"), nb::arg("grid"),
          nb::arg("memory_layout") = nb::none())
      .def_prop_ro("shape",
                   [](LayoutAttr &self) {
                     auto s = self.getShape();
                     return std::vector<int64_t>(s.begin(), s.end());
                   })
      .def_prop_ro("element_type",
                   [](LayoutAttr &self) { return wrap(self.getElementType()); })
      .def_prop_ro("buffer_type",
                   [](LayoutAttr &self) {
                     return static_cast<uint32_t>(self.getBufferType());
                   })
      .def_prop_ro("grid",
                   [](LayoutAttr &self) {
                     auto g = self.getGrid();
                     return std::vector<int64_t>(g.begin(), g.end());
                   })
      .def_prop_ro("memory_layout", [](LayoutAttr &self) {
        return static_cast<uint32_t>(self.getMemoryLayout());
      });
}
