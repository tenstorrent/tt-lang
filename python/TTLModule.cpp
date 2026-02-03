// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/CAPI/IR.h"

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
             int64_t bufferFactor) {
            return wrap(CircularBufferType::get(
                unwrap(ctx), shape, unwrap(elementType), bufferFactor));
          },
          nb::arg("context"), nb::arg("shape"), nb::arg("element_type"),
          nb::arg("buffer_factor"))
      .def_prop_ro("shape",
                   [](CircularBufferType &self) {
                     return std::vector<int64_t>(self.getShape().begin(),
                                                 self.getShape().end());
                   })
      .def_prop_ro(
          "element_type",
          [](CircularBufferType &self) { return wrap(self.getElementType()); })
      .def_prop_ro("buffer_factor", &CircularBufferType::getBufferFactor);

  //===--------------------------------------------------------------------===//
  // PipeType
  //===--------------------------------------------------------------------===//

  tt_type_class<PipeType>(m, "PipeType")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t srcX, int64_t srcY, int64_t dstStartX,
             int64_t dstStartY, int64_t dstEndX, int64_t dstEndY) {
            return wrap(PipeType::get(unwrap(ctx), srcX, srcY, dstStartX,
                                      dstStartY, dstEndX, dstEndY));
          },
          nb::arg("context"), nb::arg("src_x"), nb::arg("src_y"),
          nb::arg("dst_start_x"), nb::arg("dst_start_y"), nb::arg("dst_end_x"),
          nb::arg("dst_end_y"))
      .def_prop_ro("src_x", &PipeType::getSrcX)
      .def_prop_ro("src_y", &PipeType::getSrcY)
      .def_prop_ro("dst_start_x", &PipeType::getDstStartX)
      .def_prop_ro("dst_start_y", &PipeType::getDstStartY)
      .def_prop_ro("dst_end_x", &PipeType::getDstEndX)
      .def_prop_ro("dst_end_y", &PipeType::getDstEndY)
      .def("is_unicast", &PipeType::isUnicast)
      .def("is_multicast", &PipeType::isMulticast);
}
