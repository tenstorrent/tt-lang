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
  // PipeRecordAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<PipeRecordAttr>(m, "PipeRecordAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t srcX, int64_t srcY, int64_t dstStartX,
             int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
             bool isMulticast) {
            return wrap(PipeRecordAttr::get(unwrap(ctx), srcX, srcY, dstStartX,
                                            dstStartY, dstEndX, dstEndY,
                                            isMulticast));
          },
          nb::arg("context"), nb::arg("src_x"), nb::arg("src_y"),
          nb::arg("dst_start_x"), nb::arg("dst_start_y"), nb::arg("dst_end_x"),
          nb::arg("dst_end_y"), nb::arg("is_multicast") = false)
      .def_prop_ro("src_x", &PipeRecordAttr::getSrcX)
      .def_prop_ro("src_y", &PipeRecordAttr::getSrcY)
      .def_prop_ro("dst_start_x", &PipeRecordAttr::getDstStartX)
      .def_prop_ro("dst_start_y", &PipeRecordAttr::getDstStartY)
      .def_prop_ro("dst_end_x", &PipeRecordAttr::getDstEndX)
      .def_prop_ro("dst_end_y", &PipeRecordAttr::getDstEndY)
      .def_prop_ro("is_multicast", &PipeRecordAttr::getIsMulticast);

  //===--------------------------------------------------------------------===//
  // PipeNetRecordsAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<PipeNetRecordsAttr>(m, "PipeNetRecordsAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t pipeNetId,
             std::optional<std::string> pipeNetName,
             std::vector<MlirAttribute> pipes) {
            SmallVector<PipeRecordAttr> records;
            records.reserve(pipes.size());
            for (MlirAttribute attr : pipes) {
              records.push_back(mlir::cast<PipeRecordAttr>(unwrap(attr)));
            }
            StringAttr nameAttr;
            if (pipeNetName.has_value()) {
              nameAttr = StringAttr::get(unwrap(ctx), *pipeNetName);
            }
            return wrap(PipeNetRecordsAttr::get(unwrap(ctx), pipeNetId,
                                                nameAttr, records));
          },
          nb::arg("context"), nb::arg("pipe_net_id"),
          nb::arg("pipe_net_name").none() = nb::none(), nb::arg("pipes"))
      .def_prop_ro("pipe_net_id", &PipeNetRecordsAttr::getPipeNetId)
      .def_prop_ro("pipe_net_name",
                   [](PipeNetRecordsAttr &self) -> std::optional<std::string> {
                     if (auto nameAttr = self.getPipeNetName()) {
                       return nameAttr.getValue().str();
                     }
                     return std::nullopt;
                   })
      .def_prop_ro("pipes", [](PipeNetRecordsAttr &self) {
        std::vector<MlirAttribute> out;
        out.reserve(self.getPipes().size());
        for (PipeRecordAttr record : self.getPipes()) {
          out.push_back(wrap(record));
        }
        return out;
      });

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

  //===--------------------------------------------------------------------===//
  // PipeType
  //===--------------------------------------------------------------------===//

  tt_type_class<PipeType>(m, "PipeType")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t srcX, int64_t srcY, int64_t dstStartX,
             int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
             int64_t pipeNetId) {
            return wrap(PipeType::get(unwrap(ctx), srcX, srcY, dstStartX,
                                      dstStartY, dstEndX, dstEndY, pipeNetId));
          },
          nb::arg("context"), nb::arg("src_x"), nb::arg("src_y"),
          nb::arg("dst_start_x"), nb::arg("dst_start_y"), nb::arg("dst_end_x"),
          nb::arg("dst_end_y"), nb::arg("pipe_net_id"))
      .def_prop_ro("src_x", &PipeType::getSrcX)
      .def_prop_ro("src_y", &PipeType::getSrcY)
      .def_prop_ro("dst_start_x", &PipeType::getDstStartX)
      .def_prop_ro("dst_start_y", &PipeType::getDstStartY)
      .def_prop_ro("dst_end_x", &PipeType::getDstEndX)
      .def_prop_ro("dst_end_y", &PipeType::getDstEndY)
      .def_prop_ro("pipe_net_id", &PipeType::getPipeNetId)
      .def("is_unicast", &PipeType::isUnicast)
      .def("is_multicast", &PipeType::isMulticast);

  //===--------------------------------------------------------------------===//
  // SelectedPipe types
  //===--------------------------------------------------------------------===//

  tt_type_class<SelectedPipeSrcType>(m, "SelectedPipeSrcType")
      .def_static(
          "get",
          [](MlirContext ctx) {
            return wrap(SelectedPipeSrcType::get(unwrap(ctx)));
          },
          nb::arg("context"));

  tt_type_class<SelectedPipeDstType>(m, "SelectedPipeDstType")
      .def_static(
          "get",
          [](MlirContext ctx) {
            return wrap(SelectedPipeDstType::get(unwrap(ctx)));
          },
          nb::arg("context"));
}
