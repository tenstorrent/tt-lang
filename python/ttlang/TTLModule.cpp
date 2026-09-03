// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Bindings/Python/TTLangModule.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::tt::ttl;

//===----------------------------------------------------------------------===//
// TTL Module Population
//===----------------------------------------------------------------------===//

void populateTTLModule(nb::module_ &m) {
  m.doc() = "TTL (TT-Lang) dialect Python bindings";
  m.attr("PIPE_SYNC_SEMAPHORE_COUNT_ATTR") =
      nb::str(kPipeSyncSemaphoreCountAttrName.data(),
              kPipeSyncSemaphoreCountAttrName.size());
  m.attr("PIPE_GLOBAL_SEMAPHORE_COUNT_ATTR") =
      nb::str(kPipeGlobalSemaphoreCountAttrName.data(),
              kPipeGlobalSemaphoreCountAttrName.size());
  m.attr("PIPE_SRAM_SCRATCH_BYTES_ATTR") =
      nb::str(kPipeSramScratchBytesAttrName.data(),
              kPipeSramScratchBytesAttrName.size());
  m.attr("DFB_RESET_COUNT_ATTR") =
      nb::str(kDFBResetCountAttrName.data(), kDFBResetCountAttrName.size());
  m.attr("PIPE_COMPUTED_ADDRESS_DFB_INDICES_ATTR") =
      nb::str(kPipeComputedAddressDFBIndicesAttrName.data(),
              kPipeComputedAddressDFBIndicesAttrName.size());
  m.attr("FABRIC_ROUTES_ATTR") =
      nb::str(kFabricRoutesAttrName.data(), kFabricRoutesAttrName.size());
  m.attr("FABRIC_RUNTIME_ARG_BASE_COMMON_INDEX_ATTR") =
      nb::str(kFabricRuntimeArgBaseCommonIndexAttrName.data(),
              kFabricRuntimeArgBaseCommonIndexAttrName.size());
  m.attr("FABRIC_MANAGER_INTERVALS_ATTR") =
      nb::str(kFabricManagerIntervalsAttrName.data(),
              kFabricManagerIntervalsAttrName.size());
  m.attr("USED_DFB_INDICES_ATTR") =
      nb::str(kUsedDFBIndicesAttrName.data(), kUsedDFBIndicesAttrName.size());
  m.attr("LOGICAL_KERNEL_ATTR") =
      nb::str(kLogicalKernelAttrName.data(), kLogicalKernelAttrName.size());
  m.attr("CRTA_INDICES_ATTR") =
      nb::str(kCRTAIndicesAttrName.data(), kCRTAIndicesAttrName.size());
  m.attr("LOCAL_TENSOR_INDICES_ATTR") = nb::str(
      kLocalTensorIndicesAttrName.data(), kLocalTensorIndicesAttrName.size());

  nb::enum_<LogicalKernelKind>(m, "LogicalKernelKind")
      .value("Compute", LogicalKernelKind::Compute)
      .value("DataMovement", LogicalKernelKind::DataMovement);

  tt_attribute_class<LogicalKernelAttr>(m, "LogicalKernelAttr")
      .def_static(
          "get",
          [](MlirContext context, LogicalKernelKind kind,
             const std::optional<std::string> &identity,
             const std::optional<std::string> &operation,
             const std::optional<std::string> &role) {
            MLIRContext *cppContext = unwrap(context);
            auto optionalStringAttr =
                [cppContext](
                    const std::optional<std::string> &value) -> StringAttr {
              return value ? StringAttr::get(cppContext, *value) : StringAttr();
            };
            LogicalKernelAttr attribute = LogicalKernelAttr::getChecked(
                [cppContext]() {
                  return emitError(UnknownLoc::get(cppContext));
                },
                cppContext, kind, optionalStringAttr(identity),
                optionalStringAttr(operation), optionalStringAttr(role));
            if (!attribute) {
              throw nb::value_error("invalid logical kernel metadata");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("kind"), nb::arg("identity") = nb::none(),
          nb::arg("operation") = nb::none(), nb::arg("role") = nb::none())
      .def_prop_ro("kind", &LogicalKernelAttr::getKind)
      .def_prop_ro("identity",
                   [](LogicalKernelAttr attribute) {
                     StringAttr identity = attribute.getIdentity();
                     return identity ? std::optional<std::string>(
                                           identity.getValue().str())
                                     : std::nullopt;
                   })
      .def_prop_ro("operation",
                   [](LogicalKernelAttr attribute) {
                     StringAttr operation = attribute.getOperation();
                     return operation ? std::optional<std::string>(
                                            operation.getValue().str())
                                      : std::nullopt;
                   })
      .def_prop_ro("role", [](LogicalKernelAttr attribute) {
        StringAttr role = attribute.getRole();
        return role ? std::optional<std::string>(role.getValue().str())
                    : std::nullopt;
      });

  tt_attribute_class<DispatchConditionAttr>(m, "DispatchConditionAttr")
      .def_static(
          "get",
          [](MlirContext context, int64_t ordinal, MlirType scalarType) {
            MLIRContext *cppContext = unwrap(context);
            DispatchConditionAttr attribute = DispatchConditionAttr::getChecked(
                [cppContext]() {
                  return emitError(UnknownLoc::get(cppContext));
                },
                cppContext, ordinal, unwrap(scalarType));
            if (!attribute) {
              throw nb::value_error("invalid dispatch condition");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("ordinal"), nb::arg("scalar_type"))
      .def_prop_ro("ordinal", &DispatchConditionAttr::getOrdinal)
      .def_prop_ro("scalar_type", &DispatchConditionAttr::getScalarType);

  tt_attribute_class<DFBAllocationGroupAttr>(m, "DFBAllocationGroupAttr")
      .def_static(
          "get",
          [](MlirContext context, int64_t ordinal) {
            MLIRContext *cppContext = unwrap(context);
            DFBAllocationGroupAttr attribute =
                DFBAllocationGroupAttr::getChecked(
                    [cppContext]() {
                      return emitError(UnknownLoc::get(cppContext));
                    },
                    cppContext, ordinal);
            if (!attribute) {
              throw nb::value_error("invalid DFB allocation group");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("ordinal"))
      .def_prop_ro("ordinal", &DFBAllocationGroupAttr::getOrdinal);

  tt_attribute_class<SynchronizedDFBResetAttr>(m, "SynchronizedDFBResetAttr")
      .def_static(
          "get",
          [](MlirContext context, int64_t ordinal,
             const std::vector<MlirAttribute> &participants) {
            MLIRContext *cppContext = unwrap(context);
            SmallVector<LogicalKernelAttr> participantAttrs;
            participantAttrs.reserve(participants.size());
            for (MlirAttribute participant : participants) {
              participantAttrs.push_back(
                  cast<LogicalKernelAttr>(unwrap(participant)));
            }
            SynchronizedDFBResetAttr attribute =
                SynchronizedDFBResetAttr::getCheckedInstance(
                    UnknownLoc::get(cppContext), cppContext, ordinal,
                    participantAttrs);
            if (!attribute) {
              throw nb::value_error("invalid synchronized DFB reset");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("ordinal"), nb::arg("participants"))
      .def_prop_ro("ordinal", &SynchronizedDFBResetAttr::getOrdinal)
      .def_prop_ro("participants", [](SynchronizedDFBResetAttr attribute) {
        std::vector<MlirAttribute> participants;
        participants.reserve(attribute.getParticipants().size());
        for (LogicalKernelAttr participant : attribute.getParticipants()) {
          participants.push_back(wrap(participant));
        }
        return participants;
      });

  tt_attribute_class<DFBReconfigurationAttr>(m, "DFBReconfigurationAttr")
      .def_static(
          "get",
          [](MlirContext context, int64_t ordinal,
             const std::vector<MlirAttribute> &participants) {
            MLIRContext *cppContext = unwrap(context);
            SmallVector<LogicalKernelAttr> participantAttrs;
            participantAttrs.reserve(participants.size());
            for (MlirAttribute participant : participants) {
              participantAttrs.push_back(
                  cast<LogicalKernelAttr>(unwrap(participant)));
            }
            DFBReconfigurationAttr attribute =
                DFBReconfigurationAttr::getCheckedInstance(
                    UnknownLoc::get(cppContext), cppContext, ordinal,
                    participantAttrs);
            if (!attribute) {
              throw nb::value_error("invalid DFB reconfiguration");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("ordinal"), nb::arg("participants"))
      .def_prop_ro("ordinal", &DFBReconfigurationAttr::getOrdinal)
      .def_prop_ro("participants", [](DFBReconfigurationAttr attribute) {
        std::vector<MlirAttribute> participants;
        participants.reserve(attribute.getParticipants().size());
        for (LogicalKernelAttr participant : attribute.getParticipants()) {
          participants.push_back(wrap(participant));
        }
        return participants;
      });

  nb::enum_<ExternalTemplateArgKind>(m, "ExternalTemplateArgKind")
      .value("SignedInteger", ExternalTemplateArgKind::SignedInteger)
      .value("Boolean", ExternalTemplateArgKind::Boolean)
      .value("UnsignedInteger", ExternalTemplateArgKind::UnsignedInteger)
      .value("DFBIndex", ExternalTemplateArgKind::DFBIndex)
      .value("DFBDescriptor", ExternalTemplateArgKind::DFBDescriptor);

  tt_attribute_class<ExternalTemplateArgAttr>(m, "ExternalTemplateArgAttr")
      .def_static(
          "get",
          [](MlirContext context, ExternalTemplateArgKind kind, int64_t value) {
            MLIRContext *cppContext = unwrap(context);
            ExternalTemplateArgAttr attribute =
                ExternalTemplateArgAttr::getChecked(
                    [cppContext]() {
                      return emitError(UnknownLoc::get(cppContext));
                    },
                    cppContext, kind, value);
            if (!attribute) {
              throw nb::value_error("invalid external template argument");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("kind"), nb::arg("value"))
      .def_prop_ro("kind", &ExternalTemplateArgAttr::getKind)
      .def_prop_ro("value", &ExternalTemplateArgAttr::getValue);

  nb::enum_<FabricManagerEffectKind>(m, "FabricManagerEffectKind")
      .value("Acquire", FabricManagerEffectKind::Acquire)
      .value("Use", FabricManagerEffectKind::Use)
      .value("Release", FabricManagerEffectKind::Release)
      .value("Scoped", FabricManagerEffectKind::Scoped);

  tt_attribute_class<FabricManagerEffectAttr>(m, "FabricManagerEffectAttr")
      .def_static(
          "get",
          [](MlirContext context, std::string claim,
             FabricManagerEffectKind kind) {
            MLIRContext *cppContext = unwrap(context);
            return wrap(FabricManagerEffectAttr::get(
                cppContext, StringAttr::get(cppContext, claim), kind));
          },
          nb::arg("context"), nb::arg("claim"), nb::arg("kind"))
      .def_prop_ro("claim",
                   [](FabricManagerEffectAttr attribute) {
                     return attribute.getClaim().getValue().str();
                   })
      .def_prop_ro("kind", &FabricManagerEffectAttr::getKind);

  nb::enum_<FabricManagerIntervalKind>(m, "FabricManagerIntervalKind")
      .value("GeneratedReceiver", FabricManagerIntervalKind::GeneratedReceiver)
      .value("GeneratedSender", FabricManagerIntervalKind::GeneratedSender)
      .value("GeneratedMixed", FabricManagerIntervalKind::GeneratedMixed)
      .value("External", FabricManagerIntervalKind::External);

  tt_attribute_class<FabricManagerIntervalAttr>(m, "FabricManagerIntervalAttr")
      .def_prop_ro("identity",
                   [](FabricManagerIntervalAttr attribute) {
                     return attribute.getIdentity().getValue().str();
                   })
      .def_prop_ro("kind", &FabricManagerIntervalAttr::getKind)
      .def_prop_ro("claim",
                   [](FabricManagerIntervalAttr attribute)
                       -> std::optional<std::string> {
                     if (StringAttr claim = attribute.getClaim()) {
                       return claim.getValue().str();
                     }
                     return std::nullopt;
                   })
      .def_prop_ro("route_indices",
                   [](FabricManagerIntervalAttr attribute) {
                     ArrayRef<int64_t> routeIndices =
                         attribute.getRouteIndices().asArrayRef();
                     return std::vector<int64_t>(routeIndices.begin(),
                                                 routeIndices.end());
                   })
      .def_prop_ro(
          "interfering_intervals",
          [](FabricManagerIntervalAttr attribute) {
            std::vector<std::string> identities;
            identities.reserve(attribute.getInterferingIntervals().size());
            for (StringAttr identity : attribute.getInterferingIntervals()) {
              identities.push_back(identity.getValue().str());
            }
            return identities;
          })
      .def_prop_ro("launch_nodes",
                   [](FabricManagerIntervalAttr attribute)
                       -> std::optional<std::vector<std::array<int64_t, 2>>> {
                     DenseI64ArrayAttr launchNodes = attribute.getLaunchNodes();
                     if (!launchNodes) {
                       return std::nullopt;
                     }
                     ArrayRef<int64_t> coordinates = launchNodes.asArrayRef();
                     assert(coordinates.size() % 2 == 0 &&
                            "launch-node coordinates must contain x/y pairs");
                     std::vector<std::array<int64_t, 2>> nodes;
                     nodes.reserve(coordinates.size() / 2);
                     for (std::size_t index = 0; index < coordinates.size();
                          index += 2) {
                       nodes.push_back(
                           {coordinates[index], coordinates[index + 1]});
                     }
                     return nodes;
                   });

  //===--------------------------------------------------------------------===//
  // Device-domain attributes
  //===--------------------------------------------------------------------===//

  tt_attribute_class<DeviceDomainComponentAttr>(m, "DeviceDomainComponentAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::string name, std::vector<int64_t> extent) {
            MLIRContext *context = unwrap(ctx);
            return wrap(DeviceDomainComponentAttr::get(
                context, StringAttr::get(context, name),
                DenseI64ArrayAttr::get(context, extent)));
          },
          nb::arg("context"), nb::arg("name"), nb::arg("extent"));

  tt_attribute_class<DeviceDomainAttr>(m, "DeviceDomainAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<MlirAttribute> components) {
            SmallVector<DeviceDomainComponentAttr> componentAttrs;
            componentAttrs.reserve(components.size());
            for (MlirAttribute component : components) {
              componentAttrs.push_back(
                  mlir::cast<DeviceDomainComponentAttr>(unwrap(component)));
            }
            return wrap(DeviceDomainAttr::get(unwrap(ctx), componentAttrs));
          },
          nb::arg("context"), nb::arg("components"));

  tt_attribute_class<DeviceRefAttr>(m, "DeviceRefAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<std::vector<int64_t>> coordinates) {
            MLIRContext *context = unwrap(ctx);
            SmallVector<DenseI64ArrayAttr> coordinateAttrs;
            coordinateAttrs.reserve(coordinates.size());
            for (const std::vector<int64_t> &coordinate : coordinates) {
              coordinateAttrs.push_back(
                  DenseI64ArrayAttr::get(context, coordinate));
            }
            return wrap(DeviceRefAttr::get(context, coordinateAttrs));
          },
          nb::arg("context"), nb::arg("coordinates"))
      .def_prop_ro("coordinates", [](DeviceRefAttr &self) {
        std::vector<std::vector<int64_t>> coordinates;
        coordinates.reserve(self.getCoordinates().size());
        for (DenseI64ArrayAttr coordinate : self.getCoordinates()) {
          coordinates.emplace_back(coordinate.asArrayRef().begin(),
                                   coordinate.asArrayRef().end());
        }
        return coordinates;
      });

  tt_attribute_class<DeviceRangeAttr>(m, "DeviceRangeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute lo, MlirAttribute hi) {
            return wrap(DeviceRangeAttr::get(
                unwrap(ctx), mlir::cast<DeviceRefAttr>(unwrap(lo)),
                mlir::cast<DeviceRefAttr>(unwrap(hi))));
          },
          nb::arg("context"), nb::arg("lo"), nb::arg("hi"));

  tt_attribute_class<TransferEdgeAttr>(m, "TransferEdgeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute source,
             std::optional<MlirAttribute> destination,
             std::optional<MlirAttribute> destinationRange) {
            DeviceRefAttr destinationAttr;
            DeviceRangeAttr destinationRangeAttr;
            if (destination) {
              destinationAttr = mlir::cast<DeviceRefAttr>(unwrap(*destination));
            }
            if (destinationRange) {
              destinationRangeAttr =
                  mlir::cast<DeviceRangeAttr>(unwrap(*destinationRange));
            }
            return wrap(TransferEdgeAttr::get(
                unwrap(ctx), mlir::cast<DeviceRefAttr>(unwrap(source)),
                destinationAttr, destinationRangeAttr));
          },
          nb::arg("context"), nb::arg("source"),
          nb::arg("destination") = nb::none(),
          nb::arg("destination_range") = nb::none());

  tt_attribute_class<DeviceTransferAttr>(m, "DeviceTransferAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute domain, MlirAttribute edge) {
            return wrap(DeviceTransferAttr::get(
                unwrap(ctx), mlir::cast<DeviceDomainAttr>(unwrap(domain)),
                mlir::cast<TransferEdgeAttr>(unwrap(edge))));
          },
          nb::arg("context"), nb::arg("domain"), nb::arg("edge"));

  nb::enum_<DFBProtocolEffectKind>(m, "DFBProtocolEffectKind")
      .value("Reserve", DFBProtocolEffectKind::Reserve)
      .value("Push", DFBProtocolEffectKind::Push)
      .value("Wait", DFBProtocolEffectKind::Wait)
      .value("Pop", DFBProtocolEffectKind::Pop);

  tt_attribute_class<DFBProtocolEffectAttr>(m, "DFBProtocolEffectAttr")
      .def_static(
          "get",
          [](MlirContext context, DFBProtocolEffectKind kind,
             int64_t dependencyIndex, int64_t numTiles) {
            MLIRContext *cppContext = unwrap(context);
            DFBProtocolEffectAttr attribute = DFBProtocolEffectAttr::getChecked(
                [cppContext]() {
                  return emitError(UnknownLoc::get(cppContext));
                },
                cppContext, kind, dependencyIndex, numTiles);
            if (!attribute) {
              throw nb::value_error("invalid DFB protocol effect");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("kind"), nb::arg("dependency_index"),
          nb::arg("num_tiles"))
      .def_prop_ro("kind", &DFBProtocolEffectAttr::getKind)
      .def_prop_ro("dependency_index",
                   &DFBProtocolEffectAttr::getDependencyIndex)
      .def_prop_ro("num_tiles", &DFBProtocolEffectAttr::getNumTiles);

  nb::enum_<DFBNonTransactionalAccessKind>(m, "DFBNonTransactionalAccessKind")
      .value("Inspect", DFBNonTransactionalAccessKind::Inspect);

  tt_attribute_class<DFBNonTransactionalAccessAttr>(
      m, "DFBNonTransactionalAccessAttr")
      .def_static(
          "get",
          [](MlirContext context, DFBNonTransactionalAccessKind kind,
             int64_t dependencyIndex) {
            MLIRContext *cppContext = unwrap(context);
            DFBNonTransactionalAccessAttr attribute =
                DFBNonTransactionalAccessAttr::getChecked(
                    [cppContext]() {
                      return emitError(UnknownLoc::get(cppContext));
                    },
                    cppContext, kind, dependencyIndex);
            if (!attribute) {
              throw nb::value_error("invalid DFB non-transactional access");
            }
            return wrap(attribute);
          },
          nb::arg("context"), nb::arg("kind"), nb::arg("dependency_index"))
      .def_prop_ro("kind", &DFBNonTransactionalAccessAttr::getKind)
      .def_prop_ro("dependency_index",
                   &DFBNonTransactionalAccessAttr::getDependencyIndex);

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
  // TensorBackingAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<TensorBackingAttr>(m, "TensorBackingAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t tensorIndex, int64_t byteOffset,
             int64_t byteSize) {
            return wrap(TensorBackingAttr::get(unwrap(ctx), tensorIndex,
                                               byteOffset, byteSize));
          },
          nb::arg("context"), nb::arg("tensor_index"), nb::arg("byte_offset"),
          nb::arg("byte_size"))
      .def_prop_ro("tensor_index", &TensorBackingAttr::getTensorIndex)
      .def_prop_ro("byte_offset", &TensorBackingAttr::getByteOffset)
      .def_prop_ro("byte_size", &TensorBackingAttr::getByteSize);

  //===--------------------------------------------------------------------===//
  // PipeRecordAttr
  //===--------------------------------------------------------------------===//

  tt_attribute_class<PipeRecordAttr>(m, "PipeRecordAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t srcX, int64_t srcY, int64_t dstStartX,
             int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
             bool isCollective, std::optional<MlirAttribute> deviceTransfer) {
            DeviceTransferAttr deviceTransferAttr;
            if (deviceTransfer) {
              deviceTransferAttr =
                  mlir::cast<DeviceTransferAttr>(unwrap(*deviceTransfer));
            }
            return wrap(PipeRecordAttr::get(unwrap(ctx), srcX, srcY, dstStartX,
                                            dstStartY, dstEndX, dstEndY,
                                            isCollective, deviceTransferAttr));
          },
          nb::arg("context"), nb::arg("src_x"), nb::arg("src_y"),
          nb::arg("dst_start_x"), nb::arg("dst_start_y"), nb::arg("dst_end_x"),
          nb::arg("dst_end_y"), nb::arg("is_collective") = false,
          nb::arg("device_transfer").none() = nb::none())
      .def_prop_ro("src_x", &PipeRecordAttr::getSrcX)
      .def_prop_ro("src_y", &PipeRecordAttr::getSrcY)
      .def_prop_ro("dst_start_x", &PipeRecordAttr::getDstStartX)
      .def_prop_ro("dst_start_y", &PipeRecordAttr::getDstStartY)
      .def_prop_ro("dst_end_x", &PipeRecordAttr::getDstEndX)
      .def_prop_ro("dst_end_y", &PipeRecordAttr::getDstEndY)
      .def_prop_ro("is_collective", &PipeRecordAttr::getIsCollective)
      .def_prop_ro("device_transfer", [](PipeRecordAttr &self) -> nb::object {
        if (DeviceTransferAttr transfer = self.getDeviceTransfer()) {
          return nb::cast(wrap(transfer));
        }
        return nb::none();
      });

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
      .def("has_single_receiver", &PipeType::hasSingleReceiver)
      .def("has_multiple_receivers", &PipeType::hasMultipleReceivers)
      .def(
          "is_unicast",
          [](PipeType type) {
            if (PyErr_WarnEx(PyExc_DeprecationWarning,
                             "PipeType.is_unicast() is deprecated; use "
                             "has_single_receiver().",
                             1) < 0) {
              throw nb::python_error();
            }
            return type.hasSingleReceiver();
          },
          "Deprecated. Use has_single_receiver().")
      .def(
          "is_multicast",
          [](PipeType type) {
            if (PyErr_WarnEx(PyExc_DeprecationWarning,
                             "PipeType.is_multicast() is deprecated; use "
                             "has_multiple_receivers().",
                             1) < 0) {
              throw nb::python_error();
            }
            return type.hasMultipleReceivers();
          },
          "Deprecated. Use has_multiple_receivers().");

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

  tt_type_class<ReceiveRequestType>(m, "ReceiveRequestType")
      .def_static(
          "get",
          [](MlirContext ctx) {
            return wrap(ReceiveRequestType::get(unwrap(ctx)));
          },
          nb::arg("context"));

  tt_type_class<ReadyReceiveType>(m, "ReadyReceiveType")
      .def_static(
          "get",
          [](MlirContext ctx) {
            return wrap(ReadyReceiveType::get(unwrap(ctx)));
          },
          nb::arg("context"));
}
