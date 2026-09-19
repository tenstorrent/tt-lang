// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELSPECIALIZEDFBRECONFIGURATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr uint32_t kReconfigurationRecordWordCount = 12;

using CoreCoordinate = std::pair<int64_t, int64_t>;

struct SpecializedReconfiguration {
  SmallVector<Attribute> templateArguments;
  SmallVector<int32_t> dfbIndices;
  bool updatesDescriptors = false;
};

using OptionalSpecializedReconfiguration =
    std::optional<SpecializedReconfiguration>;

struct ReconfigurationUpdate {
  ttk::OpaqueCallOp call;
  OptionalSpecializedReconfiguration specializedReconfiguration;
};

static FailureOr<std::optional<CoreCoordinate>>
getSpecializedCoreCoordinate(func::FuncOp function) {
  Attribute coreCoordinateAttribute = function->getAttr(kCoreCoordAttrName);
  if (!coreCoordinateAttribute) {
    return std::optional<CoreCoordinate>{};
  }
  auto coreCoordinates = dyn_cast<ArrayAttr>(coreCoordinateAttribute);
  if (!coreCoordinates || coreCoordinates.size() != 1) {
    return failure();
  }
  auto coordinate = dyn_cast<ArrayAttr>(coreCoordinates[0]);
  if (!coordinate || coordinate.size() != 2) {
    return failure();
  }
  auto coreX = dyn_cast<IntegerAttr>(coordinate[0]);
  auto coreY = dyn_cast<IntegerAttr>(coordinate[1]);
  if (!coreX || !coreY) {
    return failure();
  }
  return std::optional<CoreCoordinate>(
      CoreCoordinate{coreX.getInt(), coreY.getInt()});
}

static FailureOr<bool>
storageSegmentContainsCore(DictionaryAttr segment,
                           CoreCoordinate coreCoordinate) {
  auto nodes = segment.getAs<ArrayAttr>("nodes");
  if (!nodes) {
    return failure();
  }
  for (Attribute nodeAttribute : nodes) {
    auto node = dyn_cast<ArrayAttr>(nodeAttribute);
    if (!node || node.size() != 2) {
      return failure();
    }
    auto nodeX = dyn_cast<IntegerAttr>(node[0]);
    auto nodeY = dyn_cast<IntegerAttr>(node[1]);
    if (!nodeX || !nodeY) {
      return failure();
    }
    if (nodeX.getInt() == coreCoordinate.first &&
        nodeY.getInt() == coreCoordinate.second) {
      return true;
    }
  }
  return false;
}

static FailureOr<bool>
configurationAppliesToCore(DictionaryAttr configuration,
                           CoreCoordinate coreCoordinate) {
  auto storageSegments = configuration.getAs<ArrayAttr>("storage_segments");
  if (!storageSegments) {
    return true;
  }

  bool appliesToCore = false;
  for (Attribute segmentAttribute : storageSegments) {
    auto segment = dyn_cast<DictionaryAttr>(segmentAttribute);
    if (!segment) {
      return failure();
    }
    FailureOr<bool> containsCore =
        storageSegmentContainsCore(segment, coreCoordinate);
    if (failed(containsCore)) {
      return failure();
    }
    if (!*containsCore) {
      continue;
    }
    if (appliesToCore) {
      return failure();
    }
    appliesToCore = true;
  }
  return appliesToCore;
}

static FailureOr<uint32_t> getUI32(Attribute attribute) {
  auto integer = dyn_cast<IntegerAttr>(attribute);
  auto type = integer ? dyn_cast<IntegerType>(integer.getType()) : IntegerType();
  if (!type || type.getWidth() != 32 || !type.isUnsigned()) {
    return failure();
  }
  return static_cast<uint32_t>(integer.getValue().getZExtValue());
}

static FailureOr<OptionalSpecializedReconfiguration>
buildSpecializedReconfiguration(ttk::OpaqueCallOp call, ArrayAttr dfbEntries,
                                int64_t ordinal, func::FuncOp function,
                                Builder &builder) {
  FailureOr<std::optional<CoreCoordinate>> coreCoordinate =
      getSpecializedCoreCoordinate(function);
  if (failed(coreCoordinate)) {
    return failure();
  }
  if (!*coreCoordinate) {
    return OptionalSpecializedReconfiguration{};
  }

  std::optional<ArrayAttr> templateArguments = call.getTemplateArgs();
  if (!templateArguments || templateArguments->empty()) {
    return failure();
  }
  FailureOr<uint32_t> runtimeRecordCount = getUI32((*templateArguments)[0]);
  if (failed(runtimeRecordCount)) {
    return failure();
  }
  std::optional<uint64_t> recordWordCount = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(*runtimeRecordCount),
      static_cast<uint64_t>(kReconfigurationRecordWordCount));
  if (!recordWordCount || *recordWordCount + 1 != templateArguments->size()) {
    return failure();
  }

  SpecializedReconfiguration result;
  result.templateArguments.push_back(builder.getUI32IntegerAttr(
      static_cast<uint32_t>(*runtimeRecordCount)));
  result.templateArguments.push_back(builder.getUI32IntegerAttr(0));

  uint32_t runtimeRecordOffset = 0;
  int64_t previousDFBIndex = -1;
  for (Attribute dfbEntryAttribute : dfbEntries) {
    auto dfbEntry = dyn_cast<DictionaryAttr>(dfbEntryAttribute);
    auto dfbIndex =
        dfbEntry ? dfbEntry.getAs<IntegerAttr>("dfb_index") : IntegerAttr();
    auto configurations =
        dfbEntry ? dfbEntry.getAs<ArrayAttr>("configurations") : ArrayAttr();
    if (!dfbIndex || !configurations || dfbIndex.getInt() <= previousDFBIndex ||
        dfbIndex.getInt() > std::numeric_limits<int32_t>::max()) {
      return failure();
    }
    previousDFBIndex = dfbIndex.getInt();

    DictionaryAttr selectedConfiguration;
    for (Attribute configurationAttribute : configurations) {
      auto configuration = dyn_cast<DictionaryAttr>(configurationAttribute);
      if (!configuration) {
        return failure();
      }
      auto entry = configuration.getAs<IntegerAttr>("entry_reconfiguration");
      if (!entry || entry.getInt() != ordinal) {
        continue;
      }
      if (selectedConfiguration) {
        return failure();
      }
      selectedConfiguration = configuration;
    }
    if (!selectedConfiguration) {
      continue;
    }

    if (runtimeRecordOffset >= *runtimeRecordCount) {
      return failure();
    }
    size_t sourceRecordBegin =
        1 + runtimeRecordOffset * kReconfigurationRecordWordCount;
    FailureOr<uint32_t> sourceDFBIndex =
        getUI32((*templateArguments)[sourceRecordBegin]);
    if (failed(sourceDFBIndex) || *sourceDFBIndex != dfbIndex.getInt()) {
      return failure();
    }

    FailureOr<bool> appliesToCore =
        configurationAppliesToCore(selectedConfiguration, **coreCoordinate);
    if (failed(appliesToCore)) {
      return failure();
    }
    if (*appliesToCore) {
      result.dfbIndices.push_back(static_cast<int32_t>(dfbIndex.getInt()));
      result.templateArguments.push_back(
          builder.getUI32IntegerAttr(runtimeRecordOffset));
      result.templateArguments.append(
          templateArguments->begin() + sourceRecordBegin,
          templateArguments->begin() + sourceRecordBegin +
              kReconfigurationRecordWordCount);
      FailureOr<uint32_t> updateDescriptor =
          getUI32((*templateArguments)[sourceRecordBegin + 4]);
      if (failed(updateDescriptor) || *updateDescriptor > 1) {
        return failure();
      }
      result.updatesDescriptors |= *updateDescriptor != 0;
    }
    ++runtimeRecordOffset;
  }
  if (runtimeRecordOffset != *runtimeRecordCount) {
    return failure();
  }

  result.templateArguments[1] = builder.getUI32IntegerAttr(
      static_cast<uint32_t>(result.dfbIndices.size()));
  return OptionalSpecializedReconfiguration(std::move(result));
}

struct TTKernelSpecializeDFBReconfigurationPass
    : impl::TTKernelSpecializeDFBReconfigurationBase<
          TTKernelSpecializeDFBReconfigurationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto plan =
        module->getAttrOfType<DictionaryAttr>(kDFBReconfigurationPlanAttrName);
    auto boundaryOrdinals =
        plan ? plan.getAs<DenseI64ArrayAttr>("boundary_ordinals")
             : DenseI64ArrayAttr();
    auto dfbEntries = plan ? plan.getAs<ArrayAttr>("dfbs") : ArrayAttr();
    SmallVector<ReconfigurationUpdate> updates;
    Builder builder(module.getContext());

    WalkResult walkResult =
        module.walk([&](ttk::OpaqueCallOp call) -> WalkResult {
          auto ordinal = call->getAttrOfType<IntegerAttr>(
              kDFBReconfigurationOrdinalAttrName);
          if (!ordinal) {
            return WalkResult::advance();
          }
          if (!boundaryOrdinals || !dfbEntries) {
            call.emitOpError("requires finalized DFB reconfiguration metadata");
            return WalkResult::interrupt();
          }
          if (!llvm::is_contained(boundaryOrdinals.asArrayRef(),
                                  ordinal.getInt())) {
            call.emitOpError(
                "references an unknown DFB reconfiguration ordinal");
            return WalkResult::interrupt();
          }
          StringRef callee = call.getCallee();
          if (callee != "experimental::reconfigure_dfb_interfaces" &&
              callee != "experimental::reconfigure_dfb_descriptors") {
            call.emitOpError("has an invalid DFB reconfiguration callee");
            return WalkResult::interrupt();
          }
          func::FuncOp function = call->getParentOfType<func::FuncOp>();
          if (!function) {
            call.emitOpError("must be nested in a kernel function");
            return WalkResult::interrupt();
          }
          FailureOr<OptionalSpecializedReconfiguration> specialization =
              buildSpecializedReconfiguration(
                  call, dfbEntries, ordinal.getInt(), function, builder);
          if (failed(specialization)) {
            call.emitOpError("contains malformed DFB reconfiguration metadata");
            return WalkResult::interrupt();
          }
          updates.push_back(
              ReconfigurationUpdate{call, std::move(*specialization)});
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (ReconfigurationUpdate &update : updates) {
      ttk::OpaqueCallOp call = update.call;
      if (update.specializedReconfiguration) {
        SpecializedReconfiguration &specialization =
            *update.specializedReconfiguration;
        call->setAttr("callee", builder.getStringAttr(
                                    specialization.updatesDescriptors
                                        ? "experimental::reconfigure_dfb_"
                                          "descriptors_specialized"
                                        : "experimental::reconfigure_dfb_"
                                          "interfaces_specialized"));
        call->setAttr("template_args",
                      builder.getArrayAttr(specialization.templateArguments));
        if (specialization.dfbIndices.empty()) {
          call->removeAttr("dfb_resource_indices");
        } else {
          call->setAttr(
              "dfb_resource_indices",
              builder.getDenseI32ArrayAttr(specialization.dfbIndices));
        }
      }
      call->removeAttr(kDFBReconfigurationOrdinalAttrName);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
