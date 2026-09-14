// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
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

struct StorageSource {
  std::optional<int64_t> tensorIndex;
  int64_t byteOffset = 0;

  bool operator==(const StorageSource &rhs) const {
    return tensorIndex == rhs.tensorIndex && byteOffset == rhs.byteOffset;
  }
};

struct StaticReconfiguration {
  SmallVector<Attribute> templateArguments;
  SmallVector<int32_t> dfbIndices;
};

using CoreCoordinate = std::pair<int64_t, int64_t>;
using OptionalStaticReconfiguration = std::optional<StaticReconfiguration>;

struct ReconfigurationUpdate {
  ttk::OpaqueCallOp call;
  OptionalStaticReconfiguration staticReconfiguration;
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

static FailureOr<std::optional<StorageSource>>
getStorageSource(DictionaryAttr configuration, CoreCoordinate coreCoordinate) {
  auto storageSegments = configuration.getAs<ArrayAttr>("storage_segments");
  if (!storageSegments) {
    return std::optional<StorageSource>(StorageSource{});
  }

  std::optional<StorageSource> source;
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
    if (source) {
      return failure();
    }
    if (auto tensorBacking =
            segment.getAs<TensorBackingAttr>("tensor_backing")) {
      source = StorageSource{tensorBacking.getTensorIndex(),
                             tensorBacking.getByteOffset()};
    } else {
      source = StorageSource{};
    }
  }
  return source;
}

static FailureOr<OptionalStaticReconfiguration>
buildStaticReconfiguration(ArrayAttr dfbEntries, int64_t ordinal,
                           func::FuncOp function, Builder &builder) {
  FailureOr<std::optional<CoreCoordinate>> coreCoordinate =
      getSpecializedCoreCoordinate(function);
  if (failed(coreCoordinate)) {
    return failure();
  }
  if (!*coreCoordinate) {
    return OptionalStaticReconfiguration{};
  }

  StaticReconfiguration result;
  result.templateArguments.push_back(builder.getUI32IntegerAttr(0));
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

    std::optional<StorageSource> stableSource;
    DictionaryAttr selectedConfiguration;
    for (Attribute configurationAttribute : configurations) {
      auto configuration = dyn_cast<DictionaryAttr>(configurationAttribute);
      if (!configuration) {
        return failure();
      }
      FailureOr<std::optional<StorageSource>> source =
          getStorageSource(configuration, **coreCoordinate);
      if (failed(source)) {
        return failure();
      }
      if (*source) {
        if (stableSource && !(*stableSource == **source)) {
          return OptionalStaticReconfiguration{};
        }
        stableSource = **source;
      }
      auto entry = configuration.getAs<IntegerAttr>("entry_reconfiguration");
      if (entry && entry.getInt() == ordinal && *source) {
        if (selectedConfiguration) {
          return failure();
        }
        selectedConfiguration = configuration;
      }
    }
    if (!selectedConfiguration) {
      continue;
    }

    auto numTiles = selectedConfiguration.getAs<IntegerAttr>("num_tiles");
    auto pageSize = selectedConfiguration.getAs<IntegerAttr>("page_size");
    auto blockCount = selectedConfiguration.getAs<IntegerAttr>("block_count");
    if (!numTiles || !pageSize || !blockCount || numTiles.getInt() <= 0 ||
        pageSize.getInt() <= 0 || blockCount.getInt() <= 0) {
      return failure();
    }
    std::optional<uint64_t> numPages =
        llvm::checkedMulUnsigned(static_cast<uint64_t>(numTiles.getInt()),
                                 static_cast<uint64_t>(blockCount.getInt()));
    std::optional<uint64_t> totalBytes =
        numPages ? llvm::checkedMulUnsigned(
                       *numPages, static_cast<uint64_t>(pageSize.getInt()))
                 : std::nullopt;
    if (!numPages || !totalBytes ||
        *numPages > std::numeric_limits<uint32_t>::max() ||
        *totalBytes > std::numeric_limits<uint32_t>::max() ||
        pageSize.getInt() > std::numeric_limits<uint32_t>::max()) {
      return failure();
    }

    result.dfbIndices.push_back(static_cast<int32_t>(dfbIndex.getInt()));
    result.templateArguments.push_back(
        builder.getUI32IntegerAttr(static_cast<uint32_t>(dfbIndex.getInt())));
    result.templateArguments.push_back(
        builder.getUI32IntegerAttr(static_cast<uint32_t>(*totalBytes)));
    result.templateArguments.push_back(
        builder.getUI32IntegerAttr(static_cast<uint32_t>(*numPages)));
    result.templateArguments.push_back(
        builder.getUI32IntegerAttr(static_cast<uint32_t>(pageSize.getInt())));
  }
  result.templateArguments.front() = builder.getUI32IntegerAttr(
      static_cast<uint32_t>(result.dfbIndices.size()));
  return OptionalStaticReconfiguration(std::move(result));
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

    WalkResult result = module.walk([&](ttk::OpaqueCallOp call) -> WalkResult {
      auto ordinal =
          call->getAttrOfType<IntegerAttr>(kDFBReconfigurationOrdinalAttrName);
      if (!ordinal) {
        return WalkResult::advance();
      }
      if (!boundaryOrdinals || !dfbEntries) {
        call.emitOpError("requires finalized DFB reconfiguration metadata");
        return WalkResult::interrupt();
      }
      if (!llvm::is_contained(boundaryOrdinals.asArrayRef(),
                              ordinal.getInt())) {
        call.emitOpError("references an unknown DFB reconfiguration ordinal");
        return WalkResult::interrupt();
      }
      func::FuncOp function = call->getParentOfType<func::FuncOp>();
      if (!function) {
        call.emitOpError("must be nested in a kernel function");
        return WalkResult::interrupt();
      }
      FailureOr<OptionalStaticReconfiguration> staticReconfiguration =
          buildStaticReconfiguration(dfbEntries, ordinal.getInt(), function,
                                     builder);
      if (failed(staticReconfiguration)) {
        call.emitOpError("contains malformed DFB reconfiguration metadata");
        return WalkResult::interrupt();
      }
      updates.push_back(
          ReconfigurationUpdate{call, std::move(*staticReconfiguration)});
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (ReconfigurationUpdate &update : updates) {
      ttk::OpaqueCallOp call = update.call;
      if (update.staticReconfiguration) {
        call->setAttr("template_args",
                      builder.getArrayAttr(
                          update.staticReconfiguration->templateArguments));
        if (update.staticReconfiguration->dfbIndices.empty()) {
          call->removeAttr("dfb_resource_indices");
        } else {
          call->setAttr("dfb_resource_indices",
                        builder.getDenseI32ArrayAttr(
                            update.staticReconfiguration->dfbIndices));
        }
      }
      call->removeAttr(kDFBReconfigurationOrdinalAttrName);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
