// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELANALYZEDFBRESOURCES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLaunchGridAttrName = "ttl.launch_grid";
constexpr llvm::StringLiteral kCoreCoordAttrName = "ttl.core_coord";
constexpr llvm::StringLiteral kLogicalConfigsAttrName =
    "ttl.logical_dfb_configs";
constexpr llvm::StringLiteral kLogicalIndexAttrName =
    "ttl.dfb_logical_index";
constexpr llvm::StringLiteral kPerCoreConfigsAttrName =
    "ttl.per_core_dfb_configs";
constexpr llvm::StringLiteral kResetEpochAttrName = "ttl.dfb_reset_epoch";
constexpr llvm::StringLiteral kResetPreservedIndicesAttrName =
    "ttl.dfb_reset_preserved_indices";
constexpr llvm::StringLiteral kResetCallee =
    "ttlang::reset_dataflow_buffers";
constexpr size_t kResetConfigWords = 11;

enum class AddressScope { Local, RemoteUniform, Legacy };

struct CoreCoord {
  int64_t x;
  int64_t y;
};

struct LogicalConfig {
  int64_t logicalIndex;
  int64_t physicalIndex;
  int64_t epoch;
  int64_t numPages;
  uint64_t pageBytes;
  AddressScope scope;
};

struct PhysicalInfo {
  int64_t initialEpoch = std::numeric_limits<int64_t>::max();
  int64_t initialLogicalIndex = std::numeric_limits<int64_t>::max();
  uint64_t initialPageBytes = 0;
  AddressScope scope = AddressScope::Local;
};

struct PhysicalConfig {
  int64_t physicalIndex;
  int64_t numPages;
  AddressScope scope;

  bool operator==(const PhysicalConfig &other) const {
    return physicalIndex == other.physicalIndex &&
           numPages == other.numPages && scope == other.scope;
  }
};

struct CoreGroup {
  SmallVector<PhysicalConfig> configs;
  SmallVector<CoreCoord> coords;
};

struct EpochPhysicalConfig {
  uint64_t bytes;
  uint64_t pageBytes;

  bool operator==(const EpochPhysicalConfig &other) const {
    return bytes == other.bytes && pageBytes == other.pageBytes;
  }

  bool operator!=(const EpochPhysicalConfig &other) const {
    return !(*this == other);
  }
};

static AddressScope joinScope(AddressScope lhs, AddressScope rhs) {
  if (lhs == AddressScope::RemoteUniform ||
      rhs == AddressScope::RemoteUniform) {
    return AddressScope::RemoteUniform;
  }
  if (lhs == AddressScope::Legacy || rhs == AddressScope::Legacy) {
    return AddressScope::Legacy;
  }
  return AddressScope::Local;
}

static StringRef stringifyScope(AddressScope scope) {
  switch (scope) {
  case AddressScope::Local:
    return "local";
  case AddressScope::RemoteUniform:
    return "remote_uniform";
  case AddressScope::Legacy:
    return "legacy";
  }
  llvm_unreachable("unknown DFB address scope");
}

static FailureOr<AddressScope> parseScope(DictionaryAttr config) {
  auto attr = config.getAs<StringAttr>("address_scope");
  if (!attr) {
    return AddressScope::Legacy;
  }
  if (attr.getValue() == "local") {
    return AddressScope::Local;
  }
  if (attr.getValue() == "remote_uniform") {
    return AddressScope::RemoteUniform;
  }
  return failure();
}

static FailureOr<std::pair<int64_t, int64_t>> readGrid(ModuleOp module) {
  auto attr = module->getAttrOfType<ArrayAttr>(kLaunchGridAttrName);
  if (!attr || attr.size() != 2) {
    return failure();
  }
  auto x = dyn_cast<IntegerAttr>(attr[0]);
  auto y = dyn_cast<IntegerAttr>(attr[1]);
  if (!x || !y || x.getInt() <= 0 || y.getInt() <= 0) {
    return failure();
  }
  return std::pair<int64_t, int64_t>{x.getInt(), y.getInt()};
}

static FailureOr<uint64_t> getPageBytes(Type elementType) {
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    return tileType.getSizeBytes();
  }
  if (!elementType.isIntOrFloat()) {
    return failure();
  }
  return ttcore::TileType::get(elementType).getSizeBytes();
}

static FailureOr<int64_t> getInteger(DictionaryAttr config, StringRef name) {
  auto attr = config.getAs<IntegerAttr>(name);
  if (!attr) {
    return failure();
  }
  return attr.getInt();
}

static FailureOr<SmallVector<size_t>>
getCoveredCores(func::FuncOp func, int64_t gridX, int64_t gridY) {
  SmallVector<size_t> result;
  auto coords = func->getAttrOfType<ArrayAttr>(kCoreCoordAttrName);
  if (!coords) {
    result.reserve(static_cast<size_t>(gridX * gridY));
    for (int64_t y = 0; y < gridY; ++y) {
      for (int64_t x = 0; x < gridX; ++x) {
        result.push_back(static_cast<size_t>(y * gridX + x));
      }
    }
    return result;
  }
  if (coords.empty()) {
    return failure();
  }

  llvm::SmallDenseSet<size_t, 4> seen;
  for (Attribute coordAttr : coords) {
    auto coord = dyn_cast<ArrayAttr>(coordAttr);
    if (!coord || coord.size() != 2) {
      return failure();
    }
    auto x = dyn_cast<IntegerAttr>(coord[0]);
    auto y = dyn_cast<IntegerAttr>(coord[1]);
    if (!x || !y || x.getInt() < 0 || x.getInt() >= gridX ||
        y.getInt() < 0 || y.getInt() >= gridY) {
      return failure();
    }
    size_t index = static_cast<size_t>(y.getInt() * gridX + x.getInt());
    if (seen.insert(index).second) {
      result.push_back(index);
    }
  }
  return result;
}

static FailureOr<llvm::SmallDenseSet<int64_t, 8>>
getPreservedPhysicalIndices(ttk::OpaqueCallOp call) {
  llvm::SmallDenseSet<int64_t, 8> result;
  auto attr = call->getAttrOfType<ArrayAttr>(kResetPreservedIndicesAttrName);
  if (!attr) {
    return result;
  }
  for (Attribute value : attr) {
    auto index = dyn_cast<IntegerAttr>(value);
    if (!index || index.getInt() < 0 || !result.insert(index.getInt()).second) {
      return failure();
    }
  }
  return result;
}

static bool isMetadataOnlyResetUse(Operation *user) {
  auto call = dyn_cast<ttk::OpaqueCallOp>(user);
  return call && call.getCallee() == kResetCallee;
}

struct TTKernelAnalyzeDFBResourcesPass
    : impl::TTKernelAnalyzeDFBResourcesBase<
          TTKernelAnalyzeDFBResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto configsAttr =
        module->getAttrOfType<ArrayAttr>(kLogicalConfigsAttrName);
    if (!configsAttr) {
      return;
    }

    FailureOr<std::pair<int64_t, int64_t>> grid = readGrid(module);
    if (failed(grid)) {
      module.emitOpError() << "requires `" << kLaunchGridAttrName
                           << "` to be a length-2 array of positive extents";
      signalPassFailure();
      return;
    }
    auto [gridX, gridY] = *grid;

    llvm::MapVector<int64_t, LogicalConfig> logicalConfigs;
    llvm::MapVector<int64_t, PhysicalInfo> physicalInfos;
    for (Attribute attr : configsAttr) {
      auto config = dyn_cast<DictionaryAttr>(attr);
      if (!config) {
        module.emitOpError()
            << "`" << kLogicalConfigsAttrName
            << "` entries must be dictionaries";
        signalPassFailure();
        return;
      }
      FailureOr<int64_t> logicalIndex = getInteger(config, "logical_index");
      FailureOr<int64_t> physicalIndex = getInteger(config, "physical_index");
      FailureOr<int64_t> epoch = getInteger(config, "epoch");
      FailureOr<int64_t> numPages = getInteger(config, "num_pages");
      auto elementType = config.getAs<TypeAttr>("element_type");
      FailureOr<AddressScope> scope = parseScope(config);
      if (failed(logicalIndex) || failed(physicalIndex) || failed(epoch) ||
          failed(numPages) || !elementType || failed(scope) ||
          *logicalIndex < 0 || *physicalIndex < 0 || *epoch < 0 ||
          *numPages <= 0) {
        module.emitOpError() << "has malformed `" << kLogicalConfigsAttrName
                             << "` entry " << config;
        signalPassFailure();
        return;
      }
      FailureOr<uint64_t> pageBytes = getPageBytes(elementType.getValue());
      if (failed(pageBytes) || *pageBytes == 0 ||
          static_cast<uint64_t>(*numPages) >
              std::numeric_limits<uint64_t>::max() / *pageBytes) {
        module.emitOpError() << "cannot size logical DFB " << *logicalIndex;
        signalPassFailure();
        return;
      }
      if (logicalConfigs.count(*logicalIndex)) {
        module.emitOpError() << "has duplicate logical DFB index "
                             << *logicalIndex;
        signalPassFailure();
        return;
      }

      LogicalConfig logical{*logicalIndex, *physicalIndex, *epoch, *numPages,
                            *pageBytes, *scope};
      logicalConfigs.insert({logical.logicalIndex, logical});

      PhysicalInfo &physical = physicalInfos[logical.physicalIndex];
      physical.scope = joinScope(physical.scope, logical.scope);
      if (logical.epoch == physical.initialEpoch &&
          logical.pageBytes != physical.initialPageBytes) {
        module.emitOpError()
            << "physical DFB " << logical.physicalIndex
            << " has incompatible page sizes in initial epoch";
        signalPassFailure();
        return;
      }
      if (std::tie(logical.epoch, logical.logicalIndex) <
          std::tie(physical.initialEpoch, physical.initialLogicalIndex)) {
        physical.initialEpoch = logical.epoch;
        physical.initialLogicalIndex = logical.logicalIndex;
        physical.initialPageBytes = logical.pageBytes;
      }
    }

    const size_t coreCount = static_cast<size_t>(gridX * gridY);
    SmallVector<llvm::SmallDenseSet<int64_t, 8>> logicalsByCore(coreCount);
    bool walkFailed = false;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      llvm::SmallDenseSet<int64_t, 8> usedLogicals;
      WalkResult result = func.walk([&](ttk::GetCompileArgValOp argOp) {
        if (llvm::all_of(argOp.getResult().getUsers(),
                         isMetadataOnlyResetUse)) {
          return WalkResult::advance();
        }
        auto logicalIndex =
            argOp->getAttrOfType<IntegerAttr>(kLogicalIndexAttrName);
        if (!logicalIndex) {
          if (isa<ttk::CBType>(argOp.getResult().getType())) {
            argOp.emitOpError()
                << "is CB-typed but missing `" << kLogicalIndexAttrName << "`";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }
        auto config = logicalConfigs.find(logicalIndex.getInt());
        if (config == logicalConfigs.end()) {
          argOp.emitOpError() << "references logical DFB "
                              << logicalIndex.getInt() << " absent from `"
                              << kLogicalConfigsAttrName << "`";
          return WalkResult::interrupt();
        }
        if (static_cast<int64_t>(argOp.getArgIndex()) !=
            config->second.physicalIndex) {
          argOp.emitOpError()
              << "reads physical DFB " << argOp.getArgIndex()
              << " but logical DFB " << logicalIndex.getInt() << " maps to "
              << config->second.physicalIndex;
          return WalkResult::interrupt();
        }
        usedLogicals.insert(logicalIndex.getInt());
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        walkFailed = true;
        break;
      }
      FailureOr<SmallVector<size_t>> covered =
          getCoveredCores(func, gridX, gridY);
      if (failed(covered)) {
        func.emitOpError() << "has malformed `" << kCoreCoordAttrName << "`";
        walkFailed = true;
        break;
      }
      for (size_t core : *covered) {
        for (int64_t logicalIndex : usedLogicals) {
          logicalsByCore[core].insert(logicalIndex);
        }
      }
    }
    if (walkFailed) {
      signalPassFailure();
      return;
    }

    SmallVector<std::map<int64_t, uint64_t>> bytesByCore(coreCount);
    for (size_t core = 0; core < coreCount; ++core) {
      for (int64_t logicalIndex : logicalsByCore[core]) {
        const LogicalConfig &logical = logicalConfigs[logicalIndex];
        uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                         logical.pageBytes;
        uint64_t &current = bytesByCore[core][logical.physicalIndex];
        current = std::max(current, bytes);
      }
    }

    for (const auto &[logicalIndex, logical] : logicalConfigs) {
      (void)logicalIndex;
      if (logical.scope != AddressScope::Legacy) {
        continue;
      }
      uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                       logical.pageBytes;
      for (auto &core : bytesByCore) {
        uint64_t &current = core[logical.physicalIndex];
        current = std::max(current, bytes);
      }
    }

    SmallVector<SmallVector<PhysicalConfig>> configsByCore(coreCount);
    for (size_t core = 0; core < coreCount; ++core) {
      for (auto [physicalIndex, bytes] : bytesByCore[core]) {
        const PhysicalInfo &physical = physicalInfos[physicalIndex];
        uint64_t pages =
            (bytes + physical.initialPageBytes - 1) /
            physical.initialPageBytes;
        if (pages > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          module.emitOpError() << "physical DFB " << physicalIndex
                               << " capacity exceeds i32 page count";
          signalPassFailure();
          return;
        }
        configsByCore[core].push_back(PhysicalConfig{
            physicalIndex, static_cast<int64_t>(pages), physical.scope});
      }
    }

    SmallVector<ttk::OpaqueCallOp> resetCalls;
    llvm::SmallDenseSet<int64_t, 8> pinnedPhysicalIndices;
    module.walk([&](ttk::OpaqueCallOp call) {
      if (call.getCallee() != kResetCallee) {
        return;
      }
      resetCalls.push_back(call);
      FailureOr<llvm::SmallDenseSet<int64_t, 8>> preserved =
          getPreservedPhysicalIndices(call);
      if (failed(preserved)) {
        call.emitOpError() << "has malformed `"
                           << kResetPreservedIndicesAttrName << "`";
        walkFailed = true;
        return;
      }
      pinnedPhysicalIndices.insert(preserved->begin(), preserved->end());
    });
    if (walkFailed) {
      signalPassFailure();
      return;
    }

    llvm::MapVector<int64_t, LogicalConfig> pinnedConfigs;
    for (int64_t physicalIndex : pinnedPhysicalIndices) {
      const LogicalConfig *pinned = nullptr;
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        (void)logicalIndex;
        if (logical.physicalIndex != physicalIndex) {
          continue;
        }
        if (pinned) {
          module.emitOpError()
              << "preserved physical DFB " << physicalIndex
              << " is shared by more than one logical DFB";
          signalPassFailure();
          return;
        }
        pinned = &logical;
      }
      if (!pinned) {
        module.emitOpError() << "preserves unknown physical DFB "
                             << physicalIndex;
        signalPassFailure();
        return;
      }
      pinnedConfigs.insert({physicalIndex, *pinned});
    }

    auto getEpochConfigs =
        [&](size_t core, int64_t epoch)
        -> FailureOr<std::map<int64_t, EpochPhysicalConfig>> {
      std::map<int64_t, EpochPhysicalConfig> result;
      auto addConfig = [&](const LogicalConfig &logical) -> LogicalResult {
        if (logical.scope != AddressScope::Legacy &&
            !logicalsByCore[core].contains(logical.logicalIndex)) {
          return success();
        }
        uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                         logical.pageBytes;
        auto [it, inserted] = result.try_emplace(
            logical.physicalIndex,
            EpochPhysicalConfig{bytes, logical.pageBytes});
        if (!inserted) {
          if (it->second.pageBytes != logical.pageBytes) {
            module.emitOpError()
                << "physical DFB " << logical.physicalIndex << " in epoch "
                << epoch << " has incompatible page sizes";
            return failure();
          }
          it->second.bytes = std::max(it->second.bytes, bytes);
        }
        return success();
      };
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        (void)logicalIndex;
        if (logical.epoch != epoch) {
          continue;
        }
        if (failed(addConfig(logical))) {
          return failure();
        }
      }
      for (const auto &[physicalIndex, logical] : pinnedConfigs) {
        (void)physicalIndex;
        if (logical.epoch != epoch && failed(addConfig(logical))) {
          return failure();
        }
      }
      return result;
    };

    for (ttk::OpaqueCallOp call : resetCalls) {
      auto epochAttr = call->getAttrOfType<IntegerAttr>(kResetEpochAttrName);
      if (!epochAttr || epochAttr.getInt() < 0) {
        call.emitOpError() << "is missing a valid `" << kResetEpochAttrName
                           << "`";
        signalPassFailure();
        return;
      }
      func::FuncOp func = call->getParentOfType<func::FuncOp>();
      FailureOr<SmallVector<size_t>> covered =
          getCoveredCores(func, gridX, gridY);
      if (failed(covered)) {
        func.emitOpError() << "has malformed `" << kCoreCoordAttrName << "`";
        signalPassFailure();
        return;
      }
      FailureOr<llvm::SmallDenseSet<int64_t, 8>> preserved =
          getPreservedPhysicalIndices(call);
      assert(succeeded(preserved) && "preserved indices were validated above");

      std::optional<std::map<int64_t, EpochPhysicalConfig>> desired;
      for (size_t core : *covered) {
        FailureOr<std::map<int64_t, EpochPhysicalConfig>> coreDesired =
            getEpochConfigs(core, epochAttr.getInt());
        if (failed(coreDesired)) {
          signalPassFailure();
          return;
        }
        for (const auto &[physicalIndex, epochConfig] : *coreDesired) {
          const int64_t targetPhysicalIndex = physicalIndex;
          auto allocation = llvm::find_if(
              configsByCore[core], [&](const PhysicalConfig &config) {
                return config.physicalIndex == targetPhysicalIndex;
              });
          if (allocation == configsByCore[core].end()) {
            call.emitOpError() << "retains physical DFB " << physicalIndex
                               << " without backing on core " << core;
            signalPassFailure();
            return;
          }
          const PhysicalInfo &physical = physicalInfos[physicalIndex];
          uint64_t capacity =
              static_cast<uint64_t>(allocation->numPages) *
              physical.initialPageBytes;
          if (epochConfig.bytes > capacity) {
            call.emitOpError()
                << "configures physical DFB " << physicalIndex << " for "
                << epochConfig.bytes << " bytes with only " << capacity
                << " bytes allocated on core " << core;
            signalPassFailure();
            return;
          }
        }
        for (int64_t physicalIndex : *preserved) {
          coreDesired->erase(physicalIndex);
        }
        if (desired && *desired != *coreDesired) {
          call.emitOpError()
              << "covers cores with different reset configurations; enable "
                 "core specialization for this kernel";
          signalPassFailure();
          return;
        }
        desired = std::move(*coreDesired);
      }
      assert(desired && "reset kernel must cover at least one core");

      ArrayAttr oldArgs = call.getTemplateArgsAttr();
      auto oldCount = oldArgs && !oldArgs.empty()
                          ? dyn_cast<IntegerAttr>(oldArgs[0])
                          : IntegerAttr();
      if (!oldCount || oldCount.getInt() < 0 ||
          oldArgs.size() !=
              1 + static_cast<size_t>(oldCount.getInt()) *
                      kResetConfigWords) {
        call.emitOpError() << "has malformed reset configuration arguments";
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      SmallVector<Attribute> newArgs{
          builder.getI64IntegerAttr(static_cast<int64_t>(desired->size()))};
      llvm::SmallDenseSet<int64_t, 8> retained;
      for (int64_t record = 0; record < oldCount.getInt(); ++record) {
        size_t base = 1 + static_cast<size_t>(record) * kResetConfigWords;
        auto physicalIndex = dyn_cast<IntegerAttr>(oldArgs[base]);
        auto pageBytes = dyn_cast<IntegerAttr>(oldArgs[base + 3]);
        if (!physicalIndex || !pageBytes || pageBytes.getInt() <= 0) {
          call.emitOpError() << "has a malformed reset configuration record";
          signalPassFailure();
          return;
        }
        auto config = desired->find(physicalIndex.getInt());
        if (config == desired->end()) {
          continue;
        }
        if (config->second.pageBytes !=
                static_cast<uint64_t>(pageBytes.getInt()) ||
            config->second.bytes % config->second.pageBytes != 0 ||
            config->second.bytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          call.emitOpError()
              << "cannot specialize physical DFB " << physicalIndex.getInt()
              << " reset geometry";
          signalPassFailure();
          return;
        }
        retained.insert(physicalIndex.getInt());
        newArgs.push_back(oldArgs[base]);
        newArgs.push_back(builder.getI64IntegerAttr(
            static_cast<int64_t>(config->second.bytes)));
        newArgs.push_back(builder.getI64IntegerAttr(
            static_cast<int64_t>(config->second.bytes /
                                 config->second.pageBytes)));
        newArgs.append(oldArgs.begin() + base + 3,
                       oldArgs.begin() + base + kResetConfigWords);
      }
      if (retained.size() != desired->size()) {
        call.emitOpError()
            << "reset table does not contain every active physical DFB";
        signalPassFailure();
        return;
      }
      call.setTemplateArgsAttr(builder.getArrayAttr(newArgs));
      call->removeAttr(kResetEpochAttrName);
      call->removeAttr(kResetPreservedIndicesAttrName);
    }

    SmallVector<CoreGroup> groups;
    for (int64_t y = 0; y < gridY; ++y) {
      for (int64_t x = 0; x < gridX; ++x) {
        size_t core = static_cast<size_t>(y * gridX + x);
        auto group = llvm::find_if(groups, [&](const CoreGroup &candidate) {
          return candidate.configs == configsByCore[core];
        });
        if (group == groups.end()) {
          groups.push_back(CoreGroup{configsByCore[core], {}});
          group = std::prev(groups.end());
        }
        group->coords.push_back(CoreCoord{x, y});
      }
    }

    OpBuilder builder(module.getContext());
    SmallVector<Attribute> groupAttrs;
    groupAttrs.reserve(groups.size());
    for (const CoreGroup &group : groups) {
      SmallVector<Attribute> configAttrs;
      configAttrs.reserve(group.configs.size());
      for (const PhysicalConfig &config : group.configs) {
        configAttrs.push_back(DictionaryAttr::get(
            module.getContext(),
            {builder.getNamedAttr("dfb_index", builder.getI32IntegerAttr(
                                                   config.physicalIndex)),
             builder.getNamedAttr("num_pages",
                                  builder.getI32IntegerAttr(config.numPages)),
             builder.getNamedAttr("address_scope", builder.getStringAttr(
                                                       stringifyScope(
                                                           config.scope)))}));
      }
      SmallVector<Attribute> coordAttrs;
      coordAttrs.reserve(group.coords.size());
      for (CoreCoord coord : group.coords) {
        coordAttrs.push_back(builder.getI64ArrayAttr({coord.x, coord.y}));
      }
      groupAttrs.push_back(DictionaryAttr::get(
          module.getContext(),
          {builder.getNamedAttr("core_coords", builder.getArrayAttr(coordAttrs)),
           builder.getNamedAttr("configs", builder.getArrayAttr(configAttrs))}));
    }
    module->setAttr(kPerCoreConfigsAttrName, builder.getArrayAttr(groupAttrs));
  }
};

} // namespace

} // namespace mlir::tt::ttl
