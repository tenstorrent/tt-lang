// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Launch Node Domain Analysis
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "ttlang/Analysis/ExecutionCountAnalysis.h"
#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <utility>

namespace mlir::tt::ttl {

namespace {

/// Result of evaluating a predicate over the launch grid.
struct LaunchNodeDomainResult {
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

/// Domains reached by the true and false successors of a branch.
struct BranchLaunchNodeDomains {
  LaunchNodeDomain thenDomain;
  LaunchNodeDomain elseDomain;
  Operation *unanalyzableOp = nullptr;
};

} // namespace

static LaunchNodeDomainResult
getAffineIfLaunchNodeDomain(affine::AffineIfOp ifOp,
                            const LaunchNodeDomain &baseDomain);

bool LaunchNodeCoord::operator<(const LaunchNodeCoord &rhs) const {
  return std::tie(x, y) < std::tie(rhs.x, rhs.y);
}

bool LaunchNodeCoord::operator==(const LaunchNodeCoord &rhs) const {
  return x == rhs.x && y == rhs.y;
}

LaunchExecutionLocation::LaunchExecutionLocation(LaunchNodeCoord node)
    : node(node) {}

LaunchExecutionLocation::LaunchExecutionLocation(LaunchNodeCoord node,
                                                 DeviceDomainAttr deviceDomain,
                                                 DeviceRefAttr device)
    : node(node), deviceDomain(deviceDomain), device(device) {
  assert(deviceDomain && device &&
         "device-aware launch location requires a domain and device");
}

bool LaunchExecutionLocation::operator<(
    const LaunchExecutionLocation &rhs) const {
  if (node < rhs.node) {
    return true;
  }
  if (rhs.node < node) {
    return false;
  }
  std::less<const void *> less;
  if (deviceDomain != rhs.deviceDomain) {
    return less(deviceDomain ? deviceDomain.getAsOpaquePointer() : nullptr,
                rhs.deviceDomain ? rhs.deviceDomain.getAsOpaquePointer()
                                 : nullptr);
  }
  return less(device ? device.getAsOpaquePointer() : nullptr,
              rhs.device ? rhs.device.getAsOpaquePointer() : nullptr);
}

bool LaunchExecutionLocation::operator==(
    const LaunchExecutionLocation &rhs) const {
  return node == rhs.node && deviceDomain == rhs.deviceDomain &&
         device == rhs.device;
}

FailureOr<LaunchExecutionLocation>
getPipeExecutionLocation(LaunchNodeCoord node, DeviceTransferAttr transfer,
                         PipeRole role) {
  if (!transfer) {
    return LaunchExecutionLocation(node);
  }
  assert((role == PipeRole::Source || role == PipeRole::Destination) &&
         "pipe execution location requires an endpoint role");
  DeviceRefAttr device = role == PipeRole::Source
                             ? transfer.getEdge().getSource()
                             : transfer.getEdge().getDestination();
  if (!device) {
    return failure();
  }
  return LaunchExecutionLocation(node, transfer.getDomain(), device);
}

LaunchNodeDomain LaunchNodeDomain::unknown() {
  return {/*known=*/false, /*hasUpperBound=*/false, {}};
}

LaunchNodeDomain
LaunchNodeDomain::unknownWithin(const LaunchNodeDomain &domain) {
  LaunchNodeDomain result = LaunchNodeDomain::unknown();
  if (const std::set<LaunchNodeCoord> *bound = domain.getUpperBoundNodes()) {
    if (bound->empty()) {
      return LaunchNodeDomain{};
    }
    result.hasUpperBound = true;
    result.nodes = *bound;
  }
  return result;
}

bool LaunchNodeDomain::isSubsetOf(const LaunchNodeDomain &rhs) const {
  if (!known || !rhs.known) {
    return false;
  }
  return std::includes(rhs.nodes.begin(), rhs.nodes.end(), nodes.begin(),
                       nodes.end());
}

bool LaunchNodeDomain::isUpperBoundSubsetOf(const LaunchNodeDomain &rhs) const {
  if (!rhs.known) {
    return false;
  }
  const std::set<LaunchNodeCoord> *bound = getUpperBoundNodes();
  return bound && std::includes(rhs.nodes.begin(), rhs.nodes.end(),
                                bound->begin(), bound->end());
}

const std::set<LaunchNodeCoord> *LaunchNodeDomain::getUpperBoundNodes() const {
  if (!known && !hasUpperBound) {
    return nullptr;
  }
  return &nodes;
}

static LaunchNodeDomain
getUnknownDomainWithBound(std::set<LaunchNodeCoord> boundNodes) {
  LaunchNodeDomain bound;
  bound.nodes = std::move(boundNodes);
  return LaunchNodeDomain::unknownWithin(bound);
}

LaunchNodeDomain
LaunchNodeDomain::unionWith(const LaunchNodeDomain &rhs) const {
  if (known && rhs.isUpperBoundSubsetOf(*this)) {
    return *this;
  }
  if (rhs.known && isUpperBoundSubsetOf(rhs)) {
    return rhs;
  }
  LaunchNodeDomain result =
      known && rhs.known ? LaunchNodeDomain{} : LaunchNodeDomain::unknown();
  const std::set<LaunchNodeCoord> *lhsBound = getUpperBoundNodes();
  const std::set<LaunchNodeCoord> *rhsBound = rhs.getUpperBoundNodes();
  if (!lhsBound || !rhsBound) {
    return result;
  }
  std::set<LaunchNodeCoord> boundNodes;
  std::set_union(lhsBound->begin(), lhsBound->end(), rhsBound->begin(),
                 rhsBound->end(), std::inserter(boundNodes, boundNodes.end()));
  if (result.known) {
    result.nodes = std::move(boundNodes);
    return result;
  }
  return getUnknownDomainWithBound(std::move(boundNodes));
}

LaunchNodeDomain
LaunchNodeDomain::intersectWith(const LaunchNodeDomain &rhs) const {
  if ((known && nodes.empty()) || (rhs.known && rhs.nodes.empty())) {
    return LaunchNodeDomain{};
  }
  LaunchNodeDomain result =
      known && rhs.known ? LaunchNodeDomain{} : LaunchNodeDomain::unknown();
  const std::set<LaunchNodeCoord> *lhsBound = getUpperBoundNodes();
  const std::set<LaunchNodeCoord> *rhsBound = rhs.getUpperBoundNodes();
  if (!lhsBound && !rhsBound) {
    return result;
  }
  std::set<LaunchNodeCoord> boundNodes;
  if (!lhsBound) {
    boundNodes = *rhsBound;
  } else if (!rhsBound) {
    boundNodes = *lhsBound;
  } else {
    std::set_intersection(lhsBound->begin(), lhsBound->end(), rhsBound->begin(),
                          rhsBound->end(),
                          std::inserter(boundNodes, boundNodes.end()));
  }
  if (result.known) {
    result.nodes = std::move(boundNodes);
    return result;
  }
  return getUnknownDomainWithBound(std::move(boundNodes));
}

LaunchNodeDomain LaunchNodeDomain::subtract(const LaunchNodeDomain &rhs) const {
  LaunchNodeDomain result =
      known && rhs.known ? LaunchNodeDomain{} : LaunchNodeDomain::unknown();
  const std::set<LaunchNodeCoord> *lhsBound = getUpperBoundNodes();
  if (!lhsBound) {
    return result;
  }
  std::set<LaunchNodeCoord> boundNodes = *lhsBound;
  if (rhs.known) {
    boundNodes.clear();
    std::set_difference(lhsBound->begin(), lhsBound->end(), rhs.nodes.begin(),
                        rhs.nodes.end(),
                        std::inserter(boundNodes, boundNodes.end()));
  }
  if (result.known) {
    result.nodes = std::move(boundNodes);
    return result;
  }
  return getUnknownDomainWithBound(std::move(boundNodes));
}

bool LaunchNodeDomain::operator==(const LaunchNodeDomain &rhs) const {
  return known == rhs.known && hasUpperBound == rhs.hasUpperBound &&
         nodes == rhs.nodes;
}

LaunchNodeDomain getFullLaunchNodeDomain(int64_t gridX, int64_t gridY) {
  LaunchNodeDomain result;
  for (int64_t x = 0; x < gridX; ++x) {
    for (int64_t y = 0; y < gridY; ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

LaunchNodeDomain getPipeSourceLaunchNodeDomain(PipeType pipeType) {
  LaunchNodeDomain result;
  result.nodes.insert({pipeType.getSrcX(), pipeType.getSrcY()});
  return result;
}

LaunchNodeDomain
getPipeDestinationLaunchNodeDomain(PipeType pipeType,
                                   const LaunchNodeDomain &baseDomain) {
  LaunchNodeCoord start{pipeType.getDstStartX(), pipeType.getDstStartY()};
  LaunchNodeCoord end{pipeType.getDstEndX(), pipeType.getDstEndY()};
  if (!knownLaunchNodeDomainContains(baseDomain, start) ||
      !knownLaunchNodeDomainContains(baseDomain, end)) {
    return LaunchNodeDomain::unknown();
  }
  LaunchNodeDomain result;
  for (int64_t x = pipeType.getDstStartX(); x <= pipeType.getDstEndX(); ++x) {
    for (int64_t y = pipeType.getDstStartY(); y <= pipeType.getDstEndY(); ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

LaunchNodeDomain getSingleLaunchNodeDomain(LaunchNodeCoord coord) {
  LaunchNodeDomain result;
  result.nodes.insert(coord);
  return result;
}

bool launchNodeDomainsOverlap(const LaunchNodeDomain &lhs,
                              const LaunchNodeDomain &rhs) {
  const std::set<LaunchNodeCoord> *lhsBound = lhs.getUpperBoundNodes();
  const std::set<LaunchNodeCoord> *rhsBound = rhs.getUpperBoundNodes();
  if (!lhsBound || !rhsBound) {
    return true;
  }
  auto lhsIt = lhsBound->begin();
  auto rhsIt = rhsBound->begin();
  while (lhsIt != lhsBound->end() && rhsIt != rhsBound->end()) {
    if (*lhsIt == *rhsIt) {
      return true;
    }
    if (*lhsIt < *rhsIt) {
      ++lhsIt;
    } else {
      ++rhsIt;
    }
  }
  return false;
}

bool knownLaunchNodeDomainContains(const LaunchNodeDomain &domain,
                                   LaunchNodeCoord coord) {
  return domain.known && domain.nodes.find(coord) != domain.nodes.end();
}

LaunchNodeDomain getPipeRecordRoleLaunchNodeDomain(PipeRecordAttr record,
                                                   PipeRole role) {
  LaunchNodeDomain result;
  for (const PipeRecordRoleFacts &facts :
       getPipeRecordRoleFacts(record, role)) {
    for (int64_t nodeX = facts.minX; nodeX <= facts.maxX; ++nodeX) {
      for (int64_t nodeY = facts.minY; nodeY <= facts.maxY; ++nodeY) {
        result.nodes.insert({nodeX, nodeY});
      }
    }
  }
  return result;
}

LaunchNodeDomain getPipeRecordsRoleLaunchNodeDomain(PipeNetRecordsAttr records,
                                                    PipeRole role) {
  LaunchNodeDomain result;
  for (PipeRecordAttr record : records.getPipes()) {
    LaunchNodeDomain recordDomain =
        getPipeRecordRoleLaunchNodeDomain(record, role);
    result = result.unionWith(recordDomain);
  }
  return result;
}

/// Normalize integer-array attributes before verifier-specific interpretation.
static bool readI64ArrayAttr(Operation *op, llvm::StringLiteral name,
                             SmallVectorImpl<int64_t> &values) {
  if (auto dense = op->getAttrOfType<DenseI64ArrayAttr>(name)) {
    values.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return true;
  }
  auto array = op->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    return false;
  }
  for (Attribute attr : array) {
    auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      return false;
    }
    values.push_back(intAttr.getInt());
  }
  return true;
}

FailureOr<std::pair<int64_t, int64_t>> getLaunchGrid(Operation *op) {
  ModuleOp module = mlir::dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  if (!module) {
    return failure();
  }
  SmallVector<int64_t, 2> extents;
  if (!readI64ArrayAttr(module.getOperation(), kLaunchGridAttrName, extents) ||
      extents.size() != 2 || extents[0] <= 0 || extents[1] <= 0) {
    return failure();
  }
  return std::make_pair(extents[0], extents[1]);
}

bool readPipeNetScopeIds(PipeNetScopeOp scopeOp,
                         SmallVectorImpl<int64_t> &ids) {
  return readI64ArrayAttr(scopeOp.getOperation(), kPipeNetIdsAttrName, ids);
}

bool LaunchNodeDomainState::hasPipes() const { return !pipeNetLocs.empty(); }

std::string LaunchNodeDomainState::netName(int64_t netId) const {
  auto it = pipeNetNames.find(netId);
  if (it != pipeNetNames.end() && !it->second.empty()) {
    return it->second;
  }
  return "net_" + std::to_string(netId);
}

LaunchNodeDomain LaunchNodeDomainState::getRoleDomain(int64_t netId,
                                                      PipeRole role) const {
  if (!pipeNetLocs.contains(netId)) {
    return LaunchNodeDomain::unknown();
  }
  if (role == PipeRole::Source) {
    auto it = netSourceDomains.find(netId);
    return it == netSourceDomains.end() ? LaunchNodeDomain{} : it->second;
  }
  if (role == PipeRole::Destination) {
    auto it = netDestinationDomains.find(netId);
    return it == netDestinationDomains.end() ? LaunchNodeDomain{} : it->second;
  }
  LaunchNodeDomain src;
  LaunchNodeDomain dst;
  if (auto it = netSourceDomains.find(netId); it != netSourceDomains.end()) {
    src = it->second;
  }
  if (auto it = netDestinationDomains.find(netId);
      it != netDestinationDomains.end()) {
    dst = it->second;
  }
  return src.unionWith(dst);
}

void LaunchNodeDomainState::recordPipeNet(PipeType pipeType, Location loc,
                                          std::optional<StringRef> name) {
  int64_t pipeNetId = pipeType.getPipeNetId();
  LaunchNodeDomain sourceDomain = getPipeSourceLaunchNodeDomain(pipeType);
  if (!sourceDomain.isSubsetOf(baseDomain)) {
    sourceDomain = LaunchNodeDomain::unknown();
  }
  netSourceDomains[pipeNetId] =
      netSourceDomains[pipeNetId].unionWith(sourceDomain);
  netDestinationDomains[pipeNetId] = netDestinationDomains[pipeNetId].unionWith(
      getPipeDestinationLaunchNodeDomain(pipeType, baseDomain));
  pipeNetLocs[pipeNetId].push_back(loc);
  auto &storedName = pipeNetNames[pipeNetId];
  if (storedName.empty() && name && !name->empty()) {
    storedName = name->str();
  }
}

void LaunchNodeDomainState::recordPipeNetRecords(PipeNetRecordsAttr records,
                                                 Location loc) {
  std::optional<StringRef> name;
  if (StringAttr attr = records.getPipeNetName()) {
    name = attr.getValue();
  }
  int64_t pipeNetId = records.getPipeNetId();
  for (PipeRecordAttr record : records.getPipes()) {
    PipeType pipeType =
        PipeType::get(records.getContext(), record.getSrcX(), record.getSrcY(),
                      record.getDstStartX(), record.getDstStartY(),
                      record.getDstEndX(), record.getDstEndY(), pipeNetId);
    LaunchNodeDomain sourceDomain = getPipeSourceLaunchNodeDomain(pipeType);
    if (!sourceDomain.isSubsetOf(baseDomain)) {
      sourceDomain = LaunchNodeDomain::unknown();
    }
    netSourceDomains[pipeNetId] =
        netSourceDomains[pipeNetId].unionWith(sourceDomain);
    netDestinationDomains[pipeNetId] =
        netDestinationDomains[pipeNetId].unionWith(
            getPipeDestinationLaunchNodeDomain(pipeType, baseDomain));
  }
  // One location identifies the declaration; recording it per row would
  // duplicate every diagnostic note for the same PipeNet.
  pipeNetLocs[pipeNetId].push_back(loc);
  auto &storedName = pipeNetNames[pipeNetId];
  if (storedName.empty() && name && !name->empty()) {
    storedName = name->str();
  }
}

void LaunchNodeDomainState::initialize(ModuleOp module) {
  executionCountAnalysesByFunction.clear();
  FailureOr<std::pair<int64_t, int64_t>> launchGrid = getLaunchGrid(module);
  if (failed(launchGrid)) {
    hasLaunchGrid = false;
  } else {
    hasLaunchGrid = true;
    baseDomain = getFullLaunchNodeDomain(launchGrid->first, launchGrid->second);
  }

  module.walk([&](CreatePipeOp pipe) {
    std::optional<StringRef> name;
    if (auto attr = pipe.getPipeNetNameAttr()) {
      name = attr.getValue();
    }
    recordPipeNet(mlir::cast<PipeType>(pipe.getResult().getType()),
                  pipe.getLoc(), name);
  });
  module.walk([&](PipeNetForeachSrcOp op) {
    recordPipeNetRecords(op.getRecords(), op.getLoc());
  });
  module.walk([&](PipeNetForeachDstOp op) {
    recordPipeNetRecords(op.getRecords(), op.getLoc());
  });
  module.walk([&](SelectPipeSrcOp op) {
    recordPipeNetRecords(op.getRecords(), op.getLoc());
  });
  module.walk([&](SelectPipeDstOp op) {
    recordPipeNetRecords(op.getRecords(), op.getLoc());
  });
}

static std::optional<llvm::APInt>
evaluateLaunchNodeContextValue(Value value, LaunchNodeCoord coord,
                               const LaunchNodeDomainState *state) {
  if (value.getDefiningOp<CoreXOp>() ||
      value.getDefiningOp<ttkernel::MyLogicalXOp>()) {
    return llvm::APInt(IndexType::kInternalStorageBitWidth, coord.x);
  }
  if (value.getDefiningOp<CoreYOp>() ||
      value.getDefiningOp<ttkernel::MyLogicalYOp>()) {
    return llvm::APInt(IndexType::kInternalStorageBitWidth, coord.y);
  }
  if (state) {
    if (auto predicate = value.getDefiningOp<PipeNetPredicateOpInterface>()) {
      if (predicate.getReferencedRecords()) {
        return std::nullopt;
      }
      bool selected = knownLaunchNodeDomainContains(
          state->getRoleDomain(predicate.getReferencedPipeNetId(),
                               predicate.getReferencedRole()),
          coord);
      return llvm::APInt(/*numBits=*/1, selected);
    }
  }
  return std::nullopt;
}

std::optional<bool>
pipeRecordRoleMatchesAtLaunchLocation(PipeRecordAttr record, PipeRole role,
                                      const LaunchExecutionLocation &location) {
  bool hasUnknownDeviceMatch = false;
  for (const PipeRecordRoleFacts &facts :
       getPipeRecordRoleFacts(record, role)) {
    bool nodeMatches =
        location.node.x >= facts.minX && location.node.x <= facts.maxX &&
        location.node.y >= facts.minY && location.node.y <= facts.maxY;
    if (!nodeMatches) {
      continue;
    }
    if (!facts.device) {
      return true;
    }
    if (!location.device || location.deviceDomain != facts.deviceDomain) {
      hasUnknownDeviceMatch = true;
      continue;
    }
    if (location.device == facts.device) {
      return true;
    }
  }
  return hasUnknownDeviceMatch ? std::nullopt : std::optional<bool>(false);
}

static std::optional<bool> evaluatePipeNetPredicateAtLaunchLocation(
    PipeNetPredicateOpInterface predicate,
    const LaunchExecutionLocation &location) {
  PipeNetRecordsAttr records = predicate.getReferencedRecords();
  if (!records) {
    return std::nullopt;
  }
  bool selected = false;
  for (PipeRecordAttr record : records.getPipes()) {
    std::optional<bool> recordMatches = pipeRecordRoleMatchesAtLaunchLocation(
        record, predicate.getReferencedRole(), location);
    if (!recordMatches) {
      return std::nullopt;
    }
    selected |= *recordMatches;
  }
  return selected;
}

static std::optional<llvm::APInt>
evaluateLaunchLocationContextValue(Value value,
                                   const LaunchExecutionLocation &location,
                                   const LaunchNodeDomainState *state) {
  if (auto predicate = value.getDefiningOp<PipeNetPredicateOpInterface>()) {
    if (predicate.getReferencedRecords()) {
      std::optional<bool> selected =
          evaluatePipeNetPredicateAtLaunchLocation(predicate, location);
      return selected ? std::optional<llvm::APInt>(llvm::APInt(
                            /*numBits=*/1, *selected))
                      : std::nullopt;
    }
  }
  if (std::optional<llvm::APInt> nodeValue =
          evaluateLaunchNodeContextValue(value, location.node, state)) {
    return nodeValue;
  }
  if (auto isDeviceOp = value.getDefiningOp<IsDeviceOp>()) {
    if (!location.device || location.deviceDomain != isDeviceOp.getDomain()) {
      return std::nullopt;
    }
    return llvm::APInt(/*numBits=*/1,
                       location.device == isDeviceOp.getDevice());
  }
  if (auto isDeviceInRangeOp = value.getDefiningOp<IsDeviceInRangeOp>()) {
    if (!location.device ||
        location.deviceDomain != isDeviceInRangeOp.getDomain()) {
      return std::nullopt;
    }
    return llvm::APInt(
        /*numBits=*/1,
        deviceRangeContains(isDeviceInRangeOp.getRange(), location.device));
  }
  if (auto currentDeviceOp = value.getDefiningOp<CurrentDeviceIndexOp>()) {
    if (!location.device ||
        location.deviceDomain != currentDeviceOp.getDomain()) {
      return std::nullopt;
    }
    return llvm::APInt(
        IndexType::kInternalStorageBitWidth,
        getLogicalDeviceIndex(currentDeviceOp.getDomain(), location.device));
  }
  return std::nullopt;
}

/// Use shared integer folding for every launch-domain expression.
static IntegerExpressionEvaluator createLaunchLocationIntegerEvaluator(
    const LaunchExecutionLocation &location,
    const LaunchNodeDomainState *state = nullptr) {
  return IntegerExpressionEvaluator(
      [location, state](Value value) -> std::optional<llvm::APInt> {
        return evaluateLaunchLocationContextValue(value, location, state);
      });
}

static IntegerExpressionEvaluator createLaunchLocationIntegerEvaluator(
    const LaunchExecutionLocation &location, const LaunchNodeDomainState *state,
    const IntegerExpressionEvaluator::ValueEvaluator &contextValueEvaluator) {
  return IntegerExpressionEvaluator(
      [location, state,
       &contextValueEvaluator](Value value) -> std::optional<llvm::APInt> {
        if (contextValueEvaluator) {
          if (std::optional<llvm::APInt> contextValue =
                  contextValueEvaluator(value)) {
            return contextValue;
          }
        }
        return evaluateLaunchLocationContextValue(value, location, state);
      });
}

static IntegerExpressionEvaluator
createLaunchNodeIntegerEvaluator(LaunchNodeCoord coord,
                                 const LaunchNodeDomainState *state = nullptr) {
  return createLaunchLocationIntegerEvaluator(LaunchExecutionLocation(coord),
                                              state);
}

std::optional<bool>
evaluatePredicateAtLaunchNode(Value value, LaunchNodeCoord coord,
                              const LaunchNodeDomainState &state) {
  return evaluatePredicateAtLaunchLocation(
      value, LaunchExecutionLocation(coord), state);
}

std::optional<bool>
evaluatePredicateAtLaunchLocation(Value value,
                                  const LaunchExecutionLocation &location,
                                  const LaunchNodeDomainState &state) {
  std::optional<llvm::APInt> maybeValue =
      evaluateIntegerAtLaunchLocation(value, location, state);
  if (!maybeValue || maybeValue->getBitWidth() != 1) {
    return std::nullopt;
  }
  return maybeValue->getBoolValue();
}

std::optional<llvm::APInt>
evaluateIntegerAtLaunchLocation(Value value,
                                const LaunchExecutionLocation &location,
                                const LaunchNodeDomainState &state) {
  return createLaunchLocationIntegerEvaluator(location, &state).evaluate(value);
}

namespace {

static std::optional<std::uint64_t>
evaluateRegionInvocationCountAtLaunchLocation(
    Region &region, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state) {
  Operation *parent = region.getParentOp();
  if (isa<PipeNetScopeOp>(parent)) {
    return 1;
  }
  if (auto ifSrcOp = dyn_cast<IfSrcOp>(parent)) {
    auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
               getPipeSourceLaunchNodeDomain(pipeType), location.node)
               ? 1
               : 0;
  }
  if (auto ifDstOp = dyn_cast<IfDstOp>(parent)) {
    auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
               getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain),
               location.node)
               ? 1
               : 0;
  }
  if (auto foreachSrcOp = dyn_cast<PipeNetForeachSrcOp>(parent)) {
    std::uint64_t count = 0;
    for (PipeRecordAttr record : foreachSrcOp.getRecords().getPipes()) {
      std::optional<bool> matches = pipeRecordRoleMatchesAtLaunchLocation(
          record, PipeRole::Source, location);
      if (!matches) {
        return std::nullopt;
      }
      count += *matches;
    }
    return count;
  }
  if (auto foreachDstOp = dyn_cast<PipeNetForeachDstOp>(parent)) {
    std::uint64_t count = 0;
    for (PipeRecordAttr record : foreachDstOp.getRecords().getPipes()) {
      std::optional<bool> matches = pipeRecordRoleMatchesAtLaunchLocation(
          record, PipeRole::Destination, location);
      if (!matches) {
        return std::nullopt;
      }
      count += *matches;
    }
    return count;
  }
  if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(parent)) {
    LaunchNodeDomainResult trueDomain =
        getAffineIfLaunchNodeDomain(affineIfOp, state.baseDomain);
    if (!trueDomain.domain.known) {
      return std::nullopt;
    }
    bool selectsThen =
        knownLaunchNodeDomainContains(trueDomain.domain, location.node);
    return (selectsThen == (region.getRegionNumber() == 0)) ? 1 : 0;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::uint64_t> getRegionInvocationCountAtLaunchLocation(
    Region &region, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state) {
  return evaluateRegionInvocationCountAtLaunchLocation(region, location, state);
}

std::optional<std::uint64_t>
getExactExecutionCountAtLaunchNode(Operation *op, LaunchNodeCoord coord,
                                   const LaunchNodeDomainState &state) {
  return getExactExecutionCountAtLaunchLocation(
      op, LaunchExecutionLocation(coord), state);
}

std::optional<std::uint64_t>
getExactExecutionCountAtLaunchLocation(Operation *op,
                                       const LaunchExecutionLocation &location,
                                       const LaunchNodeDomainState &state) {
  func::FuncOp function = op->getParentOfType<func::FuncOp>();
  if (!function) {
    return std::nullopt;
  }
  auto &functionCache =
      state.executionCountAnalysesByFunction[function.getOperation()];
  if (!functionCache.sharedState) {
    functionCache.sharedState =
        std::make_unique<ExecutionCountAnalysisSharedState>(function.getBody());
  }
  auto createAnalysis = [&] {
    return std::make_unique<ExecutionCountAnalysis>(
        *functionCache.sharedState,
        [location, &state](Value value) {
          return evaluateLaunchLocationContextValue(value, location, &state);
        },
        [location, &state](Region &region) {
          return getRegionInvocationCountAtLaunchLocation(region, location,
                                                          state);
        });
  };
  ExecutionCountAnalysis *analysis = nullptr;
  if (!location.device) {
    auto [analysisIt, inserted] =
        functionCache.analysesByNode.try_emplace(location.node);
    if (inserted) {
      analysisIt->second = createAnalysis();
    }
    analysis = analysisIt->second.get();
  } else {
    analysis = &functionCache.analysesByDeviceLocation.getOrCreate(
        location, createAnalysis);
  }
  assert(analysis && "execution-count analysis cache returned null");
  return analysis->getExecutionCount(op);
}

bool hasExactEmptyLaunchDomain(Operation *op,
                               const LaunchNodeDomainState &state) {
  if (!state.hasLaunchGrid || state.sawError) {
    return false;
  }
  for (LaunchNodeCoord node : state.baseDomain.nodes) {
    std::optional<std::uint64_t> executionCount =
        getExactExecutionCountAtLaunchNode(op, node, state);
    if (!executionCount || *executionCount != 0) {
      return false;
    }
  }
  return true;
}

/// Return true if evaluating `value` can depend on the current launch
/// coordinate.
static bool dependsOnCoord(Value value, llvm::DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }
  Operation *op = value.getDefiningOp();
  bool result = false;
  if (op) {
    if (mlir::isa<CoreXOp, CoreYOp, PipeNetPredicateOpInterface,
                  ttkernel::MyLogicalXOp, ttkernel::MyLogicalYOp>(op)) {
      result = true;
    } else {
      for (Value operand : op->getOperands()) {
        if (dependsOnCoord(operand, cache)) {
          result = true;
          break;
        }
      }
    }
  }
  cache[value] = result;
  return result;
}

/// Compute the exact set of launch nodes satisfying an `affine.if` integer set.
static LaunchNodeDomainResult
getAffineIfLaunchNodeDomain(affine::AffineIfOp ifOp,
                            const LaunchNodeDomain &baseDomain) {
  IntegerSet set = ifOp.getIntegerSet();
  ValueRange operands = ifOp.getOperands();
  MLIRContext *ctx = ifOp.getContext();

  SmallVector<AffineExpr> constraintExprs;
  constraintExprs.reserve(set.getNumConstraints());
  for (unsigned idx = 0; idx < set.getNumConstraints(); ++idx) {
    constraintExprs.push_back(set.getConstraint(idx));
  }
  AffineMap map = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                                 constraintExprs, ctx);

  LaunchNodeDomain result;
  SmallVector<Attribute> operandConstants(set.getNumInputs());
  for (LaunchNodeCoord coord : baseDomain.nodes) {
    IntegerExpressionEvaluator integerEvaluator =
        createLaunchNodeIntegerEvaluator(coord);
    bool resolved = true;
    for (unsigned idx = 0; idx < set.getNumInputs(); ++idx) {
      std::optional<llvm::APInt> maybeValue =
          integerEvaluator.evaluate(operands[idx]);
      if (!maybeValue) {
        resolved = false;
        break;
      }
      operandConstants[idx] =
          IntegerAttr::get(operands[idx].getType(), *maybeValue);
    }
    if (!resolved) {
      return {LaunchNodeDomain::unknownWithin(baseDomain), ifOp};
    }
    SmallVector<Attribute> folded;
    if (failed(map.constantFold(operandConstants, folded))) {
      return {LaunchNodeDomain::unknownWithin(baseDomain), ifOp};
    }
    bool ok = true;
    for (unsigned idx = 0; idx < set.getNumConstraints(); ++idx) {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(folded[idx]);
      if (!intAttr) {
        return {LaunchNodeDomain::unknownWithin(baseDomain), ifOp};
      }
      int64_t value = intAttr.getInt();
      if (set.isEq(idx) ? value != 0 : value < 0) {
        ok = false;
        break;
      }
    }
    if (ok) {
      result.nodes.insert(coord);
    }
  }
  return {result, nullptr};
}

namespace {

enum class UnresolvedControlFrameKind { ScfIf, AffineIf, ScfFor };

struct UnresolvedControlFrame {
  UnresolvedControlFrameKind kind = UnresolvedControlFrameKind::ScfIf;
  Operation *operation = nullptr;
  std::size_t regionNumber = 0;
  IntegerSetAttr affinePredicate;
  SmallVector<Value, 3> controlValues;
};

/// Structured-control frames that determine an unresolved count.
struct UnresolvedExecutionCountContext {
  func::FuncOp function;
  SmallVector<UnresolvedControlFrame> frames;
};

static std::optional<UnresolvedExecutionCountContext>
getUnresolvedExecutionCountContext(
    Operation *op, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state,
    const IntegerExpressionEvaluator::ValueEvaluator &contextValueEvaluator,
    Operation *exclusiveAncestor = nullptr) {
  UnresolvedExecutionCountContext context;
  context.function = op->getParentOfType<func::FuncOp>();
  if (!context.function ||
      (exclusiveAncestor && !exclusiveAncestor->isProperAncestor(op))) {
    return std::nullopt;
  }
  IntegerExpressionEvaluator integerEvaluator =
      createLaunchLocationIntegerEvaluator(location, &state,
                                           contextValueEvaluator);
  Operation *current = op;
  while (Block *block = current->getBlock()) {
    Operation *parent = block->getParentOp();
    if (auto function = dyn_cast_if_present<func::FuncOp>(parent)) {
      if (block != &function.getBody().front()) {
        return std::nullopt;
      }
      context.function = function;
      break;
    }
    if (!parent) {
      return std::nullopt;
    }
    if (parent == exclusiveAncestor) {
      break;
    }

    Region *region = block->getParent();
    if (isa<PipeNetScopeOp>(parent)) {
      current = parent;
      continue;
    }
    if (auto ifSrcOp = dyn_cast<IfSrcOp>(parent)) {
      auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
      if (!knownLaunchNodeDomainContains(
              getPipeSourceLaunchNodeDomain(pipeType), location.node)) {
        return std::nullopt;
      }
      current = parent;
      continue;
    }
    if (auto ifDstOp = dyn_cast<IfDstOp>(parent)) {
      auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
      if (!knownLaunchNodeDomainContains(
              getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain),
              location.node)) {
        return std::nullopt;
      }
      current = parent;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      std::optional<llvm::APInt> maybeCondition =
          integerEvaluator.evaluate(ifOp.getCondition());
      std::optional<bool> maybeSelected;
      if (maybeCondition && maybeCondition->getBitWidth() == 1) {
        maybeSelected = maybeCondition->getBoolValue();
      }
      if (maybeSelected) {
        if (region->getRegionNumber() != (*maybeSelected ? 0 : 1)) {
          return std::nullopt;
        }
        current = parent;
        continue;
      }
      context.frames.push_back({UnresolvedControlFrameKind::ScfIf,
                                parent,
                                region->getRegionNumber(),
                                nullptr,
                                {ifOp.getCondition()}});
    } else if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(parent);
               affineIfOp && state.hasLaunchGrid) {
      LaunchNodeDomainResult trueDomain =
          getAffineIfLaunchNodeDomain(affineIfOp, state.baseDomain);
      if (trueDomain.domain.known) {
        bool selectsThen =
            knownLaunchNodeDomainContains(trueDomain.domain, location.node);
        if (region->getRegionNumber() != (selectsThen ? 0 : 1)) {
          return std::nullopt;
        }
        current = parent;
        continue;
      }
      UnresolvedControlFrame &frame = context.frames.emplace_back();
      frame.kind = UnresolvedControlFrameKind::AffineIf;
      frame.operation = parent;
      frame.regionNumber = region->getRegionNumber();
      frame.affinePredicate = affineIfOp.getConditionAttr();
      llvm::append_range(frame.controlValues, affineIfOp.getOperands());
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      context.frames.push_back(
          {UnresolvedControlFrameKind::ScfFor,
           parent,
           region->getRegionNumber(),
           nullptr,
           {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}});
    } else if (isa<scf::ExecuteRegionOp>(parent)) {
      current = parent;
      continue;
    } else {
      return std::nullopt;
    }
    current = parent;
  }
  if (!context.function) {
    return std::nullopt;
  }
  return context;
}

/// Return true when two SSA values have the same runtime value at their launch
/// locations and active call sites.
static bool proveEqualValuesAtLaunchLocations(
    Value lhsValue, const LaunchExecutionLocation &lhsLocation,
    const IntegerExpressionEvaluator::ValueEvaluator &lhsContextValueEvaluator,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    Value rhsValue, const LaunchExecutionLocation &rhsLocation,
    const IntegerExpressionEvaluator::ValueEvaluator &rhsContextValueEvaluator,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument,
    const LaunchNodeDomainState &state,
    llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  std::pair<Value, Value> cacheKey{lhsValue, rhsValue};
  if (auto it = cache.find(cacheKey); it != cache.end()) {
    return it->second;
  }
  cache[cacheKey] = false;

  std::optional<llvm::APInt> maybeLhsValue =
      createLaunchLocationIntegerEvaluator(lhsLocation, &state,
                                           lhsContextValueEvaluator)
          .evaluate(lhsValue);
  std::optional<llvm::APInt> maybeRhsValue =
      createLaunchLocationIntegerEvaluator(rhsLocation, &state,
                                           rhsContextValueEvaluator)
          .evaluate(rhsValue);
  if (maybeLhsValue && maybeRhsValue) {
    bool equal = *maybeLhsValue == *maybeRhsValue;
    cache[cacheKey] = equal;
    return equal;
  }

  auto lhsArgument = dyn_cast<BlockArgument>(lhsValue);
  auto rhsArgument = dyn_cast<BlockArgument>(rhsValue);
  if (lhsArgument || rhsArgument) {
    if (!lhsArgument || !rhsArgument) {
      return false;
    }

    Block *lhsOwner = lhsArgument.getOwner();
    Block *rhsOwner = rhsArgument.getOwner();
    auto lhsFunction =
        dyn_cast_if_present<func::FuncOp>(lhsOwner->getParentOp());
    auto rhsFunction =
        dyn_cast_if_present<func::FuncOp>(rhsOwner->getParentOp());
    bool lhsIsFunctionArgument =
        lhsFunction && lhsOwner == &lhsFunction.getBody().front();
    bool rhsIsFunctionArgument =
        rhsFunction && rhsOwner == &rhsFunction.getBody().front();
    if (lhsIsFunctionArgument || rhsIsFunctionArgument) {
      if (!lhsIsFunctionArgument || !rhsIsFunctionArgument) {
        return false;
      }
      // The launch ABI supplies one common argument vector to every node of a
      // kernel-thread function. Helper arguments instead take their values
      // from the active call site.
      if (lhsFunction->hasAttr(kKernelThreadAttrName) ||
          rhsFunction->hasAttr(kKernelThreadAttrName)) {
        bool equal = lhsValue == rhsValue && lhsFunction == rhsFunction &&
                     lhsFunction->hasAttr(kKernelThreadAttrName) &&
                     rhsFunction->hasAttr(kKernelThreadAttrName);
        cache[cacheKey] = equal;
        return equal;
      }
      std::optional<Value> maybeLhsOperand =
          resolveLhsFunctionArgument(lhsArgument);
      std::optional<Value> maybeRhsOperand =
          resolveRhsFunctionArgument(rhsArgument);
      if (!maybeLhsOperand || !maybeRhsOperand) {
        return false;
      }
      bool equal = proveEqualValuesAtLaunchLocations(
          *maybeLhsOperand, lhsLocation, lhsContextValueEvaluator,
          resolveLhsFunctionArgument, *maybeRhsOperand, rhsLocation,
          rhsContextValueEvaluator, resolveRhsFunctionArgument, state, cache);
      cache[cacheKey] = equal;
      return equal;
    }

    auto lhsForOp = dyn_cast_if_present<scf::ForOp>(lhsOwner->getParentOp());
    auto rhsForOp = dyn_cast_if_present<scf::ForOp>(rhsOwner->getParentOp());
    bool equal =
        lhsForOp && rhsForOp && lhsValue == lhsForOp.getInductionVar() &&
        rhsValue == rhsForOp.getInductionVar() &&
        proveEqualValuesAtLaunchLocations(
            lhsForOp.getLowerBound(), lhsLocation, lhsContextValueEvaluator,
            resolveLhsFunctionArgument, rhsForOp.getLowerBound(), rhsLocation,
            rhsContextValueEvaluator, resolveRhsFunctionArgument, state,
            cache) &&
        proveEqualValuesAtLaunchLocations(
            lhsForOp.getUpperBound(), lhsLocation, lhsContextValueEvaluator,
            resolveLhsFunctionArgument, rhsForOp.getUpperBound(), rhsLocation,
            rhsContextValueEvaluator, resolveRhsFunctionArgument, state,
            cache) &&
        proveEqualValuesAtLaunchLocations(
            lhsForOp.getStep(), lhsLocation, lhsContextValueEvaluator,
            resolveLhsFunctionArgument, rhsForOp.getStep(), rhsLocation,
            rhsContextValueEvaluator, resolveRhsFunctionArgument, state, cache);
    cache[cacheKey] = equal;
    return equal;
  }

  if (lhsValue == rhsValue && lhsLocation == rhsLocation) {
    cache[cacheKey] = true;
    return true;
  }

  Operation *lhsDefiningOp = lhsValue.getDefiningOp();
  Operation *rhsDefiningOp = rhsValue.getDefiningOp();
  if (!lhsDefiningOp || !rhsDefiningOp || lhsDefiningOp->getNumRegions() != 0 ||
      rhsDefiningOp->getNumRegions() != 0 ||
      lhsDefiningOp->getNumOperands() == 0 ||
      rhsDefiningOp->getNumOperands() == 0 ||
      !isMemoryEffectFree(lhsDefiningOp) ||
      !isMemoryEffectFree(rhsDefiningOp)) {
    return false;
  }
  auto lhsResult = dyn_cast<OpResult>(lhsValue);
  auto rhsResult = dyn_cast<OpResult>(rhsValue);
  if (!lhsResult || !rhsResult ||
      lhsResult.getResultNumber() != rhsResult.getResultNumber()) {
    return false;
  }
  auto proveEqualOperands = [&](Value lhsOperand, Value rhsOperand) {
    return proveEqualValuesAtLaunchLocations(
        lhsOperand, lhsLocation, lhsContextValueEvaluator,
        resolveLhsFunctionArgument, rhsOperand, rhsLocation,
        rhsContextValueEvaluator, resolveRhsFunctionArgument, state, cache);
  };
  if (lhsDefiningOp != rhsDefiningOp &&
      !OperationEquivalence::isEquivalentTo(
          lhsDefiningOp, rhsDefiningOp,
          OperationEquivalence::ignoreValueEquivalence, nullptr,
          OperationEquivalence::Flags::IgnoreLocations)) {
    return false;
  }
  bool equal = llvm::all_of(
      llvm::zip(lhsDefiningOp->getOperands(), rhsDefiningOp->getOperands()),
      [&](auto operands) {
        return proveEqualOperands(std::get<0>(operands), std::get<1>(operands));
      });
  cache[cacheKey] = equal;
  return equal;
}

static std::optional<llvm::APInt> getIntegerConstant(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant))) {
    return std::nullopt;
  }
  auto integer = dyn_cast<IntegerAttr>(constant);
  if (!integer) {
    return std::nullopt;
  }
  return integer.getValue();
}

// Prove equality between expressions rooted in typed dispatch conditions.
// Polarity tracks whether the caller observes zero or nonzero as true.
static bool proveEquivalentDispatchConditionExpressions(Value lhsValue,
                                                        bool lhsNonzeroIsTrue,
                                                        Value rhsValue,
                                                        bool rhsNonzeroIsTrue) {
  std::optional<llvm::APInt> lhsConstant = getIntegerConstant(lhsValue);
  std::optional<llvm::APInt> rhsConstant = getIntegerConstant(rhsValue);
  if (lhsConstant || rhsConstant) {
    return lhsConstant && rhsConstant &&
           (lhsConstant->isZero() != lhsNonzeroIsTrue) ==
               (rhsConstant->isZero() != rhsNonzeroIsTrue);
  }

  auto lhsComparison = lhsValue.getDefiningOp<arith::CmpIOp>();
  auto rhsComparison = rhsValue.getDefiningOp<arith::CmpIOp>();
  if (lhsComparison || rhsComparison) {
    if (!lhsComparison || !rhsComparison) {
      return false;
    }
    auto stripZeroComparison =
        [](arith::CmpIOp comparison) -> std::optional<std::pair<Value, bool>> {
      arith::CmpIPredicate predicate = comparison.getPredicate();
      if (predicate != arith::CmpIPredicate::eq &&
          predicate != arith::CmpIPredicate::ne) {
        return std::nullopt;
      }
      if (std::optional<llvm::APInt> lhs =
              getIntegerConstant(comparison.getLhs());
          lhs && lhs->isZero()) {
        return std::pair<Value, bool>{comparison.getRhs(),
                                      predicate == arith::CmpIPredicate::ne};
      }
      if (std::optional<llvm::APInt> rhs =
              getIntegerConstant(comparison.getRhs());
          rhs && rhs->isZero()) {
        return std::pair<Value, bool>{comparison.getLhs(),
                                      predicate == arith::CmpIPredicate::ne};
      }
      return std::nullopt;
    };
    std::optional<std::pair<Value, bool>> lhsExpression =
        stripZeroComparison(lhsComparison);
    std::optional<std::pair<Value, bool>> rhsExpression =
        stripZeroComparison(rhsComparison);
    if (!lhsExpression || !rhsExpression) {
      return false;
    }
    return proveEquivalentDispatchConditionExpressions(
        lhsExpression->first, lhsNonzeroIsTrue == lhsExpression->second,
        rhsExpression->first, rhsNonzeroIsTrue == rhsExpression->second);
  }

  auto lhsCall = lhsValue.getDefiningOp<OpaqueCallOp>();
  auto rhsCall = rhsValue.getDefiningOp<OpaqueCallOp>();
  if (lhsCall || rhsCall) {
    return lhsCall && rhsCall && lhsNonzeroIsTrue == rhsNonzeroIsTrue &&
           lhsCall.getResult() == lhsValue && rhsCall.getResult() == rhsValue &&
           lhsCall.getConditionResultAttr() &&
           lhsCall.getConditionResultAttr() == rhsCall.getConditionResultAttr();
  }

  if (lhsNonzeroIsTrue != rhsNonzeroIsTrue) {
    return false;
  }
  auto proveBinaryOperands = [&](auto lhsOperation, auto rhsOperation) {
    return lhsOperation && rhsOperation &&
           lhsOperation.getType().isInteger(1) &&
           rhsOperation.getType().isInteger(1) &&
           proveEquivalentDispatchConditionExpressions(
               lhsOperation.getLhs(), true, rhsOperation.getLhs(), true) &&
           proveEquivalentDispatchConditionExpressions(
               lhsOperation.getRhs(), true, rhsOperation.getRhs(), true);
  };
  if (auto lhsAnd = lhsValue.getDefiningOp<arith::AndIOp>()) {
    return proveBinaryOperands(lhsAnd, rhsValue.getDefiningOp<arith::AndIOp>());
  }
  if (auto lhsOr = lhsValue.getDefiningOp<arith::OrIOp>()) {
    return proveBinaryOperands(lhsOr, rhsValue.getDefiningOp<arith::OrIOp>());
  }
  if (auto lhsXor = lhsValue.getDefiningOp<arith::XOrIOp>()) {
    return proveBinaryOperands(lhsXor, rhsValue.getDefiningOp<arith::XOrIOp>());
  }
  return false;
}

static bool proveEquivalentUnresolvedExecutionContexts(
    const UnresolvedExecutionCountContext &lhsContext,
    const LaunchExecutionLocation &lhsLocation,
    const IntegerExpressionEvaluator::ValueEvaluator &lhsContextValueEvaluator,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    const UnresolvedExecutionCountContext &rhsContext,
    const LaunchExecutionLocation &rhsLocation,
    const IntegerExpressionEvaluator::ValueEvaluator &rhsContextValueEvaluator,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument,
    const LaunchNodeDomainState &state, bool requireConditionalExecution) {
  bool sameFunction = lhsContext.function == rhsContext.function;
  if (lhsContext.frames.size() != rhsContext.frames.size() ||
      (!requireConditionalExecution && !sameFunction) ||
      (requireConditionalExecution && !sameFunction &&
       !(lhsLocation == rhsLocation))) {
    return false;
  }

  llvm::DenseMap<std::pair<Value, Value>, bool> equalValueCache;
  for (auto &&[lhsFrame, rhsFrame] :
       llvm::zip_equal(lhsContext.frames, rhsContext.frames)) {
    if (lhsFrame.kind != rhsFrame.kind ||
        lhsFrame.regionNumber != rhsFrame.regionNumber ||
        lhsFrame.affinePredicate != rhsFrame.affinePredicate ||
        lhsFrame.controlValues.size() != rhsFrame.controlValues.size() ||
        (requireConditionalExecution &&
         lhsFrame.kind == UnresolvedControlFrameKind::ScfFor) ||
        (requireConditionalExecution && !sameFunction &&
         lhsFrame.kind == UnresolvedControlFrameKind::AffineIf)) {
      return false;
    }
    for (auto &&[lhsValue, rhsValue] :
         llvm::zip_equal(lhsFrame.controlValues, rhsFrame.controlValues)) {
      bool equalValue = sameFunction &&
                        proveEqualValuesAtLaunchLocations(
                            lhsValue, lhsLocation, lhsContextValueEvaluator,
                            resolveLhsFunctionArgument, rhsValue, rhsLocation,
                            rhsContextValueEvaluator,
                            resolveRhsFunctionArgument, state, equalValueCache);
      if (!equalValue && lhsLocation == rhsLocation &&
          requireConditionalExecution &&
          lhsFrame.kind == UnresolvedControlFrameKind::ScfIf) {
        equalValue = proveEquivalentDispatchConditionExpressions(
            lhsValue, true, rhsValue, true);
      }
      if (!equalValue) {
        return false;
      }
    }
  }
  return !requireConditionalExecution || !lhsContext.frames.empty();
}

} // namespace

bool proveEqualUnresolvedExecutionCountAtLaunchNodes(
    Operation *lhs, LaunchNodeCoord lhsCoord, Operation *rhs,
    LaunchNodeCoord rhsCoord, const LaunchNodeDomainState &state,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument) {
  return proveEqualUnresolvedExecutionCountAtLaunchLocations(
      lhs, LaunchExecutionLocation(lhsCoord), rhs,
      LaunchExecutionLocation(rhsCoord), state, resolveLhsFunctionArgument,
      resolveRhsFunctionArgument);
}

bool proveEqualUnresolvedExecutionCountAtLaunchLocations(
    Operation *lhs, const LaunchExecutionLocation &lhsLocation, Operation *rhs,
    const LaunchExecutionLocation &rhsLocation,
    const LaunchNodeDomainState &state,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument) {
  return proveEqualUnresolvedExecutionCountWithinScopesAtLaunchLocations(
      lhs, nullptr, lhsLocation, rhs, nullptr, rhsLocation, state,
      IntegerExpressionEvaluator::ValueEvaluator(),
      IntegerExpressionEvaluator::ValueEvaluator(), resolveLhsFunctionArgument,
      resolveRhsFunctionArgument);
}

bool proveEqualUnresolvedExecutionCountWithinScopesAtLaunchLocations(
    Operation *lhs, Operation *lhsExclusiveAncestor,
    const LaunchExecutionLocation &lhsLocation, Operation *rhs,
    Operation *rhsExclusiveAncestor, const LaunchExecutionLocation &rhsLocation,
    const LaunchNodeDomainState &state,
    IntegerExpressionEvaluator::ValueEvaluator lhsContextValueEvaluator,
    IntegerExpressionEvaluator::ValueEvaluator rhsContextValueEvaluator,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument) {
  std::optional<UnresolvedExecutionCountContext> maybeLhsContext =
      getUnresolvedExecutionCountContext(lhs, lhsLocation, state,
                                         lhsContextValueEvaluator,
                                         lhsExclusiveAncestor);
  std::optional<UnresolvedExecutionCountContext> maybeRhsContext =
      getUnresolvedExecutionCountContext(rhs, rhsLocation, state,
                                         rhsContextValueEvaluator,
                                         rhsExclusiveAncestor);
  if (!maybeLhsContext || !maybeRhsContext) {
    return false;
  }
  return proveEquivalentUnresolvedExecutionContexts(
      *maybeLhsContext, lhsLocation, lhsContextValueEvaluator,
      resolveLhsFunctionArgument, *maybeRhsContext, rhsLocation,
      rhsContextValueEvaluator, resolveRhsFunctionArgument, state,
      /*requireConditionalExecution=*/false);
}

bool proveEquivalentConditionalExecutionAtLaunchNodes(
    Operation *lhs, LaunchNodeCoord lhsCoord, Operation *rhs,
    LaunchNodeCoord rhsCoord, const LaunchNodeDomainState &state) {
  std::optional<UnresolvedExecutionCountContext> maybeLhsContext =
      getUnresolvedExecutionCountContext(
          lhs, LaunchExecutionLocation(lhsCoord), state,
          IntegerExpressionEvaluator::ValueEvaluator());
  std::optional<UnresolvedExecutionCountContext> maybeRhsContext =
      getUnresolvedExecutionCountContext(
          rhs, LaunchExecutionLocation(rhsCoord), state,
          IntegerExpressionEvaluator::ValueEvaluator());
  if (!maybeLhsContext || !maybeRhsContext) {
    return false;
  }
  auto resolveNoFunctionArguments = [](BlockArgument) -> std::optional<Value> {
    return std::nullopt;
  };
  return proveEquivalentUnresolvedExecutionContexts(
      *maybeLhsContext, LaunchExecutionLocation(lhsCoord),
      IntegerExpressionEvaluator::ValueEvaluator(), resolveNoFunctionArguments,
      *maybeRhsContext, LaunchExecutionLocation(rhsCoord),
      IntegerExpressionEvaluator::ValueEvaluator(), resolveNoFunctionArguments,
      state,
      /*requireConditionalExecution=*/true);
}

bool proveEqualExecutionCountAtLaunchNodes(Operation *lhs,
                                           LaunchNodeCoord lhsCoord,
                                           Operation *rhs,
                                           LaunchNodeCoord rhsCoord,
                                           const LaunchNodeDomainState &state) {
  return proveEqualExecutionCountAtLaunchLocations(
      lhs, LaunchExecutionLocation(lhsCoord), rhs,
      LaunchExecutionLocation(rhsCoord), state);
}

bool proveEqualExecutionCountAtLaunchLocations(
    Operation *lhs, const LaunchExecutionLocation &lhsLocation, Operation *rhs,
    const LaunchExecutionLocation &rhsLocation,
    const LaunchNodeDomainState &state) {
  std::optional<std::uint64_t> maybeLhsCount =
      getExactExecutionCountAtLaunchLocation(lhs, lhsLocation, state);
  std::optional<std::uint64_t> maybeRhsCount =
      getExactExecutionCountAtLaunchLocation(rhs, rhsLocation, state);
  if (maybeLhsCount && maybeRhsCount) {
    return *maybeLhsCount == *maybeRhsCount;
  }
  auto resolveNoFunctionArguments = [](BlockArgument) -> std::optional<Value> {
    return std::nullopt;
  };
  return proveEqualUnresolvedExecutionCountAtLaunchLocations(
      lhs, lhsLocation, rhs, rhsLocation, state, resolveNoFunctionArguments,
      resolveNoFunctionArguments);
}

/// Find a source file location through common composed MLIR location wrappers.
static FileLineColLoc findFileLineColLoc(Location loc) {
  if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
    return fileLoc;
  }
  if (auto fused = mlir::dyn_cast<FusedLoc>(loc)) {
    for (Location inner : fused.getLocations()) {
      if (auto fileLoc = findFileLineColLoc(inner)) {
        return fileLoc;
      }
    }
  }
  if (auto call = mlir::dyn_cast<CallSiteLoc>(loc)) {
    if (auto fileLoc = findFileLineColLoc(call.getCallee())) {
      return fileLoc;
    }
    if (auto fileLoc = findFileLineColLoc(call.getCaller())) {
      return fileLoc;
    }
  }
  return {};
}

Operation *pickEarlierBySourceLoc(Operation *lhs, Operation *rhs) {
  if (!lhs) {
    return rhs;
  }
  if (!rhs) {
    return lhs;
  }
  FileLineColLoc lhsLoc = findFileLineColLoc(lhs->getLoc());
  FileLineColLoc rhsLoc = findFileLineColLoc(rhs->getLoc());
  if (lhsLoc && rhsLoc) {
    auto key = [](FileLineColLoc loc) {
      return std::tuple(loc.getFilename().getValue(), loc.getLine(),
                        loc.getColumn());
    };
    return key(lhsLoc) <= key(rhsLoc) ? lhs : rhs;
  }
  std::string lhsStr;
  std::string rhsStr;
  llvm::raw_string_ostream(lhsStr) << lhs->getLoc();
  llvm::raw_string_ostream(rhsStr) << rhs->getLoc();
  return lhsStr <= rhsStr ? lhs : rhs;
}

/// Split the current domain using an exactly known true-domain.
static BranchLaunchNodeDomains
exactBranches(const LaunchNodeDomain &trueDomain,
              const LaunchNodeDomain &current,
              const LaunchNodeDomain &baseDomain) {
  return {current.intersectWith(trueDomain),
          current.intersectWith(baseDomain.subtract(trueDomain))};
}

/// Recursively compute branch domains for PipeNet predicates and coordinate
/// predicates while preserving unknown domains for unevaluable expressions.
static BranchLaunchNodeDomains
getBranchDomainsImpl(Value condition, const LaunchNodeDomain &current,
                     const LaunchNodeDomainState &state,
                     llvm::DenseMap<Value, bool> &coordCache) {
  if (auto pred = condition.getDefiningOp<PipeNetPredicateOpInterface>()) {
    LaunchNodeDomain roleDomain = state.getRoleDomain(
        pred.getReferencedPipeNetId(), pred.getReferencedRole());
    if (pred.getReferencedRecords()) {
      return {current.intersectWith(roleDomain), current};
    }
    return exactBranches(roleDomain, current, state.baseDomain);
  }
  if (auto andOp = condition.getDefiningOp<arith::AndIOp>()) {
    BranchLaunchNodeDomains lhs =
        getBranchDomainsImpl(andOp.getLhs(), current, state, coordCache);
    BranchLaunchNodeDomains rhs =
        getBranchDomainsImpl(andOp.getRhs(), current, state, coordCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(lhs.unanalyzableOp, rhs.unanalyzableOp);
    return {
        lhs.thenDomain.intersectWith(rhs.thenDomain),
        lhs.elseDomain.unionWith(lhs.thenDomain.intersectWith(rhs.elseDomain)),
        unanalyzable};
  }
  if (auto orOp = condition.getDefiningOp<arith::OrIOp>()) {
    BranchLaunchNodeDomains lhs =
        getBranchDomainsImpl(orOp.getLhs(), current, state, coordCache);
    BranchLaunchNodeDomains rhs =
        getBranchDomainsImpl(orOp.getRhs(), current, state, coordCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(lhs.unanalyzableOp, rhs.unanalyzableOp);
    return {
        lhs.thenDomain.unionWith(lhs.elseDomain.intersectWith(rhs.thenDomain)),
        lhs.elseDomain.intersectWith(rhs.elseDomain), unanalyzable};
  }
  if (!dependsOnCoord(condition, coordCache)) {
    return {current, current};
  }
  LaunchNodeDomain trueDomain;
  for (LaunchNodeCoord coord : state.baseDomain.nodes) {
    std::optional<bool> maybeValue =
        evaluatePredicateAtLaunchNode(condition, coord, state);
    if (!maybeValue) {
      return {LaunchNodeDomain::unknownWithin(current),
              LaunchNodeDomain::unknownWithin(current),
              condition.getDefiningOp()};
    }
    if (*maybeValue) {
      trueDomain.nodes.insert(coord);
    }
  }
  BranchLaunchNodeDomains result =
      exactBranches(trueDomain, current, state.baseDomain);
  return {result.thenDomain, result.elseDomain, nullptr};
}

/// Compute the true and false launch domains for a branch condition.
static BranchLaunchNodeDomains
getBranchLaunchNodeDomains(Value condition, const LaunchNodeDomain &current,
                           const LaunchNodeDomainState &state) {
  llvm::DenseMap<Value, bool> coordCache;
  return getBranchDomainsImpl(condition, current, state, coordCache);
}

/// Decode the PipeNet role metadata carried by one `ttl.pipenet_scope`.
static std::optional<PipeNetScopeLaunchNodeDomains>
getPipeNetScopeLaunchNodeDomains(PipeNetScopeOp scopeOp,
                                 LaunchNodeDomainState &state,
                                 bool emitDiagnostics) {
  auto recordError = [&](const Twine &message) {
    state.sawError = true;
    state.errorOperation = scopeOp;
    state.errorMessage = message.str();
    if (emitDiagnostics) {
      scopeOp.emitOpError() << message;
    }
  };
  SmallVector<int64_t> ids;
  SmallVector<int64_t> roles;
  if (!readI64ArrayAttr(scopeOp.getOperation(), kPipeNetIdsAttrName, ids) ||
      !readI64ArrayAttr(scopeOp.getOperation(), kPipeNetRolesAttrName, roles)) {
    recordError(Twine("requires `") + kPipeNetIdsAttrName + "` and `" +
                kPipeNetRolesAttrName + "` attributes");
    return std::nullopt;
  }
  if (ids.size() != roles.size()) {
    recordError("requires equal-length PipeNet id and role arrays");
    return std::nullopt;
  }
  PipeNetScopeLaunchNodeDomains result;
  for (auto [pipeNetId, roleValue] : llvm::zip_equal(ids, roles)) {
    if (roleValue != static_cast<int64_t>(PipeRole::Source) &&
        roleValue != static_cast<int64_t>(PipeRole::Destination)) {
      recordError(Twine("has invalid PipeNet role ") + Twine(roleValue) +
                  " (expected 0=src or 1=dst)");
      return std::nullopt;
    }
    auto role = static_cast<PipeRole>(roleValue);
    LaunchNodeDomain roleDomain = state.getRoleDomain(pipeNetId, role);
    result.domain = result.domain.unionWith(roleDomain);
    result.roles.emplace_back(pipeNetId, role);
  }
  return result;
}

ChangeResult LaunchNodeDomainLattice::join(const AbstractDenseLattice &rhs) {
  const auto &other = static_cast<const LaunchNodeDomainLattice &>(rhs);
  LaunchNodeDomain joined = domain.unionWith(other.domain);
  Operation *carriedOp =
      pickEarlierBySourceLoc(unanalyzableOp, other.unanalyzableOp);
  if (joined == domain && carriedOp == unanalyzableOp) {
    return ChangeResult::NoChange;
  }
  domain = std::move(joined);
  unanalyzableOp = carriedOp;
  return ChangeResult::Change;
}

ChangeResult LaunchNodeDomainLattice::setDomain(LaunchNodeDomain newDomain,
                                                Operation *newUnanalyzableOp) {
  if (newDomain == domain && newUnanalyzableOp == unanalyzableOp) {
    return ChangeResult::NoChange;
  }
  domain = std::move(newDomain);
  unanalyzableOp = newUnanalyzableOp;
  return ChangeResult::Change;
}

void LaunchNodeDomainLattice::print(raw_ostream &os) const {
  if (!domain.known) {
    os << "<unknown>";
    if (const std::set<LaunchNodeCoord> *bound = domain.getUpperBoundNodes()) {
      os << " within {";
      llvm::interleaveComma(*bound, os, [&](LaunchNodeCoord coord) {
        os << "(" << coord.x << "," << coord.y << ")";
      });
      os << "}";
    }
    return;
  }
  os << "{";
  bool first = true;
  for (LaunchNodeCoord coord : domain.nodes) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << "(" << coord.x << "," << coord.y << ")";
  }
  os << "}";
}

const LaunchNodeDomain &LaunchNodeDomainLattice::getDomain() const {
  return domain;
}

Operation *LaunchNodeDomainLattice::getUnanalyzableOp() const {
  return unanalyzableOp;
}

LaunchNodeDomainAnalysis::LaunchNodeDomainAnalysis(
    DataFlowSolver &solver, LaunchNodeDomainState &state,
    LaunchNodeDomainAnalysisOptions options)
    : DenseForwardDataFlowAnalysis(solver), state(state),
      options(std::move(options)) {}

void LaunchNodeDomainAnalysis::setToEntryState(
    LaunchNodeDomainLattice *lattice) {
  propagateIfChanged(lattice, lattice->setDomain(state.baseDomain));
}

LogicalResult
LaunchNodeDomainAnalysis::visitOperation(Operation *op,
                                         const LaunchNodeDomainLattice &before,
                                         LaunchNodeDomainLattice *after) {
  ChangeResult result = after->join(before);
  if (options.operationCallback) {
    options.operationCallback(op, before.getDomain(),
                              before.getUnanalyzableOp());
  }
  propagateIfChanged(after, result);
  return success();
}

void LaunchNodeDomainAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const LaunchNodeDomainLattice &before,
    LaunchNodeDomainLattice *after) {
  auto defaultHandling = [&]() {
    AbstractDenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after);
  };
  if (!regionTo || regionFrom) {
    defaultHandling();
    return;
  }

  Operation *op = branch.getOperation();
  LaunchNodeDomain narrowed = before.getDomain();
  Operation *unanalyzableOp = before.getUnanalyzableOp();

  if (options.computeRegionDomain) {
    std::optional<LaunchNodeDomain> computedDomain =
        options.computeRegionDomain(op, *regionTo);
    if (computedDomain) {
      narrowed = before.getDomain().intersectWith(*computedDomain);
      ChangeResult result = after->setDomain(narrowed, unanalyzableOp);
      propagateIfChanged(after, result);
      return;
    }
  }

  TypeSwitch<Operation *>(op)
      .Case<scf::IfOp>([&](scf::IfOp ifOp) {
        BranchLaunchNodeDomains domains = getBranchLaunchNodeDomains(
            ifOp.getCondition(), before.getDomain(), state);
        unanalyzableOp =
            pickEarlierBySourceLoc(unanalyzableOp, domains.unanalyzableOp);
        narrowed = (*regionTo == 0) ? domains.thenDomain : domains.elseDomain;
      })
      .Case<affine::AffineIfOp>([&](affine::AffineIfOp ifOp) {
        LaunchNodeDomainResult condDomain =
            getAffineIfLaunchNodeDomain(ifOp, state.baseDomain);
        unanalyzableOp =
            pickEarlierBySourceLoc(unanalyzableOp, condDomain.unanalyzableOp);
        if (!condDomain.domain.known) {
          narrowed = LaunchNodeDomain::unknownWithin(before.getDomain());
        } else if (*regionTo == 0) {
          narrowed = before.getDomain().intersectWith(condDomain.domain);
        } else {
          narrowed = before.getDomain().intersectWith(
              state.baseDomain.subtract(condDomain.domain));
        }
      })
      .Case<IfSrcOp>([&](IfSrcOp ifSrc) {
        auto pipeType = mlir::cast<PipeType>(ifSrc.getPipe().getType());
        narrowed = before.getDomain().intersectWith(
            getPipeSourceLaunchNodeDomain(pipeType));
      })
      .Case<IfDstOp>([&](IfDstOp ifDst) {
        auto pipeType = mlir::cast<PipeType>(ifDst.getPipe().getType());
        narrowed = before.getDomain().intersectWith(
            getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain));
      })
      .Case<PipeNetForeachSrcOp>([&](PipeNetForeachSrcOp foreachSrc) {
        narrowed =
            before.getDomain().intersectWith(getPipeRecordsRoleLaunchNodeDomain(
                foreachSrc.getRecords(), PipeRole::Source));
      })
      .Case<PipeNetForeachDstOp>([&](PipeNetForeachDstOp foreachDst) {
        narrowed =
            before.getDomain().intersectWith(getPipeRecordsRoleLaunchNodeDomain(
                foreachDst.getRecords(), PipeRole::Destination));
      })
      .Case<PipeNetScopeOp>([&](PipeNetScopeOp scopeOp) {
        auto scope = getPipeNetScopeLaunchNodeDomains(
            scopeOp, state, options.emitInvalidPipeNetDiagnostics);
        if (!scope) {
          return;
        }
        if (options.pipeNetScopeCallback) {
          options.pipeNetScopeCallback(scopeOp, before.getDomain(),
                                       before.getUnanalyzableOp(), *scope);
        }
        if (options.narrowPipeNetScopes) {
          narrowed = before.getDomain().intersectWith(scope->domain);
        }
      })
      .Default([&](auto) {});

  ChangeResult result = after->setDomain(narrowed, unanalyzableOp);
  propagateIfChanged(after, result);
}

} // namespace mlir::tt::ttl
