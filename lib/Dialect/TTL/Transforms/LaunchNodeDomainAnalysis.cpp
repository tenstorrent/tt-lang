// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Launch Node Domain Analysis
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

bool LaunchNodeCoord::operator<(const LaunchNodeCoord &rhs) const {
  return std::tie(x, y) < std::tie(rhs.x, rhs.y);
}

bool LaunchNodeCoord::operator==(const LaunchNodeCoord &rhs) const {
  return x == rhs.x && y == rhs.y;
}

LaunchNodeDomain LaunchNodeDomain::unknown() { return {/*known=*/false, {}}; }

bool LaunchNodeDomain::isSubsetOf(const LaunchNodeDomain &rhs) const {
  if (!known || !rhs.known) {
    return false;
  }
  return std::includes(rhs.nodes.begin(), rhs.nodes.end(), nodes.begin(),
                       nodes.end());
}

LaunchNodeDomain
LaunchNodeDomain::unionWith(const LaunchNodeDomain &rhs) const {
  if (!known || !rhs.known) {
    return LaunchNodeDomain::unknown();
  }
  LaunchNodeDomain result;
  std::set_union(nodes.begin(), nodes.end(), rhs.nodes.begin(), rhs.nodes.end(),
                 std::inserter(result.nodes, result.nodes.end()));
  return result;
}

LaunchNodeDomain
LaunchNodeDomain::intersectWith(const LaunchNodeDomain &rhs) const {
  if (!known || !rhs.known) {
    return LaunchNodeDomain::unknown();
  }
  LaunchNodeDomain result;
  std::set_intersection(nodes.begin(), nodes.end(), rhs.nodes.begin(),
                        rhs.nodes.end(),
                        std::inserter(result.nodes, result.nodes.end()));
  return result;
}

LaunchNodeDomain LaunchNodeDomain::subtract(const LaunchNodeDomain &rhs) const {
  if (!known || !rhs.known) {
    return LaunchNodeDomain::unknown();
  }
  LaunchNodeDomain result;
  std::set_difference(nodes.begin(), nodes.end(), rhs.nodes.begin(),
                      rhs.nodes.end(),
                      std::inserter(result.nodes, result.nodes.end()));
  return result;
}

bool LaunchNodeDomain::operator==(const LaunchNodeDomain &rhs) const {
  return known == rhs.known && nodes == rhs.nodes;
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

LaunchNodeDomain getPipeDestinationLaunchNodeDomain(PipeType pipeType) {
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
  if (!lhs.known || !rhs.known) {
    return true;
  }
  return !lhs.intersectWith(rhs).nodes.empty();
}

bool knownLaunchNodeDomainContains(const LaunchNodeDomain &domain,
                                   LaunchNodeCoord coord) {
  return domain.known && domain.nodes.find(coord) != domain.nodes.end();
}

LaunchNodeDomain getPipeRecordSourceLaunchNodeDomain(PipeRecordAttr record) {
  LaunchNodeDomain result;
  result.nodes.insert({record.getSrcX(), record.getSrcY()});
  return result;
}

LaunchNodeDomain
getPipeRecordDestinationLaunchNodeDomain(PipeRecordAttr record) {
  LaunchNodeDomain result;
  for (int64_t x = record.getDstStartX(); x <= record.getDstEndX(); ++x) {
    for (int64_t y = record.getDstStartY(); y <= record.getDstEndY(); ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

LaunchNodeDomain getPipeRecordsRoleLaunchNodeDomain(PipeNetRecordsAttr records,
                                                    PipeRole role) {
  LaunchNodeDomain result;
  for (PipeRecordAttr record : records.getPipes()) {
    LaunchNodeDomain recordDomain =
        role == PipeRole::Source
            ? getPipeRecordSourceLaunchNodeDomain(record)
            : getPipeRecordDestinationLaunchNodeDomain(record);
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

LaunchNodeDomain
LaunchNodeDomainState::getRoleDomain(int64_t netId, PipeRole role,
                                     DeviceRefAttr device) const {
  auto getRoleForDevice = [&](PipeRole requestedRole) {
    const auto &domainsByNet = requestedRole == PipeRole::Source
                                   ? netSourceDomainsByDevice
                                   : netDestinationDomainsByDevice;
    const auto &localDomains = requestedRole == PipeRole::Source
                                   ? localNetSourceDomains
                                   : localNetDestinationDomains;
    LaunchNodeDomain result;
    if (auto domainIt = localDomains.find(netId);
        domainIt != localDomains.end()) {
      result = result.unionWith(domainIt->second);
    }
    auto netIt = domainsByNet.find(netId);
    if (netIt == domainsByNet.end()) {
      return result;
    }
    auto deviceIt = netIt->second.find(device);
    if (deviceIt != netIt->second.end()) {
      result = result.unionWith(deviceIt->second);
    }
    return result;
  };

  if (device) {
    if (role == PipeRole::Source || role == PipeRole::Destination) {
      return getRoleForDevice(role);
    }
    return getRoleForDevice(PipeRole::Source)
        .unionWith(getRoleForDevice(PipeRole::Destination));
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

void LaunchNodeDomainState::recordPipeNet(PipeType pipeType,
                                          DeviceTransferAttr transfer,
                                          Location loc,
                                          std::optional<StringRef> name) {
  int64_t pipeNetId = pipeType.getPipeNetId();
  LaunchNodeDomain sourceDomain = getPipeSourceLaunchNodeDomain(pipeType);
  LaunchNodeDomain destinationDomain =
      getPipeDestinationLaunchNodeDomain(pipeType);
  netSourceDomains[pipeNetId] =
      netSourceDomains[pipeNetId].unionWith(sourceDomain);
  netDestinationDomains[pipeNetId] =
      netDestinationDomains[pipeNetId].unionWith(destinationDomain);
  if (!transfer) {
    localNetSourceDomains[pipeNetId] =
        localNetSourceDomains[pipeNetId].unionWith(sourceDomain);
    localNetDestinationDomains[pipeNetId] =
        localNetDestinationDomains[pipeNetId].unionWith(destinationDomain);
  } else {
    TransferEdgeAttr edge = transfer.getEdge();
    DeviceRefAttr sourceDevice = edge.getSource();
    netSourceDomainsByDevice[pipeNetId][sourceDevice] =
        netSourceDomainsByDevice[pipeNetId][sourceDevice].unionWith(
            sourceDomain);
    if (DeviceRefAttr destinationDevice = edge.getDestination()) {
      netDestinationDomainsByDevice[pipeNetId][destinationDevice] =
          netDestinationDomainsByDevice[pipeNetId][destinationDevice].unionWith(
              destinationDomain);
    } else {
      DeviceRangeAttr destinationRange = edge.getDestinationRange();
      for (DeviceRefAttr destinationDevice :
           enumerateDeviceDomain(transfer.getDomain())) {
        if (!deviceRangeContains(destinationRange, destinationDevice)) {
          continue;
        }
        netDestinationDomainsByDevice[pipeNetId][destinationDevice] =
            netDestinationDomainsByDevice[pipeNetId][destinationDevice]
                .unionWith(destinationDomain);
      }
    }
  }
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
    LaunchNodeDomain sourceDomain = getPipeRecordSourceLaunchNodeDomain(record);
    LaunchNodeDomain destinationDomain =
        getPipeRecordDestinationLaunchNodeDomain(record);
    netSourceDomains[pipeNetId] =
        netSourceDomains[pipeNetId].unionWith(sourceDomain);
    netDestinationDomains[pipeNetId] =
        netDestinationDomains[pipeNetId].unionWith(destinationDomain);
    localNetSourceDomains[pipeNetId] =
        localNetSourceDomains[pipeNetId].unionWith(sourceDomain);
    localNetDestinationDomains[pipeNetId] =
        localNetDestinationDomains[pipeNetId].unionWith(destinationDomain);
    pipeNetLocs[pipeNetId].push_back(loc);
  }
  auto &storedName = pipeNetNames[pipeNetId];
  if (storedName.empty() && name && !name->empty()) {
    storedName = name->str();
  }
}

void LaunchNodeDomainState::initialize(ModuleOp module) {
  module.walk([&](CreatePipeOp pipe) {
    std::optional<StringRef> name;
    if (auto attr = pipe.getPipeNetNameAttr()) {
      name = attr.getValue();
    }
    recordPipeNet(mlir::cast<PipeType>(pipe.getResult().getType()),
                  pipe.getDeviceTransferAttr(), pipe.getLoc(), name);
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

  if (!module->hasAttr(kLaunchGridAttrName)) {
    hasLaunchGrid = false;
    return;
  }

  SmallVector<int64_t> launchGrid;
  if (!readI64ArrayAttr(module.getOperation(), kLaunchGridAttrName,
                        launchGrid) ||
      launchGrid.size() != 2 || launchGrid[0] <= 0 || launchGrid[1] <= 0) {
    hasLaunchGrid = false;
    return;
  }
  hasLaunchGrid = true;
  baseDomain = getFullLaunchNodeDomain(launchGrid[0], launchGrid[1]);
}

/// Evaluate index expressions that are affine over `ttl.core_x`,
/// `ttl.core_y`, and integer constants for one launch coordinate.
static std::optional<int64_t> evalIndex(Value value, LaunchNodeCoord coord,
                                        DeviceRefAttr currentDevice) {
  if (value.getDefiningOp<CoreXOp>()) {
    return coord.x;
  }
  if (value.getDefiningOp<CoreYOp>()) {
    return coord.y;
  }
  if (auto constant = getConstantIntValue(value)) {
    return *constant;
  }
  if (auto currentDeviceIndex = value.getDefiningOp<CurrentDeviceIndexOp>()) {
    if (!currentDevice) {
      return std::nullopt;
    }
    return getDeviceLinearIndex(currentDeviceIndex.getDomain(), currentDevice);
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
    return evalIndex(castOp.getIn(), coord, currentDevice);
  }
  if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = evalIndex(addOp.getLhs(), coord, currentDevice);
    auto rhs = evalIndex(addOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs + *rhs;
    }
  }
  if (auto subOp = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = evalIndex(subOp.getLhs(), coord, currentDevice);
    auto rhs = evalIndex(subOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs - *rhs;
    }
  }
  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = evalIndex(mulOp.getLhs(), coord, currentDevice);
    auto rhs = evalIndex(mulOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs * *rhs;
    }
  }
  return std::nullopt;
}

/// Evaluate boolean predicates composed from statically evaluable comparisons
/// and integer boolean operations for one launch coordinate.
static std::optional<bool> evalBool(Value value, LaunchNodeCoord coord,
                                    DeviceRefAttr currentDevice) {
  if (value.getType().isInteger(1)) {
    if (auto constant = getConstantIntValue(value)) {
      return *constant != 0;
    }
  }
  if (auto isDevice = value.getDefiningOp<IsDeviceOp>()) {
    return currentDevice
               ? std::optional<bool>(currentDevice == isDevice.getDevice())
               : std::nullopt;
  }
  if (auto isDeviceInRange = value.getDefiningOp<IsDeviceInRangeOp>()) {
    return currentDevice ? std::optional<bool>(deviceRangeContains(
                               isDeviceInRange.getRange(), currentDevice))
                         : std::nullopt;
  }
  if (auto cmpOp = value.getDefiningOp<arith::CmpIOp>()) {
    auto lhs = evalIndex(cmpOp.getLhs(), coord, currentDevice);
    auto rhs = evalIndex(cmpOp.getRhs(), coord, currentDevice);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return *lhs == *rhs;
    case arith::CmpIPredicate::ne:
      return *lhs != *rhs;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return *lhs < *rhs;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return *lhs <= *rhs;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return *lhs > *rhs;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return *lhs >= *rhs;
    }
  }
  if (auto andOp = value.getDefiningOp<arith::AndIOp>()) {
    auto lhs = evalBool(andOp.getLhs(), coord, currentDevice);
    auto rhs = evalBool(andOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs && *rhs;
    }
  }
  if (auto orOp = value.getDefiningOp<arith::OrIOp>()) {
    auto lhs = evalBool(orOp.getLhs(), coord, currentDevice);
    auto rhs = evalBool(orOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs || *rhs;
    }
  }
  if (auto xorOp = value.getDefiningOp<arith::XOrIOp>()) {
    auto lhs = evalBool(xorOp.getLhs(), coord, currentDevice);
    auto rhs = evalBool(xorOp.getRhs(), coord, currentDevice);
    if (lhs && rhs) {
      return *lhs != *rhs;
    }
  }
  return std::nullopt;
}

/// Return true if evaluating `value` can depend on the current launch
/// coordinate.
static bool dependsOnExecutionInstance(Value value, DeviceRefAttr currentDevice,
                                       llvm::DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }
  Operation *op = value.getDefiningOp();
  bool result = false;
  if (op) {
    if (mlir::isa<CoreXOp, CoreYOp>(op) ||
        (currentDevice &&
         mlir::isa<IsDeviceOp, IsDeviceInRangeOp, CurrentDeviceIndexOp>(op))) {
      result = true;
    } else {
      for (Value operand : op->getOperands()) {
        if (dependsOnExecutionInstance(operand, currentDevice, cache)) {
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
                            const LaunchNodeDomain &baseDomain,
                            DeviceRefAttr currentDevice) {
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
    bool resolved = true;
    for (unsigned idx = 0; idx < set.getNumInputs(); ++idx) {
      auto value = evalIndex(operands[idx], coord, currentDevice);
      if (!value) {
        resolved = false;
        break;
      }
      operandConstants[idx] = IntegerAttr::get(IndexType::get(ctx), *value);
    }
    if (!resolved) {
      return {LaunchNodeDomain::unknown(), ifOp};
    }
    SmallVector<Attribute> folded;
    if (failed(map.constantFold(operandConstants, folded))) {
      return {LaunchNodeDomain::unknown(), ifOp};
    }
    bool ok = true;
    for (unsigned idx = 0; idx < set.getNumConstraints(); ++idx) {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(folded[idx]);
      if (!intAttr) {
        return {LaunchNodeDomain::unknown(), ifOp};
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
                     DeviceRefAttr currentDevice,
                     llvm::DenseMap<Value, bool> &executionInstanceCache) {
  if (auto pred = condition.getDefiningOp<PipeNetPredicateOpInterface>()) {
    LaunchNodeDomain roleDomain = state.getRoleDomain(
        pred.getReferencedPipeNetId(), pred.getReferencedRole(), currentDevice);
    return exactBranches(roleDomain, current, state.baseDomain);
  }
  if (auto andOp = condition.getDefiningOp<arith::AndIOp>()) {
    BranchLaunchNodeDomains lhs = getBranchDomainsImpl(
        andOp.getLhs(), current, state, currentDevice, executionInstanceCache);
    BranchLaunchNodeDomains rhs = getBranchDomainsImpl(
        andOp.getRhs(), current, state, currentDevice, executionInstanceCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(lhs.unanalyzableOp, rhs.unanalyzableOp);
    return {
        lhs.thenDomain.intersectWith(rhs.thenDomain),
        lhs.elseDomain.unionWith(lhs.thenDomain.intersectWith(rhs.elseDomain)),
        unanalyzable};
  }
  if (auto orOp = condition.getDefiningOp<arith::OrIOp>()) {
    BranchLaunchNodeDomains lhs = getBranchDomainsImpl(
        orOp.getLhs(), current, state, currentDevice, executionInstanceCache);
    BranchLaunchNodeDomains rhs = getBranchDomainsImpl(
        orOp.getRhs(), current, state, currentDevice, executionInstanceCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(lhs.unanalyzableOp, rhs.unanalyzableOp);
    return {
        lhs.thenDomain.unionWith(lhs.elseDomain.intersectWith(rhs.thenDomain)),
        lhs.elseDomain.intersectWith(rhs.elseDomain), unanalyzable};
  }
  if (!dependsOnExecutionInstance(condition, currentDevice,
                                  executionInstanceCache)) {
    return {current, current};
  }
  LaunchNodeDomain trueDomain;
  for (LaunchNodeCoord coord : state.baseDomain.nodes) {
    std::optional<bool> value = evalBool(condition, coord, currentDevice);
    if (!value) {
      return {LaunchNodeDomain::unknown(), LaunchNodeDomain::unknown(),
              condition.getDefiningOp()};
    }
    if (*value) {
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
                           const LaunchNodeDomainState &state,
                           DeviceRefAttr currentDevice) {
  llvm::DenseMap<Value, bool> executionInstanceCache;
  return getBranchDomainsImpl(condition, current, state, currentDevice,
                              executionInstanceCache);
}

/// Decode the PipeNet role metadata carried by one `ttl.pipenet_scope`.
static std::optional<PipeNetScopeLaunchNodeDomains>
getPipeNetScopeLaunchNodeDomains(PipeNetScopeOp scopeOp,
                                 LaunchNodeDomainState &state,
                                 DeviceRefAttr currentDevice) {
  SmallVector<int64_t> ids;
  SmallVector<int64_t> roles;
  if (!readI64ArrayAttr(scopeOp.getOperation(), kPipeNetIdsAttrName, ids) ||
      !readI64ArrayAttr(scopeOp.getOperation(), kPipeNetRolesAttrName, roles)) {
    scopeOp.emitOpError() << "requires `" << kPipeNetIdsAttrName << "` and `"
                          << kPipeNetRolesAttrName << "` attributes";
    state.sawError = true;
    return std::nullopt;
  }
  if (ids.size() != roles.size()) {
    scopeOp.emitOpError() << "requires equal-length PipeNet id and role arrays";
    state.sawError = true;
    return std::nullopt;
  }
  PipeNetScopeLaunchNodeDomains result;
  for (auto [pipeNetId, roleValue] : llvm::zip_equal(ids, roles)) {
    if (roleValue != static_cast<int64_t>(PipeRole::Source) &&
        roleValue != static_cast<int64_t>(PipeRole::Destination)) {
      scopeOp.emitOpError() << "has invalid PipeNet role " << roleValue
                            << " (expected 0=src or 1=dst)";
      state.sawError = true;
      return std::nullopt;
    }
    auto role = static_cast<PipeRole>(roleValue);
    LaunchNodeDomain roleDomain =
        state.getRoleDomain(pipeNetId, role, currentDevice);
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

  TypeSwitch<Operation *>(op)
      .Case<scf::IfOp>([&](scf::IfOp ifOp) {
        BranchLaunchNodeDomains domains =
            getBranchLaunchNodeDomains(ifOp.getCondition(), before.getDomain(),
                                       state, options.currentDevice);
        unanalyzableOp =
            pickEarlierBySourceLoc(unanalyzableOp, domains.unanalyzableOp);
        narrowed = (*regionTo == 0) ? domains.thenDomain : domains.elseDomain;
      })
      .Case<affine::AffineIfOp>([&](affine::AffineIfOp ifOp) {
        LaunchNodeDomainResult condDomain = getAffineIfLaunchNodeDomain(
            ifOp, state.baseDomain, options.currentDevice);
        unanalyzableOp =
            pickEarlierBySourceLoc(unanalyzableOp, condDomain.unanalyzableOp);
        if (!condDomain.domain.known) {
          narrowed = LaunchNodeDomain::unknown();
        } else if (*regionTo == 0) {
          narrowed = before.getDomain().intersectWith(condDomain.domain);
        } else {
          narrowed = before.getDomain().intersectWith(
              state.baseDomain.subtract(condDomain.domain));
        }
      })
      .Case<IfSrcOp>([&](IfSrcOp ifSrc) {
        auto pipeType = mlir::cast<PipeType>(ifSrc.getPipe().getType());
        LaunchNodeDomain pipeDomain = getPipeSourceLaunchNodeDomain(pipeType);
        LaunchNodeDomain deviceRoleDomain = state.getRoleDomain(
            pipeType.getPipeNetId(), PipeRole::Source, options.currentDevice);
        narrowed = before.getDomain().intersectWith(
            pipeDomain.intersectWith(deviceRoleDomain));
      })
      .Case<IfDstOp>([&](IfDstOp ifDst) {
        auto pipeType = mlir::cast<PipeType>(ifDst.getPipe().getType());
        LaunchNodeDomain pipeDomain =
            getPipeDestinationLaunchNodeDomain(pipeType);
        LaunchNodeDomain deviceRoleDomain =
            state.getRoleDomain(pipeType.getPipeNetId(), PipeRole::Destination,
                                options.currentDevice);
        narrowed = before.getDomain().intersectWith(
            pipeDomain.intersectWith(deviceRoleDomain));
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
        auto scope = getPipeNetScopeLaunchNodeDomains(scopeOp, state,
                                                      options.currentDevice);
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
