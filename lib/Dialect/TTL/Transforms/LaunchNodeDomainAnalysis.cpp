// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Launch Node Domain Analysis
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "ttlang/Analysis/ExecutionCountAnalysis.h"
#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
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

LaunchNodeDomain LaunchNodeDomainState::getRoleDomain(int64_t netId,
                                                      PipeRole role) const {
  assert(pipeNetLocs.contains(netId) && "PipeNet id must be declared");
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

void LaunchNodeDomainState::initialize(ModuleOp module) {
  executionCountAnalysesByFunctionAndCoord.clear();
  module.walk([&](CreatePipeOp pipe) {
    PipeType pipeType = mlir::cast<PipeType>(pipe.getResult().getType());
    int64_t pipeNetId = pipeType.getPipeNetId();
    netSourceDomains[pipeNetId] = netSourceDomains[pipeNetId].unionWith(
        getPipeSourceLaunchNodeDomain(pipeType));
    netDestinationDomains[pipeNetId] =
        netDestinationDomains[pipeNetId].unionWith(
            getPipeDestinationLaunchNodeDomain(pipeType));
    pipeNetLocs[pipeNetId].push_back(pipe.getLoc());
    auto &name = pipeNetNames[pipeNetId];
    if (name.empty()) {
      if (auto attr = pipe.getPipeNetNameAttr()) {
        name = attr.getValue().str();
      }
    }
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

std::optional<llvm::APInt>
evaluateLaunchNodeContextValue(Value value, LaunchNodeCoord coord,
                               const LaunchNodeDomainState *state) {
  if (value.getDefiningOp<CoreXOp>()) {
    return llvm::APInt(IndexType::kInternalStorageBitWidth, coord.x);
  }
  if (value.getDefiningOp<CoreYOp>()) {
    return llvm::APInt(IndexType::kInternalStorageBitWidth, coord.y);
  }
  if (state) {
    if (auto predicate = value.getDefiningOp<PipeNetPredicateOpInterface>()) {
      bool selected = knownLaunchNodeDomainContains(
          state->getRoleDomain(predicate.getReferencedPipeNetId(),
                               predicate.getReferencedRole()),
          coord);
      return llvm::APInt(/*numBits=*/1, selected);
    }
  }
  return std::nullopt;
}

/// Use shared integer folding for every launch-domain expression.
static IntegerExpressionEvaluator
createLaunchNodeIntegerEvaluator(LaunchNodeCoord coord,
                                 const LaunchNodeDomainState *state = nullptr) {
  return IntegerExpressionEvaluator(
      [coord, state](Value value) -> std::optional<llvm::APInt> {
        return evaluateLaunchNodeContextValue(value, coord, state);
      });
}

std::optional<bool>
evaluatePredicateAtLaunchNode(Value value, LaunchNodeCoord coord,
                              const LaunchNodeDomainState &state) {
  std::optional<llvm::APInt> maybeValue =
      createLaunchNodeIntegerEvaluator(coord, &state).evaluate(value);
  if (!maybeValue || maybeValue->getBitWidth() != 1) {
    return std::nullopt;
  }
  return maybeValue->getBoolValue();
}

namespace {

static std::optional<std::uint64_t>
getRegionInvocationCountAtLaunchNode(Region &region, LaunchNodeCoord coord,
                                     const LaunchNodeDomainState &state) {
  Operation *parent = region.getParentOp();
  if (isa<PipeNetScopeOp>(parent)) {
    return 1;
  }
  if (auto ifSrcOp = dyn_cast<IfSrcOp>(parent)) {
    auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
               getPipeSourceLaunchNodeDomain(pipeType), coord)
               ? 1
               : 0;
  }
  if (auto ifDstOp = dyn_cast<IfDstOp>(parent)) {
    auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
    return knownLaunchNodeDomainContains(
               getPipeDestinationLaunchNodeDomain(pipeType), coord)
               ? 1
               : 0;
  }
  if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(parent)) {
    LaunchNodeDomainResult trueDomain =
        getAffineIfLaunchNodeDomain(affineIfOp, state.baseDomain);
    if (!trueDomain.domain.known || region.getRegionNumber() > 1) {
      return std::nullopt;
    }
    bool selectsThen = knownLaunchNodeDomainContains(trueDomain.domain, coord);
    return (selectsThen == (region.getRegionNumber() == 0)) ? 1 : 0;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::uint64_t>
getExactExecutionCountAtLaunchNode(Operation *op, LaunchNodeCoord coord,
                                   const LaunchNodeDomainState &state) {
  func::FuncOp function = op->getParentOfType<func::FuncOp>();
  if (!function) {
    return std::nullopt;
  }
  auto &analysesByCoord =
      state.executionCountAnalysesByFunctionAndCoord[function.getOperation()];
  auto analysisIt = analysesByCoord.find(coord);
  if (analysisIt == analysesByCoord.end()) {
    auto analysis = std::make_unique<ExecutionCountAnalysis>(
        function.getBody(),
        [coord, &state](Value value) {
          return evaluateLaunchNodeContextValue(value, coord, &state);
        },
        [coord, &state](Region &region) {
          return getRegionInvocationCountAtLaunchNode(region, coord, state);
        });
    analysisIt = analysesByCoord.emplace(coord, std::move(analysis)).first;
  }
  return analysisIt->second->getExecutionCount(op);
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
    if (mlir::isa<CoreXOp, CoreYOp>(op)) {
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

namespace {

/// Structured-control frames and values that determine an unresolved count.
struct UnresolvedExecutionCountContext {
  func::FuncOp function;
  SmallVector<std::pair<Operation *, std::size_t>> frames;
  SmallVector<Value> controlValues;

  bool operator==(const UnresolvedExecutionCountContext &rhs) const {
    return function == rhs.function && frames == rhs.frames &&
           controlValues == rhs.controlValues;
  }
};

static std::optional<UnresolvedExecutionCountContext>
getUnresolvedExecutionCountContext(Operation *op, LaunchNodeCoord coord,
                                   const LaunchNodeDomainState &state) {
  UnresolvedExecutionCountContext context;
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

    Region *region = block->getParent();
    if (isa<PipeNetScopeOp>(parent)) {
      current = parent;
      continue;
    }
    if (auto ifSrcOp = dyn_cast<IfSrcOp>(parent)) {
      auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
      if (!knownLaunchNodeDomainContains(
              getPipeSourceLaunchNodeDomain(pipeType), coord)) {
        return std::nullopt;
      }
      current = parent;
      continue;
    }
    if (auto ifDstOp = dyn_cast<IfDstOp>(parent)) {
      auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
      if (!knownLaunchNodeDomainContains(
              getPipeDestinationLaunchNodeDomain(pipeType), coord)) {
        return std::nullopt;
      }
      current = parent;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      std::optional<bool> maybeSelected =
          evaluatePredicateAtLaunchNode(ifOp.getCondition(), coord, state);
      if (maybeSelected) {
        if (region->getRegionNumber() != (*maybeSelected ? 0 : 1)) {
          return std::nullopt;
        }
        current = parent;
        continue;
      }
      context.frames.push_back({parent, region->getRegionNumber()});
      context.controlValues.push_back(ifOp.getCondition());
    } else if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(parent);
               affineIfOp && state.hasLaunchGrid) {
      LaunchNodeDomainResult trueDomain =
          getAffineIfLaunchNodeDomain(affineIfOp, state.baseDomain);
      if (trueDomain.domain.known) {
        bool selectsThen =
            knownLaunchNodeDomainContains(trueDomain.domain, coord);
        if (region->getRegionNumber() != (selectsThen ? 0 : 1)) {
          return std::nullopt;
        }
        current = parent;
        continue;
      }
      context.frames.push_back({parent, region->getRegionNumber()});
      llvm::append_range(context.controlValues, affineIfOp.getOperands());
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      context.frames.push_back({parent, region->getRegionNumber()});
      context.controlValues.push_back(forOp.getLowerBound());
      context.controlValues.push_back(forOp.getUpperBound());
      context.controlValues.push_back(forOp.getStep());
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

/// Return true when one SSA value has the same runtime value at two launch
/// nodes. Function arguments are common runtime arguments. Pure expressions
/// preserve equality; coordinate queries, memory reads, and unsupported block
/// arguments require a concrete evaluation.
static bool proveEqualValueAtLaunchNodes(Value value, LaunchNodeCoord lhsCoord,
                                         LaunchNodeCoord rhsCoord,
                                         const LaunchNodeDomainState &state,
                                         llvm::DenseMap<Value, bool> &cache) {
  if (lhsCoord == rhsCoord) {
    return true;
  }
  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }

  std::optional<llvm::APInt> maybeLhsValue =
      createLaunchNodeIntegerEvaluator(lhsCoord, &state).evaluate(value);
  std::optional<llvm::APInt> maybeRhsValue =
      createLaunchNodeIntegerEvaluator(rhsCoord, &state).evaluate(value);
  if (maybeLhsValue && maybeRhsValue) {
    bool equal = *maybeLhsValue == *maybeRhsValue;
    cache[value] = equal;
    return equal;
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *owner = argument.getOwner();
    auto function = dyn_cast_if_present<func::FuncOp>(owner->getParentOp());
    bool equal = function && owner == &function.getBody().front();
    if (!equal) {
      if (auto forOp = dyn_cast_if_present<scf::ForOp>(owner->getParentOp());
          forOp && value == forOp.getInductionVar()) {
        equal = proveEqualValueAtLaunchNodes(forOp.getLowerBound(), lhsCoord,
                                             rhsCoord, state, cache) &&
                proveEqualValueAtLaunchNodes(forOp.getUpperBound(), lhsCoord,
                                             rhsCoord, state, cache) &&
                proveEqualValueAtLaunchNodes(forOp.getStep(), lhsCoord,
                                             rhsCoord, state, cache);
      }
    }
    cache[value] = equal;
    return equal;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getNumRegions() != 0 ||
      definingOp->getNumOperands() == 0 || !isMemoryEffectFree(definingOp)) {
    cache[value] = false;
    return false;
  }
  bool equal = llvm::all_of(definingOp->getOperands(), [&](Value operand) {
    return proveEqualValueAtLaunchNodes(operand, lhsCoord, rhsCoord, state,
                                        cache);
  });
  cache[value] = equal;
  return equal;
}

} // namespace

bool proveEqualExecutionCountAtLaunchNodes(Operation *lhs,
                                           LaunchNodeCoord lhsCoord,
                                           Operation *rhs,
                                           LaunchNodeCoord rhsCoord,
                                           const LaunchNodeDomainState &state) {
  std::optional<std::uint64_t> maybeLhsCount =
      getExactExecutionCountAtLaunchNode(lhs, lhsCoord, state);
  std::optional<std::uint64_t> maybeRhsCount =
      getExactExecutionCountAtLaunchNode(rhs, rhsCoord, state);
  if (maybeLhsCount && maybeRhsCount) {
    return maybeLhsCount == maybeRhsCount;
  }
  std::optional<UnresolvedExecutionCountContext> maybeLhsContext =
      getUnresolvedExecutionCountContext(lhs, lhsCoord, state);
  std::optional<UnresolvedExecutionCountContext> maybeRhsContext =
      getUnresolvedExecutionCountContext(rhs, rhsCoord, state);
  if (!maybeLhsContext || !maybeRhsContext ||
      !(*maybeLhsContext == *maybeRhsContext)) {
    return false;
  }
  llvm::DenseMap<Value, bool> equalValueCache;
  return llvm::all_of(maybeLhsContext->controlValues, [&](Value value) {
    return proveEqualValueAtLaunchNodes(value, lhsCoord, rhsCoord, state,
                                        equalValueCache);
  });
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
      return {LaunchNodeDomain::unknown(), LaunchNodeDomain::unknown(),
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
                                 LaunchNodeDomainState &state) {
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
        narrowed = before.getDomain().intersectWith(
            getPipeSourceLaunchNodeDomain(pipeType));
      })
      .Case<IfDstOp>([&](IfDstOp ifDst) {
        auto pipeType = mlir::cast<PipeType>(ifDst.getPipe().getType());
        narrowed = before.getDomain().intersectWith(
            getPipeDestinationLaunchNodeDomain(pipeType));
      })
      .Case<PipeNetScopeOp>([&](PipeNetScopeOp scopeOp) {
        auto scope = getPipeNetScopeLaunchNodeDomains(scopeOp, state);
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
