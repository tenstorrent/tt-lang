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
#include "llvm/ADT/Twine.h"
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

void LaunchNodeDomainState::initialize(ModuleOp module) {
  executionCountAnalysesByFunctionAndCoord.clear();
  if (!module->hasAttr(kLaunchGridAttrName)) {
    hasLaunchGrid = false;
  } else {
    SmallVector<int64_t> launchGrid;
    if (!readI64ArrayAttr(module.getOperation(), kLaunchGridAttrName,
                          launchGrid) ||
        launchGrid.size() != 2 || launchGrid[0] <= 0 || launchGrid[1] <= 0) {
      hasLaunchGrid = false;
    } else {
      hasLaunchGrid = true;
      baseDomain = getFullLaunchNodeDomain(launchGrid[0], launchGrid[1]);
    }
  }

  module.walk([&](CreatePipeOp pipe) {
    PipeType pipeType = mlir::cast<PipeType>(pipe.getResult().getType());
    int64_t pipeNetId = pipeType.getPipeNetId();
    LaunchNodeDomain sourceDomain = getPipeSourceLaunchNodeDomain(pipeType);
    if (!sourceDomain.isSubsetOf(baseDomain)) {
      sourceDomain = LaunchNodeDomain::unknown();
    }
    netSourceDomains[pipeNetId] =
        netSourceDomains[pipeNetId].unionWith(sourceDomain);
    netDestinationDomains[pipeNetId] =
        netDestinationDomains[pipeNetId].unionWith(
            getPipeDestinationLaunchNodeDomain(pipeType, baseDomain));
    pipeNetLocs[pipeNetId].push_back(pipe.getLoc());
    auto &name = pipeNetNames[pipeNetId];
    if (name.empty()) {
      if (auto attr = pipe.getPipeNetNameAttr()) {
        name = attr.getValue().str();
      }
    }
  });
}

static std::optional<llvm::APInt>
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
               getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain),
               coord)
               ? 1
               : 0;
  }
  if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(parent)) {
    LaunchNodeDomainResult trueDomain =
        getAffineIfLaunchNodeDomain(affineIfOp, state.baseDomain);
    if (!trueDomain.domain.known) {
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
    if (mlir::isa<CoreXOp, CoreYOp, PipeNetPredicateOpInterface>(op)) {
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
              getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain),
              coord)) {
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

/// Return true when two SSA values have the same runtime value at their launch
/// nodes and active call sites.
static bool proveEqualValuesAtLaunchNodes(
    Value lhsValue, LaunchNodeCoord lhsCoord,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    Value rhsValue, LaunchNodeCoord rhsCoord,
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
      createLaunchNodeIntegerEvaluator(lhsCoord, &state).evaluate(lhsValue);
  std::optional<llvm::APInt> maybeRhsValue =
      createLaunchNodeIntegerEvaluator(rhsCoord, &state).evaluate(rhsValue);
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
      bool equal = proveEqualValuesAtLaunchNodes(
          *maybeLhsOperand, lhsCoord, resolveLhsFunctionArgument,
          *maybeRhsOperand, rhsCoord, resolveRhsFunctionArgument, state, cache);
      cache[cacheKey] = equal;
      return equal;
    }

    auto lhsForOp = dyn_cast_if_present<scf::ForOp>(lhsOwner->getParentOp());
    auto rhsForOp = dyn_cast_if_present<scf::ForOp>(rhsOwner->getParentOp());
    bool equal =
        lhsForOp && rhsForOp && lhsValue == lhsForOp.getInductionVar() &&
        rhsValue == rhsForOp.getInductionVar() &&
        proveEqualValuesAtLaunchNodes(
            lhsForOp.getLowerBound(), lhsCoord, resolveLhsFunctionArgument,
            rhsForOp.getLowerBound(), rhsCoord, resolveRhsFunctionArgument,
            state, cache) &&
        proveEqualValuesAtLaunchNodes(
            lhsForOp.getUpperBound(), lhsCoord, resolveLhsFunctionArgument,
            rhsForOp.getUpperBound(), rhsCoord, resolveRhsFunctionArgument,
            state, cache) &&
        proveEqualValuesAtLaunchNodes(lhsForOp.getStep(), lhsCoord,
                                      resolveLhsFunctionArgument,
                                      rhsForOp.getStep(), rhsCoord,
                                      resolveRhsFunctionArgument, state, cache);
    cache[cacheKey] = equal;
    return equal;
  }

  Operation *lhsDefiningOp = lhsValue.getDefiningOp();
  Operation *rhsDefiningOp = rhsValue.getDefiningOp();
  if (!lhsDefiningOp || lhsDefiningOp != rhsDefiningOp ||
      lhsDefiningOp->getNumRegions() != 0 ||
      lhsDefiningOp->getNumOperands() == 0 ||
      !isMemoryEffectFree(lhsDefiningOp)) {
    return false;
  }
  auto lhsResult = dyn_cast<OpResult>(lhsValue);
  auto rhsResult = dyn_cast<OpResult>(rhsValue);
  if (!lhsResult || !rhsResult ||
      lhsResult.getResultNumber() != rhsResult.getResultNumber()) {
    return false;
  }
  bool equal = llvm::all_of(lhsDefiningOp->getOperands(), [&](Value operand) {
    return proveEqualValuesAtLaunchNodes(
        operand, lhsCoord, resolveLhsFunctionArgument, operand, rhsCoord,
        resolveRhsFunctionArgument, state, cache);
  });
  cache[cacheKey] = equal;
  return equal;
}

} // namespace

bool proveEqualUnresolvedExecutionCountAtLaunchNodes(
    Operation *lhs, LaunchNodeCoord lhsCoord, Operation *rhs,
    LaunchNodeCoord rhsCoord, const LaunchNodeDomainState &state,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveLhsFunctionArgument,
    llvm::function_ref<std::optional<Value>(BlockArgument)>
        resolveRhsFunctionArgument) {
  std::optional<UnresolvedExecutionCountContext> maybeLhsContext =
      getUnresolvedExecutionCountContext(lhs, lhsCoord, state);
  std::optional<UnresolvedExecutionCountContext> maybeRhsContext =
      getUnresolvedExecutionCountContext(rhs, rhsCoord, state);
  if (!maybeLhsContext || !maybeRhsContext ||
      !(*maybeLhsContext == *maybeRhsContext)) {
    return false;
  }
  llvm::DenseMap<std::pair<Value, Value>, bool> equalValueCache;
  return llvm::all_of(maybeLhsContext->controlValues, [&](Value value) {
    return proveEqualValuesAtLaunchNodes(
        value, lhsCoord, resolveLhsFunctionArgument, value, rhsCoord,
        resolveRhsFunctionArgument, state, equalValueCache);
  });
}

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
    return *maybeLhsCount == *maybeRhsCount;
  }
  auto resolveNoFunctionArguments = [](BlockArgument) -> std::optional<Value> {
    return std::nullopt;
  };
  return proveEqualUnresolvedExecutionCountAtLaunchNodes(
      lhs, lhsCoord, rhs, rhsCoord, state, resolveNoFunctionArguments,
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
            getPipeDestinationLaunchNodeDomain(pipeType, state.baseDomain));
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
