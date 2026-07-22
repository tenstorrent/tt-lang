// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ValueOriginAnalysis.h"

#include "ttlang/Analysis/LoopIterationUtils.h"

#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>

namespace mlir::tt {

namespace {

/// One SSA value or tensor element awaiting backward traversal.
struct TraversalState {
  Value value;
  Operation *access = nullptr;
  SmallVector<Value> indices;
  SmallVector<Operation *> crossedLoopBackedges;

  bool isTensorElement() const { return access != nullptr; }
};

bool operator==(const TraversalState &lhs, const TraversalState &rhs) {
  return lhs.value == rhs.value && lhs.access == rhs.access &&
         lhs.indices == rhs.indices &&
         lhs.crossedLoopBackedges == rhs.crossedLoopBackedges;
}

/// Relation between the elements read by an extract and written by an insert.
enum class IndexDomainRelation { Covered, Disjoint, Partial, Unknown };

using IndexTuple = SmallVector<std::int64_t>;

struct IndexTupleLess {
  /// Orders concrete index tuples for deterministic set operations.
  bool operator()(const IndexTuple &lhs, const IndexTuple &rhs) const {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end());
  }
};

using IndexTupleSet = std::set<IndexTuple, IndexTupleLess>;

/// A one-to-one unrealized cast associates one result with one input.
bool isOneToOneCast(UnrealizedConversionCastOp cast) {
  return cast.getInputs().size() == 1 && cast.getOutputs().size() == 1;
}

/// Whether a loop may execute zero, one, or multiple iterations.
struct LoopExecutionInfo {
  bool maySkip;
  bool mayExecute;
  bool mayRepeat;
};

/// Initial and yielded values associated with one loop-carried SSA value.
struct TiedLoopValues {
  OpOperand *initial;
  OpOperand *yielded;
};

/// Tracks work performed while enumerating one index-domain comparison.
struct EnumerationState {
  explicit EnumerationState(ValueOriginAnalysis::Options options)
      : loopIterations(options.maxEnumeratedLoopIterations),
        indexTuples(options.maxEnumeratedIndexTuples) {}

  EnumerationBudget loopIterations;
  EnumerationBudget indexTuples;
};

/// Classify loop execution using the interface's static trip count.
LoopExecutionInfo getLoopExecutionInfo(LoopLikeOpInterface loop) {
  std::optional<std::uint64_t> maybeTripCount = getLoopTripCount(loop);
  if (!maybeTripCount) {
    return {/*maySkip=*/true, /*mayExecute=*/true, /*mayRepeat=*/true};
  }
  if (*maybeTripCount == 0) {
    return {/*maySkip=*/true, /*mayExecute=*/false, /*mayRepeat=*/false};
  }
  return {/*maySkip=*/false, /*mayExecute=*/true,
          /*mayRepeat=*/*maybeTripCount != 1};
}

/// Return a loop-carried association only when every required range is
/// exposed. `scf.while` exposes region arguments but no LoopLike init operands,
/// so its associations are obtained from RegionBranchOpInterface instead.
std::optional<TiedLoopValues> getTiedLoopValues(LoopLikeOpInterface loop,
                                                Value value) {
  Block::BlockArgListType iteratedArguments = loop.getRegionIterArgs();
  MutableArrayRef<OpOperand> initialValues = loop.getInitsMutable();
  std::optional<MutableArrayRef<OpOperand>> maybeYieldedValues =
      loop.getYieldedValuesMutable();
  if (initialValues.size() != iteratedArguments.size() || !maybeYieldedValues ||
      maybeYieldedValues->size() != iteratedArguments.size()) {
    return std::nullopt;
  }

  std::size_t index;
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    auto iteratedArgument = llvm::find(iteratedArguments, blockArgument);
    if (iteratedArgument == iteratedArguments.end()) {
      return std::nullopt;
    }
    index = std::distance(iteratedArguments.begin(), iteratedArgument);
  } else {
    auto result = cast<OpResult>(value);
    std::optional<ResultRange> maybeResults = loop.getLoopResults();
    if (!maybeResults || maybeResults->size() != iteratedArguments.size()) {
      return std::nullopt;
    }
    auto loopResult = llvm::find(*maybeResults, result);
    if (loopResult == maybeResults->end()) {
      return std::nullopt;
    }
    index = std::distance(maybeResults->begin(), loopResult);
  }

  return TiedLoopValues{&initialValues[index], &(*maybeYieldedValues)[index]};
}

/// One control-flow predecessor value and the loop backedge crossed, if any.
struct PredecessorValue {
  Value value;
  Operation *crossedLoopBackedge = nullptr;
};

/// Add an empty backedge annotation to MLIR predecessor values.
std::optional<SmallVector<PredecessorValue>>
makePredecessorValues(std::optional<SmallVector<Value>> maybePredecessors) {
  if (!maybePredecessors) {
    return std::nullopt;
  }
  return llvm::map_to_vector(*maybePredecessors, [](Value predecessor) {
    return PredecessorValue{predecessor};
  });
}

/// Resolve the values associated with one region successor input.
std::optional<SmallVector<PredecessorValue>>
resolveRegionSuccessor(RegionBranchOpInterface branch,
                       RegionSuccessor successor, Value input) {
  ValueRange successorInputs = branch.getSuccessorInputs(successor);
  auto successorInput = llvm::find(successorInputs, input);
  if (successorInput == successorInputs.end()) {
    return std::nullopt;
  }

  int successorInputIndex =
      static_cast<int>(std::distance(successorInputs.begin(), successorInput));
  SmallVector<Value> predecessorValues;
  branch.getPredecessorValues(successor, successorInputIndex,
                              predecessorValues);
  return makePredecessorValues(std::move(predecessorValues));
}

/// Return the possible predecessors of `value` in structured or block control
/// flow.
///
/// A missing result means that no modeled association exists and `value` is an
/// origin. An empty result means that a modeled value has no live predecessor.
///
/// MLIR issue #175168 affects region entry arguments whose leading arguments
/// are not successor inputs. Querying the interface by successor-input index
/// avoids treating an `scf.for` induction variable as an iterated value.
std::optional<SmallVector<PredecessorValue>> getValuePredecessors(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(result.getOwner())) {
      if (std::optional<TiedLoopValues> tiedValues =
              getTiedLoopValues(loop, result)) {
        LoopExecutionInfo execution = getLoopExecutionInfo(loop);
        SmallVector<PredecessorValue> predecessors;
        if (execution.maySkip) {
          predecessors.push_back({tiedValues->initial->get()});
        }
        if (execution.mayExecute) {
          predecessors.push_back({tiedValues->yielded->get()});
        }
        return predecessors;
      }
    }

    auto branch = dyn_cast<RegionBranchOpInterface>(result.getOwner());
    if (!branch) {
      return makePredecessorValues(mlir::getControlFlowPredecessors(value));
    }

    return resolveRegionSuccessor(
        branch, RegionSuccessor(branch.getOperation()), result);
  }

  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument || !blockArgument.getOwner()->isEntryBlock()) {
    return makePredecessorValues(mlir::getControlFlowPredecessors(value));
  }

  if (auto loop = dyn_cast<LoopLikeOpInterface>(
          blockArgument.getOwner()->getParentOp())) {
    if (std::optional<TiedLoopValues> tiedValues =
            getTiedLoopValues(loop, blockArgument)) {
      LoopExecutionInfo execution = getLoopExecutionInfo(loop);
      SmallVector<PredecessorValue> predecessors;
      if (execution.mayExecute) {
        predecessors.push_back({tiedValues->initial->get()});
      }
      if (execution.mayRepeat) {
        predecessors.push_back(
            {tiedValues->yielded->get(), loop.getOperation()});
      }
      return predecessors;
    }
  }

  auto branch = dyn_cast<RegionBranchOpInterface>(
      blockArgument.getOwner()->getParentOp());
  if (!branch) {
    return makePredecessorValues(mlir::getControlFlowPredecessors(value));
  }

  return resolveRegionSuccessor(
      branch, RegionSuccessor(blockArgument.getParentRegion()), blockArgument);
}

/// Return enclosing loops from outermost to innermost.
SmallVector<LoopLikeOpInterface> getEnclosingLoops(Operation *operation) {
  SmallVector<LoopLikeOpInterface> loops;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
      loops.push_back(loop);
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

/// Evaluate an integer expression under the current loop-IV bindings.
std::optional<std::int64_t>
evaluateInteger(OpFoldResult expression,
                const LoopInductionBindings &bindings) {
  if (auto integer = dyn_cast<Attribute>(expression)) {
    auto integerAttr = dyn_cast<IntegerAttr>(integer);
    if (!integerAttr || !integerAttr.getValue().isSignedIntN(64)) {
      return std::nullopt;
    }
    return integerAttr.getInt();
  }

  std::optional<llvm::APInt> maybeValue =
      createLoopIntegerEvaluator(bindings).evaluate(cast<Value>(expression));
  if (!maybeValue || !maybeValue->isSignedIntN(64)) {
    return std::nullopt;
  }
  return maybeValue->getSExtValue();
}

/// Collect concrete index tuples over the selected enclosing loops.
LogicalResult
collectIndexTuples(ArrayRef<Value> indices, ArrayRef<LoopLikeOpInterface> loops,
                   std::size_t firstLoop, LoopInductionBindings &bindings,
                   EnumerationState &enumeration, IndexTupleSet &tuples) {
  return enumerateLoopNest(loops.drop_front(firstLoop), bindings,
                           enumeration.loopIterations,
                           [&](const LoopInductionBindings &currentBindings) {
                             if (!enumeration.indexTuples.tryConsume()) {
                               return failure();
                             }
                             IndexTuple tuple;
                             tuple.reserve(indices.size());
                             for (Value index : indices) {
                               std::optional<std::int64_t> maybeIndex =
                                   evaluateInteger(index, currentBindings);
                               if (!maybeIndex) {
                                 return failure();
                               }
                               tuple.push_back(*maybeIndex);
                             }
                             tuples.insert(std::move(tuple));
                             return success();
                           });
}

/// Return whether every tuple in `subset` is present in `superset`.
bool isSubset(const IndexTupleSet &subset, const IndexTupleSet &superset) {
  return std::includes(superset.begin(), superset.end(), subset.begin(),
                       subset.end(), IndexTupleLess());
}

/// Return whether two concrete index sets contain a common tuple.
bool intersects(const IndexTupleSet &lhs, const IndexTupleSet &rhs) {
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  IndexTupleLess less;
  while (lhsIt != lhs.end() && rhsIt != rhs.end()) {
    if (!less(*lhsIt, *rhsIt) && !less(*rhsIt, *lhsIt)) {
      return true;
    }
    if (less(*lhsIt, *rhsIt)) {
      ++lhsIt;
    } else {
      ++rhsIt;
    }
  }
  return false;
}

/// Compare read and write indices over their finite enclosing loop domains.
/// Shared enclosing loops remain correlated; independent loops contribute
/// separate iteration dimensions.
IndexDomainRelation
compareIndexDomains(Operation *read, ArrayRef<Value> readIndices,
                    Operation *write, ArrayRef<Value> writeIndices,
                    ArrayRef<Operation *> crossedLoopBackedges,
                    ValueOriginAnalysis::Options options) {
  if (readIndices.size() != writeIndices.size()) {
    return IndexDomainRelation::Unknown;
  }
  SmallVector<LoopLikeOpInterface> readLoops = getEnclosingLoops(read);
  SmallVector<LoopLikeOpInterface> writeLoops = getEnclosingLoops(write);
  if (llvm::any_of(crossedLoopBackedges, [&](Operation *loop) {
        return llvm::any_of(writeLoops, [&](LoopLikeOpInterface writeLoop) {
          return writeLoop.getOperation() == loop;
        });
      })) {
    return IndexDomainRelation::Unknown;
  }
  if (readIndices == writeIndices) {
    return IndexDomainRelation::Covered;
  }

  std::size_t commonLoopCount = 0;
  while (commonLoopCount < readLoops.size() &&
         commonLoopCount < writeLoops.size() &&
         readLoops[commonLoopCount].getOperation() ==
             writeLoops[commonLoopCount].getOperation()) {
    ++commonLoopCount;
  }

  bool foundRead = false;
  bool allReadsCovered = true;
  bool foundIntersection = false;
  EnumerationState enumeration(options);
  LoopInductionBindings bindings;
  LogicalResult comparisonResult = enumerateLoopNest(
      ArrayRef<LoopLikeOpInterface>(readLoops).take_front(commonLoopCount),
      bindings, enumeration.loopIterations, [&](const LoopInductionBindings &) {
        IndexTupleSet readTuples;
        IndexTupleSet writeTuples;
        if (failed(collectIndexTuples(readIndices, readLoops, commonLoopCount,
                                      bindings, enumeration, readTuples)) ||
            failed(collectIndexTuples(writeIndices, writeLoops, commonLoopCount,
                                      bindings, enumeration, writeTuples))) {
          return failure();
        }
        if (readTuples.empty()) {
          return success();
        }
        foundRead = true;
        allReadsCovered &= isSubset(readTuples, writeTuples);
        foundIntersection |= intersects(readTuples, writeTuples);
        return success();
      });

  if (failed(comparisonResult) || !foundRead) {
    return IndexDomainRelation::Unknown;
  }
  if (allReadsCovered) {
    return IndexDomainRelation::Covered;
  }
  return foundIntersection ? IndexDomainRelation::Partial
                           : IndexDomainRelation::Disjoint;
}

/// Tracks traversal states without conflating different indices of the same
/// tensor value.
class VisitedStates {
public:
  bool insert(const TraversalState &state) {
    if (!state.isTensorElement()) {
      return values.insert(state.value).second;
    }

    SmallVector<TraversalState> &states = elementStates[state.value];
    if (llvm::is_contained(states, state)) {
      return false;
    }
    states.push_back(state);
    return true;
  }

private:
  llvm::DenseSet<Value> values;
  llvm::DenseMap<Value, SmallVector<TraversalState>> elementStates;
};

} // namespace

/// Implements a conservative backward traversal from one SSA value.
class OriginWalker {
public:
  explicit OriginWalker(ValueOriginAnalysis::Options options)
      : options(options) {}

  SmallVector<Value> getOrigins(Value initialValue) {
    assert(initialValue && "value-origin analysis requires a value");

    SmallVector<TraversalState> worklist{{initialValue, nullptr, {}, {}}};
    VisitedStates visited;
    llvm::SetVector<Value> origins;
    while (!worklist.empty()) {
      TraversalState state = worklist.pop_back_val();
      if (!visited.insert(state)) {
        continue;
      }

      if (state.isTensorElement()) {
        visitTensorElement(state, worklist, origins);
      } else {
        visitValue(state, worklist, origins);
      }
    }
    return llvm::to_vector(origins);
  }

private:
  /// Continue from an SSA value through modeled value associations.
  void visitValue(const TraversalState &state,
                  SmallVectorImpl<TraversalState> &worklist,
                  llvm::SetVector<Value> &origins) const {
    if (auto cast = state.value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (isOneToOneCast(cast)) {
        worklist.push_back({cast.getInputs().front(), nullptr, {}, {}});
        return;
      }
    }
    if (auto extract = state.value.getDefiningOp<tensor::ExtractOp>()) {
      worklist.push_back({extract.getTensor(),
                          extract.getOperation(),
                          extract.getIndices(),
                          {}});
      return;
    }
    if (std::optional<SmallVector<PredecessorValue>> predecessors =
            getValuePredecessors(state.value)) {
      for (const PredecessorValue &predecessor : *predecessors) {
        worklist.push_back({predecessor.value, nullptr, {}, {}});
      }
      return;
    }
    origins.insert(state.value);
  }

  /// Visit one tensor element while preserving its access context.
  void visitTensorElement(const TraversalState &state,
                          SmallVectorImpl<TraversalState> &worklist,
                          llvm::SetVector<Value> &origins) const {
    if (auto cast = state.value.getDefiningOp<tensor::CastOp>()) {
      worklist.push_back({cast.getSource(), state.access, state.indices,
                          state.crossedLoopBackedges});
      return;
    }
    if (auto insert = state.value.getDefiningOp<tensor::InsertOp>()) {
      SmallVector<Value> insertIndices(insert.getIndices());
      IndexDomainRelation relation = compareIndexDomains(
          state.access, state.indices, insert.getOperation(), insertIndices,
          state.crossedLoopBackedges, options);
      if (relation != IndexDomainRelation::Disjoint) {
        worklist.push_back({insert.getScalar(), nullptr, {}, {}});
      }
      if (relation != IndexDomainRelation::Covered) {
        worklist.push_back({insert.getDest(), state.access, state.indices,
                            state.crossedLoopBackedges});
      }
      return;
    }
    if (std::optional<SmallVector<PredecessorValue>> predecessors =
            getValuePredecessors(state.value)) {
      for (const PredecessorValue &predecessor : *predecessors) {
        SmallVector<Operation *> crossedLoopBackedges =
            state.crossedLoopBackedges;
        if (predecessor.crossedLoopBackedge &&
            !llvm::is_contained(crossedLoopBackedges,
                                predecessor.crossedLoopBackedge)) {
          crossedLoopBackedges.push_back(predecessor.crossedLoopBackedge);
        }
        worklist.push_back({predecessor.value, state.access, state.indices,
                            crossedLoopBackedges});
      }
      return;
    }
    origins.insert(state.value);
  }

  ValueOriginAnalysis::Options options;
};

SmallVector<Value> ValueOriginAnalysis::getOrigins(Value value) const {
  return OriginWalker(options).getOrigins(value);
}

} // namespace mlir::tt
