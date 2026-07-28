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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::tt {

namespace {

/// One SSA value or tensor element awaiting backward traversal.
struct TraversalState {
  Value value;
  Operation *access = nullptr;
  SmallVector<Value> indices;
  llvm::SmallSetVector<Operation *, 4> crossedLoopBackedges;
  llvm::SmallSetVector<Operation *, 4> fullyOverwrittenLoops;

  bool isTensorElement() const { return access != nullptr; }

  static TraversalState forValue(Value value) {
    TraversalState state;
    state.value = value;
    return state;
  }

  static TraversalState forElement(
      Value value, Operation *access, ValueRange indices,
      const llvm::SmallSetVector<Operation *, 4> &crossedLoopBackedges = {},
      const llvm::SmallSetVector<Operation *, 4> &fullyOverwrittenLoops = {}) {
    TraversalState state;
    state.value = value;
    state.access = access;
    state.indices.append(indices.begin(), indices.end());
    state.crossedLoopBackedges = crossedLoopBackedges;
    state.fullyOverwrittenLoops = fullyOverwrittenLoops;
    return state;
  }
};

bool equalOperationSets(const llvm::SmallSetVector<Operation *, 4> &lhs,
                        const llvm::SmallSetVector<Operation *, 4> &rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(lhs, [&](Operation *operation) {
           return rhs.contains(operation);
         });
}

bool operator==(const TraversalState &lhs, const TraversalState &rhs) {
  return lhs.value == rhs.value && lhs.access == rhs.access &&
         lhs.indices == rhs.indices &&
         equalOperationSets(lhs.crossedLoopBackedges,
                            rhs.crossedLoopBackedges) &&
         equalOperationSets(lhs.fullyOverwrittenLoops,
                            rhs.fullyOverwrittenLoops);
}

/// Possible and definite effects of one tensor insertion on an element query.
struct IndexAccessRelation {
  bool mayDefine = true;
  bool mustDefine = false;
  Operation *fullyOverwrittenLoop = nullptr;
};

using IndexTuple = SmallVector<std::int64_t, 4>;

struct IndexTupleLess {
  /// Orders concrete index tuples for deterministic set operations.
  bool operator()(const IndexTuple &lhs, const IndexTuple &rhs) const {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end());
  }
};

using IndexTupleSet = SmallVector<IndexTuple, 8>;

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

/// Tracks work performed while resolving one origin query.
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

/// One control-flow predecessor value and the loop backedge crossed, if any.
struct PredecessorValue {
  Value value;
  Operation *crossedLoopBackedge = nullptr;
  Operation *initialLoop = nullptr;
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
      OpOperand *initial = loop.getTiedLoopInit(result);
      BlockArgument iteratedArgument = loop.getTiedLoopRegionIterArg(result);
      OpOperand *yielded = iteratedArgument
                               ? loop.getTiedLoopYieldedValue(iteratedArgument)
                               : nullptr;
      if (initial && yielded) {
        LoopExecutionInfo execution = getLoopExecutionInfo(loop);
        SmallVector<PredecessorValue> predecessors;
        if (execution.maySkip) {
          predecessors.push_back({initial->get()});
        }
        if (execution.mayExecute) {
          predecessors.push_back({yielded->get()});
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
    OpOperand *initial = loop.getTiedLoopInit(blockArgument);
    OpOperand *yielded = loop.getTiedLoopYieldedValue(blockArgument);
    if (initial && yielded) {
      LoopExecutionInfo execution = getLoopExecutionInfo(loop);
      SmallVector<PredecessorValue> predecessors;
      if (execution.mayExecute) {
        predecessors.push_back({initial->get(), /*crossedLoopBackedge=*/nullptr,
                                /*initialLoop=*/loop.getOperation()});
      }
      if (execution.mayRepeat) {
        predecessors.push_back({yielded->get(), loop.getOperation()});
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

/// Collect concrete index tuples over the selected enclosing loops.
LogicalResult
collectIndexTuples(ArrayRef<Value> indices, ArrayRef<LoopLikeOpInterface> loops,
                   std::size_t firstLoop, LoopInductionBindings &bindings,
                   EnumerationState &enumeration, IndexTupleSet &tuples) {
  LogicalResult result = enumerateLoopNest(
      loops.drop_front(firstLoop), bindings, enumeration.loopIterations,
      [&](const LoopInductionBindings &currentBindings) {
        if (!enumeration.indexTuples.tryConsume()) {
          return failure();
        }
        IndexTuple tuple;
        tuple.reserve(indices.size());
        for (Value index : indices) {
          std::optional<std::int64_t> maybeIndex =
              evaluateIndexExpression(index, currentBindings);
          if (!maybeIndex) {
            return failure();
          }
          tuple.push_back(*maybeIndex);
        }
        tuples.push_back(std::move(tuple));
        return success();
      });
  if (failed(result)) {
    return failure();
  }
  llvm::sort(tuples, IndexTupleLess());
  tuples.erase(std::unique(tuples.begin(), tuples.end()), tuples.end());
  return success();
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

/// A finite increasing arithmetic progression.
struct FiniteSequence {
  std::int64_t first;
  std::int64_t step;
  std::uint64_t count;
};

std::optional<FiniteSequence>
getFiniteSequence(Value index, ArrayRef<LoopLikeOpInterface> loops) {
  if (std::optional<std::int64_t> constant = evaluateIndexExpression(index)) {
    return FiniteSequence{*constant, /*step=*/0, /*count=*/1};
  }

  for (LoopLikeOpInterface loop : loops) {
    std::optional<SmallVector<Value>> inductionVariables =
        loop.getLoopInductionVars();
    std::optional<SmallVector<OpFoldResult>> lowerBounds =
        loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
    if (!inductionVariables || !lowerBounds || !steps) {
      continue;
    }
    auto inductionVariable = llvm::find(*inductionVariables, index);
    if (inductionVariable == inductionVariables->end()) {
      continue;
    }
    std::size_t dimension =
        std::distance(inductionVariables->begin(), inductionVariable);
    if (dimension >= lowerBounds->size() || dimension >= steps->size()) {
      return std::nullopt;
    }
    std::optional<std::int64_t> lower =
        evaluateIndexExpression((*lowerBounds)[dimension]);
    std::optional<std::int64_t> step =
        evaluateIndexExpression((*steps)[dimension]);
    std::optional<std::uint64_t> count = getLoopTripCount(loop);
    if (!lower || !step || *step <= 0 || !count) {
      return std::nullopt;
    }
    return FiniteSequence{*lower, *step, *count};
  }
  return std::nullopt;
}

/// Return whether every operation in a loop nest executes at least once.
std::optional<bool> isLoopNestNonEmpty(ArrayRef<LoopLikeOpInterface> loops) {
  for (LoopLikeOpInterface loop : loops) {
    std::optional<std::uint64_t> count = getLoopTripCount(loop);
    if (!count) {
      return std::nullopt;
    }
    if (*count == 0) {
      return false;
    }
  }
  return true;
}

std::optional<__int128_t> getLast(const FiniteSequence &sequence) {
  if (sequence.count == 0) {
    return std::nullopt;
  }
  return static_cast<__int128_t>(sequence.first) +
         static_cast<__int128_t>(sequence.step) * (sequence.count - 1);
}

bool contains(const FiniteSequence &sequence, std::int64_t value) {
  std::optional<__int128_t> last = getLast(sequence);
  if (!last || value < sequence.first ||
      static_cast<__int128_t>(value) > *last) {
    return false;
  }
  if (sequence.count == 1) {
    return value == sequence.first;
  }
  return (static_cast<__int128_t>(value) - sequence.first) % sequence.step == 0;
}

bool isSubset(const FiniteSequence &subset, const FiniteSequence &superset) {
  if (subset.count == 0) {
    return true;
  }
  if (superset.count == 0 || !contains(superset, subset.first)) {
    return false;
  }
  if (subset.count == 1) {
    return true;
  }
  if (superset.count == 1) {
    return false;
  }
  std::optional<__int128_t> subsetLast = getLast(subset);
  std::optional<__int128_t> supersetLast = getLast(superset);
  return subsetLast && supersetLast && *subsetLast <= *supersetLast &&
         subset.step % superset.step == 0;
}

bool areDisjoint(const FiniteSequence &lhs, const FiniteSequence &rhs) {
  std::optional<__int128_t> lhsLast = getLast(lhs);
  std::optional<__int128_t> rhsLast = getLast(rhs);
  if (!lhsLast || !rhsLast) {
    return true;
  }
  if (*lhsLast < rhs.first || *rhsLast < lhs.first) {
    return true;
  }
  if (lhs.count == 1) {
    return !contains(rhs, lhs.first);
  }
  if (rhs.count == 1) {
    return !contains(lhs, rhs.first);
  }
  return false;
}

/// Compare read and write indices over their finite enclosing loop domains.
/// Shared enclosing loops remain correlated; independent loops contribute
/// separate iteration dimensions.
IndexAccessRelation compareIndexDomains(
    Operation *read, ArrayRef<Value> readIndices, Operation *write,
    ArrayRef<Value> writeIndices,
    const llvm::SmallSetVector<Operation *, 4> &crossedLoopBackedges,
    EnumerationState &enumeration) {
  if (readIndices.size() != writeIndices.size()) {
    return {};
  }
  SmallVector<LoopLikeOpInterface> writeLoops = getEnclosingLoops(write);
  if (llvm::any_of(crossedLoopBackedges, [&](Operation *loop) {
        return llvm::any_of(writeLoops, [&](LoopLikeOpInterface writeLoop) {
          return writeLoop.getOperation() == loop;
        });
      })) {
    return {};
  }
  if (readIndices == writeIndices) {
    return {/*mayDefine=*/true, /*mustDefine=*/true};
  }

  SmallVector<LoopLikeOpInterface> readLoops = getEnclosingLoops(read);
  std::size_t commonLoopCount = 0;
  while (commonLoopCount < readLoops.size() &&
         commonLoopCount < writeLoops.size() &&
         readLoops[commonLoopCount].getOperation() ==
             writeLoops[commonLoopCount].getOperation()) {
    ++commonLoopCount;
  }

  Operation *fullyOverwrittenLoop =
      commonLoopCount < writeLoops.size()
          ? writeLoops[commonLoopCount].getOperation()
          : nullptr;

  if (readIndices.size() == 1) {
    std::optional<FiniteSequence> readSequence =
        getFiniteSequence(readIndices.front(), readLoops);
    std::optional<FiniteSequence> writeSequence =
        getFiniteSequence(writeIndices.front(), writeLoops);
    if (readSequence && writeSequence) {
      std::optional<bool> readDomainNonEmpty = isLoopNestNonEmpty(readLoops);
      std::optional<bool> writeDomainNonEmpty = isLoopNestNonEmpty(writeLoops);
      if (writeDomainNonEmpty && !*writeDomainNonEmpty) {
        return {/*mayDefine=*/false, /*mustDefine=*/false};
      }
      if (areDisjoint(*readSequence, *writeSequence)) {
        return {/*mayDefine=*/false, /*mustDefine=*/false};
      }
      if (readDomainNonEmpty && *readDomainNonEmpty && writeDomainNonEmpty &&
          *writeDomainNonEmpty && isSubset(*readSequence, *writeSequence)) {
        bool mustDefine = readSequence->count == 1 && writeSequence->count == 1;
        return {/*mayDefine=*/true, mustDefine,
                mustDefine ? nullptr : fullyOverwrittenLoop};
      }
    }
  }

  bool foundRead = false;
  bool allReadsCovered = true;
  bool foundIntersection = false;
  bool allInstancesEqual = true;
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
        allInstancesEqual &= readTuples.size() == 1 &&
                             writeTuples.size() == 1 &&
                             readTuples.front() == writeTuples.front();
        return success();
      });

  if (failed(comparisonResult) || !foundRead) {
    return {};
  }
  return {/*mayDefine=*/foundIntersection,
          /*mustDefine=*/allInstancesEqual,
          allReadsCovered && !allInstancesEqual ? fullyOverwrittenLoop
                                                : nullptr};
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

/// Implements a conservative backward traversal from one SSA value.
class OriginWalker {
public:
  explicit OriginWalker(ValueOriginAnalysis::Options options)
      : enumeration(options) {}

  SmallVector<Value, 4> getOrigins(Value initialValue) {
    assert(initialValue && "value-origin analysis requires a value");

    SmallVector<TraversalState> worklist{
        TraversalState::forValue(initialValue)};
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
    return SmallVector<Value, 4>(origins.begin(), origins.end());
  }

private:
  /// Continue from an SSA value through modeled value associations.
  void visitValue(const TraversalState &state,
                  SmallVectorImpl<TraversalState> &worklist,
                  llvm::SetVector<Value> &origins) {
    if (auto cast = state.value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (isOneToOneCast(cast)) {
        worklist.push_back(TraversalState::forValue(cast.getInputs().front()));
        return;
      }
    }
    if (auto extract = state.value.getDefiningOp<tensor::ExtractOp>()) {
      worklist.push_back(TraversalState::forElement(
          extract.getTensor(), extract.getOperation(), extract.getIndices()));
      return;
    }
    if (std::optional<SmallVector<PredecessorValue>> predecessors =
            getValuePredecessors(state.value)) {
      for (const PredecessorValue &predecessor : *predecessors) {
        worklist.push_back(TraversalState::forValue(predecessor.value));
      }
      return;
    }
    origins.insert(state.value);
  }

  /// Visit one tensor element while preserving its access context.
  void visitTensorElement(const TraversalState &state,
                          SmallVectorImpl<TraversalState> &worklist,
                          llvm::SetVector<Value> &origins) {
    if (auto cast = state.value.getDefiningOp<tensor::CastOp>()) {
      worklist.push_back(TraversalState::forElement(
          cast.getSource(), state.access, state.indices,
          state.crossedLoopBackedges, state.fullyOverwrittenLoops));
      return;
    }
    if (auto insert = state.value.getDefiningOp<tensor::InsertOp>()) {
      SmallVector<Value> insertIndices(insert.getIndices());
      IndexAccessRelation relation = compareIndexDomains(
          state.access, state.indices, insert.getOperation(), insertIndices,
          state.crossedLoopBackedges, enumeration);
      if (relation.mayDefine) {
        worklist.push_back(TraversalState::forValue(insert.getScalar()));
      }
      if (!relation.mustDefine) {
        auto fullyOverwrittenLoops = state.fullyOverwrittenLoops;
        if (relation.fullyOverwrittenLoop) {
          fullyOverwrittenLoops.insert(relation.fullyOverwrittenLoop);
        }
        worklist.push_back(TraversalState::forElement(
            insert.getDest(), state.access, state.indices,
            state.crossedLoopBackedges, fullyOverwrittenLoops));
      }
      return;
    }
    if (std::optional<SmallVector<PredecessorValue>> predecessors =
            getValuePredecessors(state.value)) {
      for (const PredecessorValue &predecessor : *predecessors) {
        if (predecessor.initialLoop &&
            state.fullyOverwrittenLoops.contains(predecessor.initialLoop)) {
          continue;
        }
        auto crossedLoopBackedges = state.crossedLoopBackedges;
        if (predecessor.crossedLoopBackedge) {
          crossedLoopBackedges.insert(predecessor.crossedLoopBackedge);
        }
        worklist.push_back(TraversalState::forElement(
            predecessor.value, state.access, state.indices,
            crossedLoopBackedges, state.fullyOverwrittenLoops));
      }
      return;
    }
    origins.insert(state.value);
  }

  EnumerationState enumeration;
};

} // namespace

class ValueOriginAnalysis::Impl {
public:
  Impl(Operation *root, Options options) : root(root), options(options) {
    assert(root && "value-origin analysis requires a root operation");
  }

  const OriginSet &getOrigins(Value value) const {
    auto cached = origins.find(value);
    if (cached != origins.end()) {
      return *cached->second;
    }

    Operation *valueScope = value.getDefiningOp();
    if (!valueScope) {
      valueScope = cast<BlockArgument>(value).getOwner()->getParentOp();
    }
    assert((valueScope == root || root->isAncestor(valueScope)) &&
           "queried value must be nested under the analysis root");

    auto result =
        std::make_unique<OriginSet>(OriginWalker(options).getOrigins(value));
    return *origins.try_emplace(value, std::move(result)).first->second;
  }

private:
  Operation *root;
  Options options;
  mutable llvm::DenseMap<Value, std::unique_ptr<OriginSet>> origins;
};

ValueOriginAnalysis::ValueOriginAnalysis(Operation *root)
    : ValueOriginAnalysis(root, Options()) {}

ValueOriginAnalysis::ValueOriginAnalysis(Operation *root, Options options)
    : impl(std::make_unique<Impl>(root, options)) {}

ValueOriginAnalysis::~ValueOriginAnalysis() = default;
ValueOriginAnalysis::ValueOriginAnalysis(ValueOriginAnalysis &&) = default;
ValueOriginAnalysis &
ValueOriginAnalysis::operator=(ValueOriginAnalysis &&) = default;

const OriginSet &ValueOriginAnalysis::getOrigins(Value value) const {
  return impl->getOrigins(value);
}

} // namespace mlir::tt
