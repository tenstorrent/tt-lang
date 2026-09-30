// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTensorRegions.h"

#include "ttlang/Dialect/TTL/Transforms/PipeNetExecutionUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace mlir::tt::ttl {

std::optional<int64_t> getTensorSliceGlobalIndex(TensorSliceOp slice) {
  auto tensorArgument = dyn_cast<BlockArgument>(slice.getTensor());
  auto function =
      tensorArgument
          ? dyn_cast<func::FuncOp>(tensorArgument.getOwner()->getParentOp())
          : func::FuncOp();
  auto crtaIndices =
      function ? function->getAttrOfType<ArrayAttr>(kCRTAIndicesAttrName)
               : ArrayAttr();
  if (!crtaIndices ||
      tensorArgument.getOwner() != &function.getBody().front() ||
      tensorArgument.getArgNumber() >= crtaIndices.size()) {
    return std::nullopt;
  }
  return cast<IntegerAttr>(crtaIndices[tensorArgument.getArgNumber()]).getInt();
}

SmallVector<int64_t> getTensorRegionExtents(ArrayRef<int64_t> tensorGridShape,
                                            ArrayRef<int64_t> sliceShape) {
  int64_t rankDifference = tensorGridShape.size() - sliceShape.size();
  SmallVector<int64_t> extents(tensorGridShape.size(), 1);
  for (int64_t dimension = rankDifference;
       dimension < static_cast<int64_t>(tensorGridShape.size()); ++dimension) {
    extents[dimension] = sliceShape[dimension - rankDifference];
  }
  return extents;
}

SmallVector<int64_t> TensorRegionOccurrences::getExtents() const {
  return getTensorRegionExtents(tensorGridShape, sliceShape);
}

FailureOr<SmallVector<int64_t>>
evaluateTensorSliceStart(TensorSliceOp slice,
                         IntegerExpressionEvaluator &evaluator,
                         std::size_t *failedDimension) {
  SmallVector<int64_t> startIndices;
  startIndices.reserve(slice.getIndices().size());
  for (auto [dimension, index] : llvm::enumerate(slice.getIndices())) {
    std::optional<llvm::APInt> resolvedIndex = evaluator.evaluate(index);
    if (!resolvedIndex || !resolvedIndex->isSignedIntN(64)) {
      if (failedDimension) {
        *failedDimension = dimension;
      }
      return failure();
    }
    startIndices.push_back(resolvedIndex->getSExtValue());
  }
  return startIndices;
}

static bool boxesOverlap(ArrayRef<int64_t> lhsStart,
                         ArrayRef<int64_t> lhsExtents,
                         ArrayRef<int64_t> rhsStart,
                         ArrayRef<int64_t> rhsExtents) {
  for (std::size_t dimension = 0; dimension < lhsStart.size(); ++dimension) {
    if (lhsStart[dimension] + lhsExtents[dimension] <= rhsStart[dimension] ||
        rhsStart[dimension] + rhsExtents[dimension] <= lhsStart[dimension]) {
      return false;
    }
  }
  return true;
}

namespace {
/// Occurrence starts indexed by the cell that contains them. Cells are at
/// least as large as every indexed or queried region, so two regions overlap
/// only when their start cells differ by at most one in every dimension. The
/// key uses at most `kMaxIndexedDimensions` dimensions, those with the most
/// distinct cells, so a query visits at most 81 keys; callers compare candidate
/// boxes in every dimension.
template <typename Entry>
class OccurrenceCellIndex {
public:
  static constexpr std::size_t kMaxIndexedDimensions = 4;

  OccurrenceCellIndex(ArrayRef<int64_t> cellExtents,
                      ArrayRef<std::pair<ArrayRef<int64_t>, Entry>> entries)
      : cellExtents(cellExtents) {
    std::size_t rank = cellExtents.size();
    SmallVector<llvm::DenseSet<int64_t>> distinctCells(rank);
    for (const auto &[start, entry] : entries) {
      for (std::size_t dimension = 0; dimension < rank; ++dimension) {
        distinctCells[dimension].insert(getCell(start, dimension));
      }
    }
    indexedDimensions = llvm::to_vector(llvm::seq<std::size_t>(0, rank));
    llvm::stable_sort(indexedDimensions, [&](std::size_t lhs, std::size_t rhs) {
      return distinctCells[lhs].size() > distinctCells[rhs].size();
    });
    indexedDimensions.truncate(std::min(rank, kMaxIndexedDimensions));
    for (const auto &[start, entry] : entries) {
      entriesByKey[getKey(start)].push_back(entry);
    }
  }

  /// Call `visit` with every entry whose start cell is within one cell of the
  /// cell of `start` in each indexed dimension, until it returns true, and
  /// return whether it did.
  bool forEachCandidate(ArrayRef<int64_t> start,
                        llvm::function_ref<bool(const Entry &)> visit) const {
    std::vector<int64_t> key = getKey(start);
    std::size_t neighborCount = 1;
    for (std::size_t position = 0; position < key.size(); ++position) {
      neighborCount *= 3;
    }
    for (std::size_t neighbor = 0; neighbor < neighborCount; ++neighbor) {
      std::vector<int64_t> neighborKey = key;
      std::size_t offsets = neighbor;
      for (int64_t &cell : neighborKey) {
        cell += static_cast<int64_t>(offsets % 3) - 1;
        offsets /= 3;
      }
      auto entriesIt = entriesByKey.find(neighborKey);
      if (entriesIt != entriesByKey.end() &&
          llvm::any_of(entriesIt->second, visit)) {
        return true;
      }
    }
    return false;
  }

private:
  int64_t getCell(ArrayRef<int64_t> start, std::size_t dimension) const {
    return llvm::divideFloorSigned(start[dimension], cellExtents[dimension]);
  }

  std::vector<int64_t> getKey(ArrayRef<int64_t> start) const {
    std::vector<int64_t> key;
    key.reserve(indexedDimensions.size());
    for (std::size_t dimension : indexedDimensions) {
      key.push_back(getCell(start, dimension));
    }
    return key;
  }

  SmallVector<int64_t> cellExtents;
  SmallVector<std::size_t> indexedDimensions;
  std::map<std::vector<int64_t>, SmallVector<Entry>> entriesByKey;
};
} // namespace

bool forEachOverlappingOccurrencePair(
    const TensorRegionOccurrences &lhs, const TensorRegionOccurrences &rhs,
    llvm::function_ref<bool(std::size_t, std::size_t)> visit) {
  if (lhs.tensorGridShape != rhs.tensorGridShape) {
    for (std::size_t lhsIndex = 0; lhsIndex < lhs.startIndices.size();
         ++lhsIndex) {
      for (std::size_t rhsIndex = 0; rhsIndex < rhs.startIndices.size();
           ++rhsIndex) {
        if (visit(lhsIndex, rhsIndex)) {
          return true;
        }
      }
    }
    return false;
  }

  SmallVector<int64_t> lhsExtents = lhs.getExtents();
  SmallVector<int64_t> rhsExtents = rhs.getExtents();
  SmallVector<int64_t> cellExtents;
  for (auto [lhsExtent, rhsExtent] : llvm::zip_equal(lhsExtents, rhsExtents)) {
    cellExtents.push_back(std::max<int64_t>({lhsExtent, rhsExtent, 1}));
  }
  SmallVector<std::pair<ArrayRef<int64_t>, std::size_t>> rhsEntries;
  for (std::size_t rhsIndex = 0; rhsIndex < rhs.startIndices.size();
       ++rhsIndex) {
    rhsEntries.push_back({rhs.startIndices[rhsIndex], rhsIndex});
  }
  OccurrenceCellIndex<std::size_t> rhsByCell(cellExtents, rhsEntries);
  for (std::size_t lhsIndex = 0; lhsIndex < lhs.startIndices.size();
       ++lhsIndex) {
    ArrayRef<int64_t> lhsStart = lhs.startIndices[lhsIndex];
    if (rhsByCell.forEachCandidate(lhsStart, [&](const std::size_t &rhsIndex) {
          return boxesOverlap(lhsStart, lhsExtents, rhs.startIndices[rhsIndex],
                              rhsExtents) &&
                 visit(lhsIndex, rhsIndex);
        })) {
      return true;
    }
  }
  return false;
}

bool hasOverlappingTensorRegionOccurrences(
    const TensorRegionOccurrences &region) {
  return forEachOverlappingOccurrencePair(
      region, region, [](std::size_t lhsIndex, std::size_t rhsIndex) {
        return lhsIndex != rhsIndex;
      });
}

bool tensorRegionOccurrencesOverlap(const TensorRegionOccurrences &lhs,
                                    const TensorRegionOccurrences &rhs) {
  return forEachOverlappingOccurrencePair(
      lhs, rhs, [](std::size_t, std::size_t) { return true; });
}

bool tensorRegionDestinationsMayAlias(const TensorRegionOccurrences &lhs,
                                      const TensorRegionOccurrences &rhs) {
  return devicesMayCoincide(lhs.device, rhs.device) &&
         lhs.globalTensorIndex == rhs.globalTensorIndex &&
         tensorRegionOccurrencesOverlap(lhs, rhs);
}

SmallVector<bool> computeDisjointTensorRegionDestinations(
    ArrayRef<TensorRegionOccurrences> destinations) {
  SmallVector<bool> disjoint = llvm::map_to_vector(
      destinations, [](const TensorRegionOccurrences &destination) {
        return !hasOverlappingTensorRegionOccurrences(destination);
      });

  // Destinations of one tensor with different tile-grid shapes may alias
  // anywhere when they may share a device. Two distinct shapes per device, per
  // unknown device, and per tensor decide whether another shape exists.
  using GridShapes = SmallVector<ArrayRef<int64_t>, 2>;
  auto addGridShape = [](GridShapes &shapes, ArrayRef<int64_t> shape) {
    if (shapes.size() < 2 && !llvm::is_contained(shapes, shape)) {
      shapes.push_back(shape);
    }
  };
  auto hasOtherGridShape = [](const GridShapes &shapes,
                              ArrayRef<int64_t> shape) {
    return shapes.size() > 1 || (shapes.size() == 1 && shapes.front() != shape);
  };
  struct TensorGridShapes {
    GridShapes all;
    GridShapes onUnknownDevice;
    llvm::DenseMap<DeviceRefAttr, GridShapes> byDevice;
  };
  std::map<int64_t, TensorGridShapes> gridShapesByTensor;
  for (const TensorRegionOccurrences &destination : destinations) {
    if (destination.startIndices.empty()) {
      continue;
    }
    TensorGridShapes &shapes =
        gridShapesByTensor[destination.globalTensorIndex];
    addGridShape(shapes.all, destination.tensorGridShape);
    addGridShape(destination.device ? shapes.byDevice[destination.device]
                                    : shapes.onUnknownDevice,
                 destination.tensorGridShape);
  }
  for (auto [index, destination] : llvm::enumerate(destinations)) {
    if (destination.startIndices.empty()) {
      continue;
    }
    const TensorGridShapes &shapes =
        gridShapesByTensor.at(destination.globalTensorIndex);
    ArrayRef<int64_t> shape = destination.tensorGridShape;
    bool otherShapeMayShareDevice =
        destination.device
            ? hasOtherGridShape(shapes.byDevice.lookup(destination.device),
                                shape) ||
                  hasOtherGridShape(shapes.onUnknownDevice, shape)
            : hasOtherGridShape(shapes.all, shape);
    if (otherShapeMayShareDevice) {
      disjoint[index] = false;
    }
  }

  std::map<std::pair<int64_t, std::vector<int64_t>>, SmallVector<std::size_t>>
      byGrid;
  for (auto [index, destination] : llvm::enumerate(destinations)) {
    std::vector<int64_t> gridShape(destination.tensorGridShape.begin(),
                                   destination.tensorGridShape.end());
    byGrid[{destination.globalTensorIndex, gridShape}].push_back(index);
  }
  for (const auto &entry : byGrid) {
    ArrayRef<std::size_t> indices = entry.second;
    if (indices.size() < 2) {
      continue;
    }
    std::size_t rank = entry.first.second.size();
    SmallVector<SmallVector<int64_t>> extents;
    SmallVector<int64_t> cellExtents(rank, 1);
    for (std::size_t index : indices) {
      extents.push_back(destinations[index].getExtents());
      for (std::size_t dimension = 0; dimension < rank; ++dimension) {
        cellExtents[dimension] =
            std::max(cellExtents[dimension], extents.back()[dimension]);
      }
    }
    // Each distinct start of a destination is indexed once, so a destination
    // that reuses one region in every iteration adds one entry.
    using CellEntry = std::pair<std::size_t, ArrayRef<int64_t>>;
    SmallVector<std::pair<ArrayRef<int64_t>, CellEntry>> cellEntries;
    SmallVector<SmallVector<ArrayRef<int64_t>>> distinctStarts(indices.size());
    for (std::size_t position = 0; position < indices.size(); ++position) {
      llvm::DenseSet<ArrayRef<int64_t>> seen;
      for (ArrayRef<int64_t> start :
           destinations[indices[position]].startIndices) {
        if (seen.insert(start).second) {
          distinctStarts[position].push_back(start);
          cellEntries.push_back({start, {position, start}});
        }
      }
    }
    OccurrenceCellIndex<CellEntry> occurrencesByCell(cellExtents, cellEntries);
    for (std::size_t position = 0; position < indices.size(); ++position) {
      std::size_t index = indices[position];
      for (ArrayRef<int64_t> start : distinctStarts[position]) {
        if (!disjoint[index]) {
          break;
        }
        occurrencesByCell.forEachCandidate(start, [&](const CellEntry &other) {
          auto [otherPosition, otherStart] = other;
          std::size_t otherIndex = indices[otherPosition];
          if (otherPosition == position ||
              !devicesMayCoincide(destinations[index].device,
                                  destinations[otherIndex].device) ||
              !boxesOverlap(start, extents[position], otherStart,
                            extents[otherPosition])) {
            return false;
          }
          disjoint[index] = false;
          disjoint[otherIndex] = false;
          return true;
        });
      }
    }
  }
  return disjoint;
}

/// Return the first dimension in which a region of `extents` starting at
/// `start` leaves `tensorGridShape`. Regions inside the grid keep start plus
/// extent free of overflow in the overlap tests.
static std::optional<std::size_t>
findDimensionOutsideTileGrid(ArrayRef<int64_t> start, ArrayRef<int64_t> extents,
                             ArrayRef<int64_t> tensorGridShape) {
  for (std::size_t dimension = 0; dimension < tensorGridShape.size();
       ++dimension) {
    if (start[dimension] < 0 || extents[dimension] <= 0 ||
        start[dimension] > tensorGridShape[dimension] - extents[dimension]) {
      return dimension;
    }
  }
  return std::nullopt;
}

static LogicalResult
reportEnumerationBound(llvm::function_ref<InFlightDiagnostic()> emitError,
                       std::uint64_t maxIterations) {
  if (emitError) {
    emitError() << "pipe receive tensor_slice enumeration supports at most "
                << maxIterations << " loop iterations and executions";
  }
  return failure();
}

static LogicalResult
reportUnenumerableLoops(llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (emitError) {
    emitError() << "pipe receive tensor_slice requires statically enumerable "
                   "enclosing loops";
  }
  return failure();
}

/// Return the `scf.for` loops enclosing `op` whose induction variables
/// `valueEvaluator` cannot evaluate, outermost first. Fails when a trip count
/// is not evaluable before any enumerated induction variable is bound, or when
/// the loops have more than `maxIterations` combined iterations.
static FailureOr<SmallVector<scf::ForOp>> resolveEnumeratedLoops(
    Operation *op,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator,
    std::uint64_t maxIterations,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  IntegerExpressionEvaluator evaluator(valueEvaluator);
  SmallVector<scf::ForOp> loops;
  for (Operation *ancestor = op->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(ancestor);
    if (loop && !evaluator.evaluate(loop.getInductionVar())) {
      loops.push_back(loop);
    }
    if (isa<func::FuncOp>(ancestor)) {
      break;
    }
  }
  std::reverse(loops.begin(), loops.end());

  SmallVector<std::uint64_t> tripCounts;
  for (scf::ForOp loop : loops) {
    std::optional<std::uint64_t> tripCount =
        getLoopTripCount(loop, LoopInductionBindings(), valueEvaluator);
    if (!tripCount) {
      return reportUnenumerableLoops(emitError);
    }
    tripCounts.push_back(*tripCount);
  }
  if (llvm::is_contained(tripCounts, 0)) {
    return loops;
  }
  std::uint64_t iterationCount = 1;
  for (std::uint64_t tripCount : tripCounts) {
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned(iterationCount, tripCount);
    if (!product || *product > maxIterations) {
      return reportEnumerationBound(emitError, maxIterations);
    }
    iterationCount = *product;
  }
  return loops;
}

/// Bind every combination of the induction values of `loops`, outermost
/// first, and call `visit` for each. Stops at the first failure; a failure of
/// the enumeration itself, such as a bound that depends on runtime values, is
/// reported through `emitError`.
static LogicalResult forEachLoopIteration(
    ArrayRef<scf::ForOp> loops, LoopInductionBindings &bindings,
    const IntegerExpressionEvaluator::ValueEvaluator &valueEvaluator,
    llvm::function_ref<InFlightDiagnostic()> emitError,
    function_ref<LogicalResult(const LoopInductionBindings &)> visit) {
  SmallVector<LoopLikeOpInterface> loopLikes = llvm::map_to_vector(
      loops, [](scf::ForOp loop) -> LoopLikeOpInterface { return loop; });
  // resolveEnumeratedLoops has bounded the iteration count.
  EnumerationBudget budget(std::numeric_limits<std::uint64_t>::max());
  bool visitFailed = false;
  if (succeeded(enumerateLoopNest(
          loopLikes, bindings, budget,
          [&](const LoopInductionBindings &iteration) {
            visitFailed = failed(visit(iteration));
            return failure(visitFailed);
          },
          valueEvaluator))) {
    return success();
  }
  return visitFailed ? failure() : reportUnenumerableLoops(emitError);
}

/// Evaluate the start indices of `slice` for the bound induction values.
static FailureOr<SmallVector<int64_t>>
evaluateSliceStart(TensorSliceOp slice, IntegerExpressionEvaluator &evaluator,
                   StringRef failureReason,
                   llvm::function_ref<InFlightDiagnostic()> emitError) {
  std::size_t failedDimension = 0;
  FailureOr<SmallVector<int64_t>> startIndices =
      evaluateTensorSliceStart(slice, evaluator, &failedDimension);
  if (failed(startIndices) && emitError) {
    InFlightDiagnostic diagnostic = emitError();
    diagnostic << "pipe receive tensor_slice start index in dimension "
               << failedDimension << " is not statically enumerable";
    if (!failureReason.empty()) {
      diagnostic << ": " << failureReason;
    }
  }
  return startIndices;
}

/// Return the bound induction values of `loops`, outermost first.
static SmallVector<int64_t>
boundInductionValues(ArrayRef<scf::ForOp> loops,
                     const LoopInductionBindings &bindings) {
  return llvm::map_to_vector(loops, [&](scf::ForOp loop) {
    return bindings.lookup(loop.getInductionVar()).getSExtValue();
  });
}

FailureOr<TensorSliceOccurrences> enumerateTensorSliceOccurrences(
    TensorSliceOp slice, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state, std::uint64_t expectedExecutionCount,
    TensorSliceOccurrenceValueEvaluator evaluateContextValue,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto reportFailure = [&](const auto &...message) -> LogicalResult {
    if (emitError) {
      (emitError() << ... << message);
    }
    return failure();
  };
  if (expectedExecutionCount > kMaxEnumeratedTensorSliceOccurrences) {
    return reportEnumerationBound(emitError,
                                  kMaxEnumeratedTensorSliceOccurrences);
  }
  LoopInductionBindings bindings;
  std::string failureReason;
  IntegerExpressionEvaluator::ValueEvaluator evaluateInContext =
      [&](Value value) -> std::optional<llvm::APInt> {
    if (std::optional<llvm::APInt> contextValue =
            evaluateContextValue(value, bindings, failureReason)) {
      return contextValue;
    }
    return evaluateIntegerAtLaunchLocation(value, location, state);
  };
  FailureOr<SmallVector<scf::ForOp>> loops =
      resolveEnumeratedLoops(slice, evaluateInContext,
                             kMaxEnumeratedTensorSliceOccurrences, emitError);
  if (failed(loops)) {
    return failure();
  }

  SmallVector<std::pair<scf::IfOp, unsigned>> enclosingBranches;
  Operation *nestedOperation = slice;
  for (Operation *ancestor = slice->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    if (auto branch = dyn_cast<scf::IfOp>(ancestor)) {
      enclosingBranches.push_back(
          {branch,
           nestedOperation->getBlock()->getParent()->getRegionNumber()});
    }
    nestedOperation = ancestor;
    if (isa<func::FuncOp>(ancestor)) {
      break;
    }
  }

  TensorSliceOccurrences result;
  result.loops = *loops;
  SmallVector<SmallVector<int64_t>> &occurrences = result.startIndices;
  failureReason.clear();
  if (failed(forEachLoopIteration(
          *loops, bindings, evaluateInContext, emitError,
          [&](const LoopInductionBindings &iteration) -> LogicalResult {
            IntegerExpressionEvaluator occurrenceEvaluator =
                createLoopIntegerEvaluator(iteration, evaluateInContext);
            for (auto [branch, selectedRegion] : enclosingBranches) {
              std::optional<llvm::APInt> condition =
                  occurrenceEvaluator.evaluate(branch.getCondition());
              if (!condition || condition->getBitWidth() != 1) {
                return reportFailure("pipe receive tensor_slice requires "
                                     "statically enumerable enclosing "
                                     "conditions");
              }
              if (condition->getBoolValue() != (selectedRegion == 0)) {
                return success();
              }
            }
            FailureOr<SmallVector<int64_t>> startIndices = evaluateSliceStart(
                slice, occurrenceEvaluator, failureReason, emitError);
            if (failed(startIndices)) {
              return failure();
            }
            if (occurrences.size() >= expectedExecutionCount) {
              return reportFailure("pipe receive tensor_slice enumeration "
                                   "exceeds its proven transfer count");
            }
            occurrences.push_back(std::move(*startIndices));
            result.inductionValues.push_back(
                boundInductionValues(*loops, iteration));
            return success();
          }))) {
    return failure();
  }
  if (loops->empty() && occurrences.size() == 1) {
    occurrences.resize(expectedExecutionCount, occurrences.front());
    result.inductionValues.resize(expectedExecutionCount);
  }
  if (occurrences.size() != expectedExecutionCount) {
    return reportFailure("pipe receive tensor_slice enumeration does not "
                         "match its proven transfer count");
  }

  ArrayRef<int64_t> tensorGridShape =
      cast<RankedTensorType>(slice.getTensor().getType()).getShape();
  SmallVector<int64_t> extents = getTensorRegionExtents(
      tensorGridShape, cast<RankedTensorType>(slice.getType()).getShape());
  for (ArrayRef<int64_t> startIndices : occurrences) {
    if (std::optional<std::size_t> dimension = findDimensionOutsideTileGrid(
            startIndices, extents, tensorGridShape)) {
      return reportFailure("pipe receive tensor_slice region in dimension ",
                           *dimension, " starts at ", startIndices[*dimension],
                           " with extent ", extents[*dimension],
                           ", outside the destination tensor tile grid ",
                           tensorGridShape[*dimension]);
    }
  }
  return result;
}

FailureOr<TensorSliceOccurrences>
enumerateTensorSliceIterationStarts(TensorSliceOp slice, Operation *user,
                                    const LaunchExecutionLocation &location,
                                    const LaunchNodeDomainState &state,
                                    std::uint64_t maxIterations) {
  IntegerExpressionEvaluator::ValueEvaluator evaluateAtLocation =
      [&](Value value) {
        return evaluateIntegerAtLaunchLocation(value, location, state);
      };
  FailureOr<SmallVector<scf::ForOp>> loops = resolveEnumeratedLoops(
      user, evaluateAtLocation, maxIterations, /*emitError=*/{});
  if (failed(loops)) {
    return failure();
  }
  ArrayRef<int64_t> tensorGridShape =
      cast<RankedTensorType>(slice.getTensor().getType()).getShape();
  SmallVector<int64_t> extents = getTensorRegionExtents(
      tensorGridShape, cast<RankedTensorType>(slice.getType()).getShape());
  TensorSliceOccurrences result;
  result.loops = *loops;
  LoopInductionBindings bindings;
  if (failed(forEachLoopIteration(
          *loops, bindings, evaluateAtLocation, /*emitError=*/{},
          [&](const LoopInductionBindings &iteration) -> LogicalResult {
            IntegerExpressionEvaluator iterationEvaluator =
                createLoopIntegerEvaluator(iteration, evaluateAtLocation);
            FailureOr<SmallVector<int64_t>> startIndices =
                evaluateTensorSliceStart(slice, iterationEvaluator);
            if (failed(startIndices) ||
                findDimensionOutsideTileGrid(*startIndices, extents,
                                             tensorGridShape)) {
              return failure();
            }
            result.startIndices.push_back(std::move(*startIndices));
            result.inductionValues.push_back(
                boundInductionValues(*loops, iteration));
            return success();
          }))) {
    return failure();
  }
  return result;
}

bool canOmitFabricReceiverRendezvous(bool isDeviceTransfer, bool isPointToPoint,
                                     bool hasSingleReceiver,
                                     bool hasDisjointTensorRegionDestination) {
  return isDeviceTransfer && isPointToPoint && hasSingleReceiver &&
         hasDisjointTensorRegionDestination;
}

} // namespace mlir::tt::ttl
