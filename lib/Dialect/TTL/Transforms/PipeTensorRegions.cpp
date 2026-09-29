// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTensorRegions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace mlir::tt::ttl {

bool tensorRegionsOverlap(const TensorRegionBounds &lhs,
                          const TensorRegionBounds &rhs) {
  if (lhs.tensorGridShape != rhs.tensorGridShape ||
      lhs.startIndices.size() != rhs.startIndices.size() ||
      lhs.extents.size() != rhs.extents.size()) {
    return true;
  }
  for (int64_t dimension = 0;
       dimension < static_cast<int64_t>(lhs.tensorGridShape.size());
       ++dimension) {
    int64_t lhsEnd = lhs.startIndices[dimension] + lhs.extents[dimension];
    int64_t rhsEnd = rhs.startIndices[dimension] + rhs.extents[dimension];
    if (lhsEnd <= rhs.startIndices[dimension] ||
        rhsEnd <= lhs.startIndices[dimension]) {
      return false;
    }
  }
  return true;
}

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

SmallVector<int64_t> TensorRegionOccurrences::getExtents() const {
  int64_t rankDifference = tensorGridShape.size() - sliceShape.size();
  SmallVector<int64_t> extents(tensorGridShape.size(), 1);
  for (int64_t dimension = rankDifference;
       dimension < static_cast<int64_t>(tensorGridShape.size()); ++dimension) {
    extents[dimension] = sliceShape[dimension - rankDifference];
  }
  return extents;
}

TensorRegionBounds
TensorRegionOccurrences::getBounds(ArrayRef<int64_t> occurrenceStart) const {
  return TensorRegionBounds{
      globalTensorIndex, SmallVector<int64_t>(tensorGridShape),
      SmallVector<int64_t>(occurrenceStart), getExtents()};
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

  // Boxes no larger than a cell overlap only when their start cells differ by
  // at most one in every dimension, so each occurrence is compared only with
  // the occurrences of the 3^rank neighboring cells.
  SmallVector<int64_t> lhsExtents = lhs.getExtents();
  SmallVector<int64_t> rhsExtents = rhs.getExtents();
  std::size_t rank = lhs.tensorGridShape.size();
  SmallVector<int64_t> cellExtents(rank);
  for (std::size_t dimension = 0; dimension < rank; ++dimension) {
    cellExtents[dimension] =
        std::max<int64_t>({lhsExtents[dimension], rhsExtents[dimension], 1});
  }
  auto getCell = [&](ArrayRef<int64_t> start) {
    std::vector<int64_t> cell(rank);
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      cell[dimension] =
          llvm::divideFloorSigned(start[dimension], cellExtents[dimension]);
    }
    return cell;
  };
  std::map<std::vector<int64_t>, SmallVector<std::size_t>> rhsByCell;
  for (auto [rhsIndex, rhsStart] : llvm::enumerate(rhs.startIndices)) {
    rhsByCell[getCell(rhsStart)].push_back(rhsIndex);
  }

  std::size_t neighborCount = 1;
  for (std::size_t dimension = 0; dimension < rank; ++dimension) {
    neighborCount *= 3;
  }
  for (auto [lhsIndex, lhsStart] : llvm::enumerate(lhs.startIndices)) {
    std::vector<int64_t> lhsCell = getCell(lhsStart);
    for (std::size_t neighbor = 0; neighbor < neighborCount; ++neighbor) {
      std::vector<int64_t> neighborCell = lhsCell;
      std::size_t offsets = neighbor;
      for (std::size_t dimension = 0; dimension < rank; ++dimension) {
        neighborCell[dimension] += static_cast<int64_t>(offsets % 3) - 1;
        offsets /= 3;
      }
      auto cellIt = rhsByCell.find(neighborCell);
      if (cellIt == rhsByCell.end()) {
        continue;
      }
      for (std::size_t rhsIndex : cellIt->second) {
        if (boxesOverlap(lhsStart, lhsExtents, rhs.startIndices[rhsIndex],
                         rhsExtents) &&
            visit(lhsIndex, rhsIndex)) {
          return true;
        }
      }
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
  bool provenDifferentDevices =
      lhs.device && rhs.device && lhs.device != rhs.device;
  return !provenDifferentDevices &&
         lhs.globalTensorIndex == rhs.globalTensorIndex &&
         tensorRegionOccurrencesOverlap(lhs, rhs);
}

SmallVector<bool> computeDisjointTensorRegionDestinations(
    ArrayRef<TensorRegionOccurrences> destinations) {
  SmallVector<bool> disjoint = llvm::map_to_vector(
      destinations, [](const TensorRegionOccurrences &destination) {
        return !hasOverlappingTensorRegionOccurrences(destination);
      });
  auto mayShareDevice = [&](std::size_t lhs, std::size_t rhs) {
    DeviceRefAttr lhsDevice = destinations[lhs].device;
    DeviceRefAttr rhsDevice = destinations[rhs].device;
    return !lhsDevice || !rhsDevice || lhsDevice == rhsDevice;
  };

  // Destinations of one tensor with different tile-grid shapes may alias
  // anywhere. Such pairs are rare and are compared directly.
  std::map<int64_t, SmallVector<std::size_t>> byTensor;
  for (auto [index, destination] : llvm::enumerate(destinations)) {
    byTensor[destination.globalTensorIndex].push_back(index);
  }
  for (const auto &entry : byTensor) {
    ArrayRef<std::size_t> indices = entry.second;
    for (std::size_t lhsPosition = 0; lhsPosition < indices.size();
         ++lhsPosition) {
      for (std::size_t rhsPosition = lhsPosition + 1;
           rhsPosition < indices.size(); ++rhsPosition) {
        std::size_t lhs = indices[lhsPosition];
        std::size_t rhs = indices[rhsPosition];
        if (destinations[lhs].tensorGridShape !=
                destinations[rhs].tensorGridShape &&
            !destinations[lhs].startIndices.empty() &&
            !destinations[rhs].startIndices.empty() &&
            mayShareDevice(lhs, rhs)) {
          disjoint[lhs] = false;
          disjoint[rhs] = false;
        }
      }
    }
  }

  // Occurrences of one tensor and tile grid are indexed by cells as large as
  // the largest region, so each occurrence is compared only with occurrences
  // in the 3^rank neighboring cells. This keeps the check linear in the total
  // occurrence count.
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
    auto getCell = [&](ArrayRef<int64_t> start) {
      std::vector<int64_t> cell(rank);
      for (std::size_t dimension = 0; dimension < rank; ++dimension) {
        cell[dimension] =
            llvm::divideFloorSigned(start[dimension], cellExtents[dimension]);
      }
      return cell;
    };
    // Each entry names a group position and an occurrence of that destination.
    std::map<std::vector<int64_t>,
             SmallVector<std::pair<std::size_t, std::size_t>>>
        byCell;
    for (auto [position, index] : llvm::enumerate(indices)) {
      for (auto [occurrence, start] :
           llvm::enumerate(destinations[index].startIndices)) {
        byCell[getCell(start)].push_back({position, occurrence});
      }
    }
    std::size_t neighborCount = 1;
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      neighborCount *= 3;
    }
    for (auto [position, index] : llvm::enumerate(indices)) {
      for (ArrayRef<int64_t> start : destinations[index].startIndices) {
        std::vector<int64_t> cell = getCell(start);
        for (std::size_t neighbor = 0; neighbor < neighborCount; ++neighbor) {
          std::vector<int64_t> neighborCell = cell;
          std::size_t offsets = neighbor;
          for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            neighborCell[dimension] += static_cast<int64_t>(offsets % 3) - 1;
            offsets /= 3;
          }
          auto cellIt = byCell.find(neighborCell);
          if (cellIt == byCell.end()) {
            continue;
          }
          for (auto [otherPosition, otherOccurrence] : cellIt->second) {
            std::size_t otherIndex = indices[otherPosition];
            if (otherPosition <= position ||
                !mayShareDevice(index, otherIndex) ||
                !boxesOverlap(
                    start, extents[position],
                    destinations[otherIndex].startIndices[otherOccurrence],
                    extents[otherPosition])) {
              continue;
            }
            disjoint[index] = false;
            disjoint[otherIndex] = false;
          }
        }
      }
    }
  }
  return disjoint;
}

namespace {
struct LoopRange {
  Value inductionVariable;
  int64_t lowerBound = 0;
  int64_t upperBound = 0;
  int64_t step = 1;
};
} // namespace

/// Resolve the `scf.for` loops enclosing `op` whose induction variables
/// `evaluator` cannot evaluate, outermost first.
static FailureOr<SmallVector<std::pair<scf::ForOp, LoopRange>>>
resolveEnumeratedLoops(Operation *op, IntegerExpressionEvaluator &evaluator,
                       llvm::function_ref<InFlightDiagnostic()> emitError) {
  SmallVector<scf::ForOp> dynamicLoops;
  for (Operation *ancestor = op->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(ancestor);
    if (loop && !evaluator.evaluate(loop.getInductionVar())) {
      dynamicLoops.push_back(loop);
    }
    if (isa<func::FuncOp>(ancestor)) {
      break;
    }
  }
  std::reverse(dynamicLoops.begin(), dynamicLoops.end());

  SmallVector<std::pair<scf::ForOp, LoopRange>> loops;
  loops.reserve(dynamicLoops.size());
  for (scf::ForOp loop : dynamicLoops) {
    std::optional<llvm::APInt> lowerBound =
        evaluator.evaluate(loop.getLowerBound());
    std::optional<llvm::APInt> upperBound =
        evaluator.evaluate(loop.getUpperBound());
    std::optional<llvm::APInt> step = evaluator.evaluate(loop.getStep());
    if (!lowerBound || !upperBound || !step || !lowerBound->isSignedIntN(64) ||
        !upperBound->isSignedIntN(64) || !step->isSignedIntN(64) ||
        step->getSExtValue() <= 0) {
      if (emitError) {
        emitError() << "pipe receive tensor_slice requires statically "
                       "enumerable enclosing loops";
      }
      return failure();
    }
    loops.push_back(
        {loop, LoopRange{loop.getInductionVar(), lowerBound->getSExtValue(),
                         upperBound->getSExtValue(), step->getSExtValue()}});
  }
  return loops;
}

/// Bind every combination of the induction values of `loops`, outermost
/// first, and call `visit` for each. Stops at the first failure.
static LogicalResult
forEachLoopIteration(ArrayRef<std::pair<scf::ForOp, LoopRange>> loops,
                     llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                     llvm::function_ref<InFlightDiagnostic()> emitError,
                     llvm::function_ref<LogicalResult()> visit) {
  if (loops.empty()) {
    return visit();
  }
  const LoopRange &range = loops.front().second;
  for (int64_t value = range.lowerBound; value < range.upperBound;) {
    inductionValues[range.inductionVariable] = llvm::APInt(
        IndexType::kInternalStorageBitWidth, value, /*isSigned=*/true);
    if (failed(forEachLoopIteration(loops.drop_front(), inductionValues,
                                    emitError, visit))) {
      return failure();
    }
    std::optional<int64_t> nextValue = llvm::checkedAdd(value, range.step);
    if (!nextValue || *nextValue <= value) {
      if (emitError) {
        emitError() << "pipe receive tensor_slice loop enumeration overflowed";
      }
      return failure();
    }
    value = *nextValue;
  }
  inductionValues.erase(range.inductionVariable);
  return success();
}

/// Evaluate the start indices of `slice` for the bound induction values.
static FailureOr<SmallVector<int64_t>>
evaluateSliceStart(TensorSliceOp slice, IntegerExpressionEvaluator &evaluator,
                   const std::string &failureReason,
                   llvm::function_ref<InFlightDiagnostic()> emitError) {
  SmallVector<int64_t> startIndices;
  startIndices.reserve(slice.getIndices().size());
  for (auto [dimension, index] : llvm::enumerate(slice.getIndices())) {
    std::optional<llvm::APInt> resolvedIndex = evaluator.evaluate(index);
    if (!resolvedIndex || !resolvedIndex->isSignedIntN(64)) {
      if (emitError) {
        emitError() << "pipe receive tensor_slice start index in dimension "
                    << dimension << " is not statically enumerable"
                    << (failureReason.empty() ? "" : ": " + failureReason);
      }
      return failure();
    }
    startIndices.push_back(resolvedIndex->getSExtValue());
  }
  return startIndices;
}

/// Return the bound induction values of `loops`, outermost first.
static SmallVector<int64_t> boundInductionValues(
    ArrayRef<std::pair<scf::ForOp, LoopRange>> loops,
    const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
  return llvm::map_to_vector(loops, [&](const auto &loop) {
    return inductionValues.lookup(loop.second.inductionVariable).getSExtValue();
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
  llvm::DenseMap<Value, llvm::APInt> inductionValues;
  std::string failureReason;
  auto evaluateValue = [&](Value value) -> std::optional<llvm::APInt> {
    auto inductionIt = inductionValues.find(value);
    if (inductionIt != inductionValues.end()) {
      return inductionIt->second;
    }
    if (std::optional<llvm::APInt> contextValue =
            evaluateContextValue(value, inductionValues, failureReason)) {
      return contextValue;
    }
    return evaluateIntegerAtLaunchLocation(value, location, state);
  };
  IntegerExpressionEvaluator contextEvaluator(evaluateValue);
  FailureOr<SmallVector<std::pair<scf::ForOp, LoopRange>>> loops =
      resolveEnumeratedLoops(slice, contextEvaluator, emitError);
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
  result.loops =
      llvm::map_to_vector(*loops, [](const auto &loop) { return loop.first; });
  SmallVector<SmallVector<int64_t>> &occurrences = result.startIndices;
  failureReason.clear();
  if (failed(forEachLoopIteration(
          *loops, inductionValues, emitError, [&]() -> LogicalResult {
            IntegerExpressionEvaluator occurrenceEvaluator(evaluateValue);
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
                boundInductionValues(*loops, inductionValues));
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
  auto sliceType = cast<RankedTensorType>(slice.getType());
  int64_t rankDifference = tensorGridShape.size() - sliceType.getRank();
  for (ArrayRef<int64_t> startIndices : occurrences) {
    for (int64_t dimension = 0;
         dimension < static_cast<int64_t>(tensorGridShape.size());
         ++dimension) {
      int64_t regionExtent =
          dimension < rankDifference
              ? 1
              : sliceType.getDimSize(dimension - rankDifference);
      if (startIndices[dimension] < 0 || regionExtent <= 0 ||
          startIndices[dimension] > tensorGridShape[dimension] - regionExtent) {
        return reportFailure("pipe receive tensor_slice region in dimension ",
                             dimension, " starts at ", startIndices[dimension],
                             " with extent ", regionExtent,
                             ", outside the destination tensor tile grid ",
                             tensorGridShape[dimension]);
      }
    }
  }
  return result;
}

FailureOr<TensorSliceOccurrences>
enumerateTensorSliceIterationStarts(TensorSliceOp slice, Operation *user,
                                    const LaunchExecutionLocation &location,
                                    const LaunchNodeDomainState &state,
                                    std::uint64_t maxIterations) {
  llvm::DenseMap<Value, llvm::APInt> inductionValues;
  auto evaluateValue = [&](Value value) -> std::optional<llvm::APInt> {
    auto inductionIt = inductionValues.find(value);
    if (inductionIt != inductionValues.end()) {
      return inductionIt->second;
    }
    return evaluateIntegerAtLaunchLocation(value, location, state);
  };
  IntegerExpressionEvaluator evaluator(evaluateValue);
  FailureOr<SmallVector<std::pair<scf::ForOp, LoopRange>>> loops =
      resolveEnumeratedLoops(user, evaluator, /*emitError=*/{});
  if (failed(loops)) {
    return failure();
  }
  TensorSliceOccurrences result;
  result.loops =
      llvm::map_to_vector(*loops, [](const auto &loop) { return loop.first; });
  std::string failureReason;
  if (failed(forEachLoopIteration(
          *loops, inductionValues, /*emitError=*/{}, [&]() -> LogicalResult {
            if (result.startIndices.size() >= maxIterations) {
              return failure();
            }
            IntegerExpressionEvaluator iterationEvaluator(evaluateValue);
            FailureOr<SmallVector<int64_t>> startIndices = evaluateSliceStart(
                slice, iterationEvaluator, failureReason, /*emitError=*/{});
            if (failed(startIndices)) {
              return failure();
            }
            result.startIndices.push_back(std::move(*startIndices));
            result.inductionValues.push_back(
                boundInductionValues(*loops, inductionValues));
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
