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

#include <algorithm>
#include <functional>
#include <utility>

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

TensorRegionBounds
TensorRegionOccurrences::getBounds(ArrayRef<int64_t> occurrenceStart) const {
  int64_t rankDifference = tensorGridShape.size() - sliceShape.size();
  SmallVector<int64_t> extents(tensorGridShape.size(), 1);
  for (int64_t dimension = rankDifference;
       dimension < static_cast<int64_t>(tensorGridShape.size()); ++dimension) {
    extents[dimension] = sliceShape[dimension - rankDifference];
  }
  return TensorRegionBounds{
      globalTensorIndex, SmallVector<int64_t>(tensorGridShape),
      SmallVector<int64_t>(occurrenceStart), std::move(extents)};
}

bool hasOverlappingTensorRegionOccurrences(
    const TensorRegionOccurrences &region) {
  for (std::size_t lhsIndex = 0; lhsIndex < region.startIndices.size();
       ++lhsIndex) {
    TensorRegionBounds lhs = region.getBounds(region.startIndices[lhsIndex]);
    for (std::size_t rhsIndex = lhsIndex + 1;
         rhsIndex < region.startIndices.size(); ++rhsIndex) {
      if (tensorRegionsOverlap(
              lhs, region.getBounds(region.startIndices[rhsIndex]))) {
        return true;
      }
    }
  }
  return false;
}

bool tensorRegionOccurrencesOverlap(const TensorRegionOccurrences &lhs,
                                    const TensorRegionOccurrences &rhs) {
  return llvm::any_of(lhs.startIndices, [&](ArrayRef<int64_t> lhsStart) {
    TensorRegionBounds lhsBounds = lhs.getBounds(lhsStart);
    return llvm::any_of(rhs.startIndices, [&](ArrayRef<int64_t> rhsStart) {
      return tensorRegionsOverlap(lhsBounds, rhs.getBounds(rhsStart));
    });
  });
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
  for (std::size_t lhsIndex = 0; lhsIndex < destinations.size(); ++lhsIndex) {
    for (std::size_t rhsIndex = lhsIndex + 1; rhsIndex < destinations.size();
         ++rhsIndex) {
      if (tensorRegionDestinationsMayAlias(destinations[lhsIndex],
                                           destinations[rhsIndex])) {
        disjoint[lhsIndex] = false;
        disjoint[rhsIndex] = false;
      }
    }
  }
  return disjoint;
}

FailureOr<SmallVector<SmallVector<int64_t>>> enumerateTensorSliceOccurrences(
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

  SmallVector<scf::ForOp> dynamicLoops;
  for (Operation *ancestor = slice->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(ancestor);
    if (loop && !contextEvaluator.evaluate(loop.getInductionVar())) {
      dynamicLoops.push_back(loop);
    }
    if (isa<func::FuncOp>(ancestor)) {
      break;
    }
  }
  std::reverse(dynamicLoops.begin(), dynamicLoops.end());

  struct LoopRange {
    Value inductionVariable;
    int64_t lowerBound = 0;
    int64_t upperBound = 0;
    int64_t step = 1;
  };
  SmallVector<LoopRange> loopRanges;
  loopRanges.reserve(dynamicLoops.size());
  for (scf::ForOp loop : dynamicLoops) {
    std::optional<llvm::APInt> lowerBound =
        contextEvaluator.evaluate(loop.getLowerBound());
    std::optional<llvm::APInt> upperBound =
        contextEvaluator.evaluate(loop.getUpperBound());
    std::optional<llvm::APInt> step = contextEvaluator.evaluate(loop.getStep());
    if (!lowerBound || !upperBound || !step || !lowerBound->isSignedIntN(64) ||
        !upperBound->isSignedIntN(64) || !step->isSignedIntN(64) ||
        step->getSExtValue() <= 0) {
      return reportFailure("pipe receive tensor_slice requires statically "
                           "enumerable enclosing loops");
    }
    loopRanges.push_back(
        LoopRange{loop.getInductionVar(), lowerBound->getSExtValue(),
                  upperBound->getSExtValue(), step->getSExtValue()});
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

  SmallVector<SmallVector<int64_t>> occurrences;
  failureReason.clear();
  std::function<LogicalResult(std::size_t)> enumerate =
      [&](std::size_t loopIndex) -> LogicalResult {
    if (loopIndex != loopRanges.size()) {
      const LoopRange &range = loopRanges[loopIndex];
      for (int64_t value = range.lowerBound; value < range.upperBound;) {
        inductionValues[range.inductionVariable] = llvm::APInt(
            IndexType::kInternalStorageBitWidth, value, /*isSigned=*/true);
        if (failed(enumerate(loopIndex + 1))) {
          return failure();
        }
        std::optional<int64_t> nextValue = llvm::checkedAdd(value, range.step);
        if (!nextValue || *nextValue <= value) {
          return reportFailure(
              "pipe receive tensor_slice loop enumeration overflowed");
        }
        value = *nextValue;
      }
      inductionValues.erase(range.inductionVariable);
      return success();
    }

    IntegerExpressionEvaluator occurrenceEvaluator(evaluateValue);
    for (auto [branch, selectedRegion] : enclosingBranches) {
      std::optional<llvm::APInt> condition =
          occurrenceEvaluator.evaluate(branch.getCondition());
      if (!condition || condition->getBitWidth() != 1) {
        return reportFailure("pipe receive tensor_slice requires statically "
                             "enumerable enclosing conditions");
      }
      if (condition->getBoolValue() != (selectedRegion == 0)) {
        return success();
      }
    }

    SmallVector<int64_t> startIndices;
    startIndices.reserve(slice.getIndices().size());
    for (auto [dimension, index] : llvm::enumerate(slice.getIndices())) {
      std::optional<llvm::APInt> resolvedIndex =
          occurrenceEvaluator.evaluate(index);
      if (!resolvedIndex || !resolvedIndex->isSignedIntN(64)) {
        return reportFailure(
            "pipe receive tensor_slice start index in dimension ", dimension,
            " is not statically enumerable",
            failureReason.empty() ? "" : ": " + failureReason);
      }
      startIndices.push_back(resolvedIndex->getSExtValue());
    }
    if (occurrences.size() >= expectedExecutionCount) {
      return reportFailure("pipe receive tensor_slice enumeration exceeds its "
                           "proven transfer count");
    }
    occurrences.push_back(std::move(startIndices));
    return success();
  };
  if (failed(enumerate(0))) {
    return failure();
  }
  if (dynamicLoops.empty() && occurrences.size() == 1) {
    occurrences.resize(expectedExecutionCount, occurrences.front());
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
  return occurrences;
}

bool canOmitFabricReceiverRendezvous(bool isDeviceTransfer, bool isPointToPoint,
                                     bool hasSingleReceiver,
                                     bool hasDisjointTensorRegionDestination) {
  return isDeviceTransfer && isPointToPoint && hasSingleReceiver &&
         hasDisjointTensorRegionDestination;
}

} // namespace mlir::tt::ttl
