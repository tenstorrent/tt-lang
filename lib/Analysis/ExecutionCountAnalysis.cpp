// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ExecutionCountAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace mlir::tt {

namespace {

struct ControlFrame {
  Operation *parent = nullptr;
  Region *region = nullptr;
};

static std::uint32_t getIntegerBitWidth(Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }
  assert(type.isIndex() && "expected an integer or index type");
  return 64;
}

static llvm::APInt convertIntegerWidth(llvm::APInt value, Type type,
                                       bool isSigned) {
  std::uint32_t width = getIntegerBitWidth(type);
  return isSigned ? value.sextOrTrunc(width) : value.zextOrTrunc(width);
}

static std::optional<std::uint64_t> checkedMultiply(std::uint64_t lhs,
                                                    std::uint64_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<std::uint64_t> checkedAdd(std::uint64_t lhs,
                                               std::uint64_t rhs) {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static bool hasOneLinearBlock(Region &region) {
  if (!region.hasOneBlock() || region.front().empty()) {
    return false;
  }
  Operation &terminator = region.front().back();
  return terminator.hasTrait<OpTrait::IsTerminator>() &&
         terminator.getNumSuccessors() == 0;
}

} // namespace

class ExecutionCountAnalysis::Impl {
public:
  Impl(Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
       RegionInvocationCountEvaluator regionInvocationCountEvaluator,
       Options options)
      : rootRegion(rootRegion),
        symbolValueEvaluator(std::move(symbolValueEvaluator)),
        regionInvocationCountEvaluator(
            std::move(regionInvocationCountEvaluator)),
        options(options) {}

  std::optional<std::uint64_t> getExecutionCount(Operation *operation) {
    auto cached = executionCountCache.find(operation);
    if (cached != executionCountCache.end()) {
      return cached->second;
    }

    SmallVector<ControlFrame> frames;
    if (!collectControlFrames(operation, frames)) {
      executionCountCache.try_emplace(operation, std::nullopt);
      return std::nullopt;
    }

    llvm::DenseMap<Value, llvm::APInt> inductionValues;
    std::uint64_t remainingIterations = options.maxEnumeratedIterations;
    std::optional<std::uint64_t> count =
        countExecutions(frames, 0, inductionValues, remainingIterations);
    executionCountCache.try_emplace(operation, count);
    return count;
  }

private:
  bool collectControlFrames(Operation *operation,
                            SmallVectorImpl<ControlFrame> &frames) const {
    if (!rootRegion.isAncestor(operation->getParentRegion())) {
      return false;
    }

    for (Operation *child = operation;
         child->getParentRegion() != &rootRegion;) {
      Region *region = child->getParentRegion();
      if (!hasOneLinearBlock(*region)) {
        return false;
      }
      Operation *parent = region->getParentOp();
      if (!parent) {
        return false;
      }
      frames.push_back({parent, region});
      child = parent;
    }
    if (!hasOneLinearBlock(rootRegion)) {
      return false;
    }
    std::reverse(frames.begin(), frames.end());
    return true;
  }

  std::optional<llvm::APInt>
  evaluateInteger(Value value,
                  const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
    if (auto known = inductionValues.find(value);
        known != inductionValues.end()) {
      return known->second;
    }

    Attribute constant;
    if (matchPattern(value, m_Constant(&constant))) {
      if (auto integer = dyn_cast_or_null<IntegerAttr>(constant)) {
        return integer.getValue();
      }
    }
    if (symbolValueEvaluator) {
      if (std::optional<llvm::APInt> symbol = symbolValueEvaluator(value)) {
        return convertIntegerWidth(*symbol, value.getType(),
                                   /*isSigned=*/true);
      }
    }

    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getNumResults() != 1) {
      return std::nullopt;
    }

    auto evaluateBinary = [&](Value lhsValue, Value rhsValue,
                              auto operation) -> std::optional<llvm::APInt> {
      std::optional<llvm::APInt> lhs =
          evaluateInteger(lhsValue, inductionValues);
      std::optional<llvm::APInt> rhs =
          evaluateInteger(rhsValue, inductionValues);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      std::uint32_t width = getIntegerBitWidth(value.getType());
      return operation(lhs->sextOrTrunc(width), rhs->sextOrTrunc(width));
    };

    if (auto castOp = dyn_cast<arith::IndexCastOp>(definingOp)) {
      std::optional<llvm::APInt> input =
          evaluateInteger(castOp.getIn(), inductionValues);
      return input ? std::optional(convertIntegerWidth(*input, value.getType(),
                                                       /*isSigned=*/true))
                   : std::nullopt;
    }
    if (auto castOp = dyn_cast<arith::IndexCastUIOp>(definingOp)) {
      std::optional<llvm::APInt> input =
          evaluateInteger(castOp.getIn(), inductionValues);
      return input ? std::optional(convertIntegerWidth(*input, value.getType(),
                                                       /*isSigned=*/false))
                   : std::nullopt;
    }
    if (auto castOp = dyn_cast<arith::ExtSIOp>(definingOp)) {
      std::optional<llvm::APInt> input =
          evaluateInteger(castOp.getIn(), inductionValues);
      return input ? std::optional(convertIntegerWidth(*input, value.getType(),
                                                       /*isSigned=*/true))
                   : std::nullopt;
    }
    if (auto castOp = dyn_cast<arith::ExtUIOp>(definingOp)) {
      std::optional<llvm::APInt> input =
          evaluateInteger(castOp.getIn(), inductionValues);
      return input ? std::optional(convertIntegerWidth(*input, value.getType(),
                                                       /*isSigned=*/false))
                   : std::nullopt;
    }
    if (auto castOp = dyn_cast<arith::TruncIOp>(definingOp)) {
      std::optional<llvm::APInt> input =
          evaluateInteger(castOp.getIn(), inductionValues);
      return input ? std::optional(
                         input->trunc(getIntegerBitWidth(value.getType())))
                   : std::nullopt;
    }
    if (auto addOp = dyn_cast<arith::AddIOp>(definingOp)) {
      return evaluateBinary(
          addOp.getLhs(), addOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs + rhs; });
    }
    if (auto subOp = dyn_cast<arith::SubIOp>(definingOp)) {
      return evaluateBinary(
          subOp.getLhs(), subOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs - rhs; });
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(definingOp)) {
      return evaluateBinary(
          mulOp.getLhs(), mulOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs * rhs; });
    }
    if (auto andOp = dyn_cast<arith::AndIOp>(definingOp)) {
      return evaluateBinary(
          andOp.getLhs(), andOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs & rhs; });
    }
    if (auto orOp = dyn_cast<arith::OrIOp>(definingOp)) {
      return evaluateBinary(
          orOp.getLhs(), orOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs | rhs; });
    }
    if (auto xorOp = dyn_cast<arith::XOrIOp>(definingOp)) {
      return evaluateBinary(
          xorOp.getLhs(), xorOp.getRhs(),
          [](llvm::APInt lhs, llvm::APInt rhs) { return lhs ^ rhs; });
    }
    if (auto cmpOp = dyn_cast<arith::CmpIOp>(definingOp)) {
      std::optional<llvm::APInt> lhs =
          evaluateInteger(cmpOp.getLhs(), inductionValues);
      std::optional<llvm::APInt> rhs =
          evaluateInteger(cmpOp.getRhs(), inductionValues);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      bool result = false;
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        result = *lhs == *rhs;
        break;
      case arith::CmpIPredicate::ne:
        result = *lhs != *rhs;
        break;
      case arith::CmpIPredicate::slt:
        result = lhs->slt(*rhs);
        break;
      case arith::CmpIPredicate::sle:
        result = lhs->sle(*rhs);
        break;
      case arith::CmpIPredicate::sgt:
        result = lhs->sgt(*rhs);
        break;
      case arith::CmpIPredicate::sge:
        result = lhs->sge(*rhs);
        break;
      case arith::CmpIPredicate::ult:
        result = lhs->ult(*rhs);
        break;
      case arith::CmpIPredicate::ule:
        result = lhs->ule(*rhs);
        break;
      case arith::CmpIPredicate::ugt:
        result = lhs->ugt(*rhs);
        break;
      case arith::CmpIPredicate::uge:
        result = lhs->uge(*rhs);
        break;
      }
      return llvm::APInt(/*numBits=*/1, result);
    }
    return std::nullopt;
  }

  std::optional<std::uint64_t> getExactRegionInvocationCount(
      ControlFrame frame,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
    if (regionInvocationCountEvaluator) {
      if (std::optional<std::uint64_t> count =
              regionInvocationCountEvaluator(*frame.region)) {
        return count;
      }
    }

    auto branch = dyn_cast<RegionBranchOpInterface>(frame.parent);
    if (!branch) {
      return std::nullopt;
    }

    SmallVector<Attribute> operands;
    operands.reserve(frame.parent->getNumOperands());
    for (Value operand : frame.parent->getOperands()) {
      std::optional<llvm::APInt> integer =
          operand.getType().isIntOrIndex()
              ? evaluateInteger(operand, inductionValues)
              : std::nullopt;
      if (integer) {
        operands.push_back(IntegerAttr::get(operand.getType(), *integer));
        continue;
      }
      Attribute constant;
      operands.push_back(matchPattern(operand, m_Constant(&constant))
                             ? constant
                             : Attribute());
    }

    SmallVector<InvocationBounds> bounds;
    branch.getRegionInvocationBounds(operands, bounds);
    std::size_t regionNumber = frame.region->getRegionNumber();
    if (regionNumber >= bounds.size()) {
      return std::nullopt;
    }
    const InvocationBounds &regionBounds = bounds[regionNumber];
    auto upperBound = regionBounds.getUpperBound();
    SmallVector<RegionSuccessor> entrySuccessors;
    branch.getEntrySuccessorRegions(operands, entrySuccessors);
    if (entrySuccessors.size() == 1) {
      Region *selectedRegion = entrySuccessors.front().getSuccessor();
      if (!selectedRegion) {
        return 0;
      }
      if (selectedRegion != frame.region) {
        if (upperBound && regionBounds.getLowerBound() == 0 &&
            *upperBound == 0) {
          return 0;
        }
        return std::nullopt;
      }
      if (upperBound && *upperBound == 1) {
        return 1;
      }

      SmallVector<RegionSuccessor> regionSuccessors;
      branch.getSuccessorRegions(*selectedRegion, regionSuccessors);
      if (llvm::all_of(regionSuccessors,
                       [](RegionSuccessor successor) {
                         return successor.isOperation();
                       })) {
        return 1;
      }
      return std::nullopt;
    }

    if (!upperBound || regionBounds.getLowerBound() != *upperBound) {
      return std::nullopt;
    }
    return *upperBound;
  }

  std::optional<llvm::APInt> getSCFForTripCount(
      scf::ForOp forOp,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
    if (std::optional<llvm::APInt> tripCount = forOp.getStaticTripCount()) {
      return tripCount;
    }

    std::optional<llvm::APInt> lowerBound =
        evaluateInteger(forOp.getLowerBound(), inductionValues);
    std::optional<llvm::APInt> upperBound =
        evaluateInteger(forOp.getUpperBound(), inductionValues);
    std::optional<llvm::APInt> step =
        evaluateInteger(forOp.getStep(), inductionValues);
    if (!lowerBound || !upperBound || !step || step->isZero()) {
      return std::nullopt;
    }

    IntegerAttr lowerBoundAttr =
        IntegerAttr::get(forOp.getLowerBound().getType(), *lowerBound);
    IntegerAttr upperBoundAttr =
        IntegerAttr::get(forOp.getUpperBound().getType(), *upperBound);
    IntegerAttr stepAttr = IntegerAttr::get(forOp.getStep().getType(), *step);
    return constantTripCount(lowerBoundAttr, upperBoundAttr, stepAttr,
                             /*isSigned=*/!forOp.getUnsignedCmp(),
                             scf::computeUbMinusLb);
  }

  std::optional<std::uint64_t>
  getLoopTripCount(LoopLikeOpInterface loop,
                   const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
    std::optional<llvm::APInt> tripCount = loop.getStaticTripCount();
    if (!tripCount) {
      if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
        tripCount = getSCFForTripCount(forOp, inductionValues);
      }
    }
    if (!tripCount || tripCount->getActiveBits() > 64) {
      return std::nullopt;
    }
    return tripCount->getZExtValue();
  }

  std::optional<std::uint64_t>
  countExecutions(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                  llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                  std::uint64_t &remainingIterations) {
    if (frameIndex == frames.size()) {
      return 1;
    }

    ControlFrame frame = frames[frameIndex];
    auto loop = dyn_cast<LoopLikeOpInterface>(frame.parent);
    if (!loop || !llvm::is_contained(loop.getLoopRegions(), frame.region)) {
      std::optional<std::uint64_t> invocationCount =
          getExactRegionInvocationCount(frame, inductionValues);
      if (!invocationCount) {
        return std::nullopt;
      }
      if (*invocationCount == 0) {
        return 0;
      }
      std::optional<std::uint64_t> nestedCount = countExecutions(
          frames, frameIndex + 1, inductionValues, remainingIterations);
      return nestedCount ? checkedMultiply(*invocationCount, *nestedCount)
                         : std::nullopt;
    }

    std::optional<std::uint64_t> tripCount =
        getLoopTripCount(loop, inductionValues);
    if (!tripCount) {
      return std::nullopt;
    }
    if (*tripCount == 0) {
      return 0;
    }

    // Multiplication avoids enumerating loops when nested control flow and
    // inner trip counts do not depend on this loop's induction variable.
    std::uint64_t independentProofBudget = remainingIterations;
    if (std::optional<std::uint64_t> nestedCount = countExecutions(
            frames, frameIndex + 1, inductionValues, independentProofBudget)) {
      remainingIterations = independentProofBudget;
      return checkedMultiply(*tripCount, *nestedCount);
    }

    auto forOp = dyn_cast<scf::ForOp>(frame.parent);
    if (!forOp || *tripCount > remainingIterations) {
      return std::nullopt;
    }
    std::optional<llvm::APInt> inductionValue =
        evaluateInteger(forOp.getLowerBound(), inductionValues);
    std::optional<llvm::APInt> step =
        evaluateInteger(forOp.getStep(), inductionValues);
    if (!inductionValue || !step) {
      return std::nullopt;
    }

    std::uint32_t inductionWidth =
        getIntegerBitWidth(forOp.getInductionVar().getType());
    *inductionValue = inductionValue->sextOrTrunc(inductionWidth);
    *step = step->sextOrTrunc(inductionWidth);
    std::uint64_t total = 0;
    for (std::uint64_t iteration = 0; iteration < *tripCount; ++iteration) {
      --remainingIterations;
      inductionValues[forOp.getInductionVar()] = *inductionValue;
      std::optional<std::uint64_t> nestedCount = countExecutions(
          frames, frameIndex + 1, inductionValues, remainingIterations);
      if (!nestedCount) {
        inductionValues.erase(forOp.getInductionVar());
        return std::nullopt;
      }
      std::optional<std::uint64_t> nextTotal = checkedAdd(total, *nestedCount);
      if (!nextTotal) {
        inductionValues.erase(forOp.getInductionVar());
        return std::nullopt;
      }
      total = *nextTotal;
      *inductionValue += *step;
    }
    inductionValues.erase(forOp.getInductionVar());
    return total;
  }

  Region &rootRegion;
  SymbolValueEvaluator symbolValueEvaluator;
  RegionInvocationCountEvaluator regionInvocationCountEvaluator;
  Options options;
  llvm::DenseMap<Operation *, std::optional<std::uint64_t>> executionCountCache;
};

ExecutionCountAnalysis::ExecutionCountAnalysis(
    Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
    RegionInvocationCountEvaluator regionInvocationCountEvaluator,
    Options options)
    : impl(std::make_unique<Impl>(rootRegion, std::move(symbolValueEvaluator),
                                  std::move(regionInvocationCountEvaluator),
                                  options)) {}

ExecutionCountAnalysis::~ExecutionCountAnalysis() = default;

ExecutionCountAnalysis::ExecutionCountAnalysis(
    ExecutionCountAnalysis &&) noexcept = default;

ExecutionCountAnalysis &
ExecutionCountAnalysis::operator=(ExecutionCountAnalysis &&) noexcept = default;

std::optional<std::uint64_t>
ExecutionCountAnalysis::getExecutionCount(Operation *operation) {
  return impl->getExecutionCount(operation);
}

} // namespace mlir::tt
