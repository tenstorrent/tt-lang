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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <utility>

namespace mlir::tt {

namespace {

struct ControlFrame {
  Operation *parent = nullptr;
  Region *region = nullptr;
};

class EnumerationBudget {
public:
  explicit EnumerationBudget(std::uint64_t remainingIterations)
      : remainingIterations(remainingIterations) {}

  bool canConsume(std::uint64_t iterationCount) const {
    return iterationCount <= remainingIterations;
  }

  bool tryConsume() {
    if (remainingIterations == 0) {
      return false;
    }
    --remainingIterations;
    return true;
  }

private:
  std::uint64_t remainingIterations;
};

using IntegerEvaluationCache =
    llvm::DenseMap<Value, std::optional<llvm::APInt>>;

struct IntegerEvaluationTask {
  Value value;
  bool operandsReady = false;
};

std::uint32_t getIntegerBitWidth(Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }
  assert(type.isIndex() && "expected an integer or index type");
  return IndexType::kInternalStorageBitWidth;
}

llvm::APInt convertIntegerWidth(llvm::APInt value, Type type, bool isSigned) {
  std::uint32_t width = getIntegerBitWidth(type);
  return isSigned ? value.sextOrTrunc(width) : value.zextOrTrunc(width);
}

bool hasOneLinearBlock(Region &region) {
  if (!region.hasOneBlock() || region.front().empty()) {
    return false;
  }
  Operation &terminator = region.front().back();
  return terminator.hasTrait<OpTrait::IsTerminator>() &&
         terminator.getNumSuccessors() == 0;
}

bool isRegionReachable(RegionBranchOpInterface branch, Region &sourceRegion,
                       Region &targetRegion) {
  SmallVector<Region *> worklist{&sourceRegion};
  llvm::SmallPtrSet<Region *, 4> visited;
  visited.insert(&sourceRegion);
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(*region, successors);
    for (RegionSuccessor successor : successors) {
      Region *successorRegion = successor.getSuccessor();
      if (!successorRegion) {
        continue;
      }
      if (successorRegion == &targetRegion) {
        return true;
      }
      if (visited.insert(successorRegion).second) {
        worklist.push_back(successorRegion);
      }
    }
  }
  return false;
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
    if (!operation) {
      return std::nullopt;
    }
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
    EnumerationBudget enumerationBudget(options.maxEnumeratedIterations);
    std::optional<std::uint64_t> maybeCount =
        countExecutions(frames, 0, inductionValues, enumerationBudget);
    executionCountCache.try_emplace(operation, maybeCount);
    return maybeCount;
  }

private:
  bool collectControlFrames(Operation *operation,
                            SmallVectorImpl<ControlFrame> &frames) const {
    Region *parentRegion = operation->getParentRegion();
    if (!parentRegion || !rootRegion.isAncestor(parentRegion)) {
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

  static bool isSupportedIntegerOperation(Operation *operation) {
    return isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
               arith::ExtUIOp, arith::TruncIOp, arith::AddIOp, arith::SubIOp,
               arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
               arith::CmpIOp>(operation);
  }

  std::optional<llvm::APInt>
  evaluateIntegerOperation(Value result, Operation *operation,
                           const IntegerEvaluationCache &cache) const {
    auto getCached = [&](Value operand) -> std::optional<llvm::APInt> {
      auto cached = cache.find(operand);
      assert(cached != cache.end() && "operand must be evaluated first");
      return cached->second;
    };
    auto evaluateWidthCast = [&](Value input,
                                 bool isSigned) -> std::optional<llvm::APInt> {
      std::optional<llvm::APInt> maybeInput = getCached(input);
      return maybeInput ? std::optional(convertIntegerWidth(
                              *maybeInput, result.getType(), isSigned))
                        : std::nullopt;
    };
    auto evaluateBinary =
        [&](Value lhsValue, Value rhsValue,
            auto binaryOperation) -> std::optional<llvm::APInt> {
      std::optional<llvm::APInt> maybeLhs = getCached(lhsValue);
      std::optional<llvm::APInt> maybeRhs = getCached(rhsValue);
      if (!maybeLhs || !maybeRhs) {
        return std::nullopt;
      }
      std::uint32_t width = getIntegerBitWidth(result.getType());
      return binaryOperation(maybeLhs->sextOrTrunc(width),
                             maybeRhs->sextOrTrunc(width));
    };

    using MaybeInteger = std::optional<llvm::APInt>;
    return llvm::TypeSwitch<Operation *, MaybeInteger>(operation)
        .Case<arith::IndexCastOp, arith::ExtSIOp>([&](auto castOp) {
          return evaluateWidthCast(castOp.getIn(), /*isSigned=*/true);
        })
        .Case<arith::IndexCastUIOp, arith::ExtUIOp>([&](auto castOp) {
          return evaluateWidthCast(castOp.getIn(), /*isSigned=*/false);
        })
        .Case<arith::TruncIOp>([&](arith::TruncIOp castOp) -> MaybeInteger {
          MaybeInteger maybeInput = getCached(castOp.getIn());
          return maybeInput ? MaybeInteger(maybeInput->trunc(
                                  getIntegerBitWidth(result.getType())))
                            : std::nullopt;
        })
        .Case<arith::AddIOp>([&](arith::AddIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs + rhs; });
        })
        .Case<arith::SubIOp>([&](arith::SubIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs - rhs; });
        })
        .Case<arith::MulIOp>([&](arith::MulIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs * rhs; });
        })
        .Case<arith::AndIOp>([&](arith::AndIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs & rhs; });
        })
        .Case<arith::OrIOp>([&](arith::OrIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs | rhs; });
        })
        .Case<arith::XOrIOp>([&](arith::XOrIOp binaryOp) {
          return evaluateBinary(
              binaryOp.getLhs(), binaryOp.getRhs(),
              [](llvm::APInt lhs, llvm::APInt rhs) { return lhs ^ rhs; });
        })
        .Case<arith::CmpIOp>([&](arith::CmpIOp compareOp) -> MaybeInteger {
          MaybeInteger maybeLhs = getCached(compareOp.getLhs());
          MaybeInteger maybeRhs = getCached(compareOp.getRhs());
          if (!maybeLhs || !maybeRhs) {
            return std::nullopt;
          }
          bool comparisonResult = false;
          switch (compareOp.getPredicate()) {
          case arith::CmpIPredicate::eq:
            comparisonResult = *maybeLhs == *maybeRhs;
            break;
          case arith::CmpIPredicate::ne:
            comparisonResult = *maybeLhs != *maybeRhs;
            break;
          case arith::CmpIPredicate::slt:
            comparisonResult = maybeLhs->slt(*maybeRhs);
            break;
          case arith::CmpIPredicate::sle:
            comparisonResult = maybeLhs->sle(*maybeRhs);
            break;
          case arith::CmpIPredicate::sgt:
            comparisonResult = maybeLhs->sgt(*maybeRhs);
            break;
          case arith::CmpIPredicate::sge:
            comparisonResult = maybeLhs->sge(*maybeRhs);
            break;
          case arith::CmpIPredicate::ult:
            comparisonResult = maybeLhs->ult(*maybeRhs);
            break;
          case arith::CmpIPredicate::ule:
            comparisonResult = maybeLhs->ule(*maybeRhs);
            break;
          case arith::CmpIPredicate::ugt:
            comparisonResult = maybeLhs->ugt(*maybeRhs);
            break;
          case arith::CmpIPredicate::uge:
            comparisonResult = maybeLhs->uge(*maybeRhs);
            break;
          }
          return llvm::APInt(/*numBits=*/1, comparisonResult);
        })
        .Default([](Operation *) -> MaybeInteger { return std::nullopt; });
  }

  std::optional<llvm::APInt>
  evaluateInteger(Value value,
                  const llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                  IntegerEvaluationCache &cache) const {
    SmallVector<IntegerEvaluationTask> worklist{{value, false}};
    llvm::DenseSet<Value> activeValues;
    while (!worklist.empty()) {
      IntegerEvaluationTask task = worklist.pop_back_val();
      if (cache.contains(task.value)) {
        if (task.operandsReady) {
          activeValues.erase(task.value);
        }
        continue;
      }

      if (task.operandsReady) {
        activeValues.erase(task.value);
        Operation *operation = task.value.getDefiningOp();
        cache.try_emplace(
            task.value, evaluateIntegerOperation(task.value, operation, cache));
        continue;
      }

      if (auto known = inductionValues.find(task.value);
          known != inductionValues.end()) {
        cache.try_emplace(task.value, known->second);
        continue;
      }

      Attribute constant;
      if (matchPattern(task.value, m_Constant(&constant))) {
        if (auto integer = dyn_cast_or_null<IntegerAttr>(constant)) {
          cache.try_emplace(task.value, integer.getValue());
          continue;
        }
      }
      if (symbolValueEvaluator) {
        if (std::optional<llvm::APInt> maybeSymbol =
                symbolValueEvaluator(task.value)) {
          cache.try_emplace(task.value,
                            maybeSymbol->getBitWidth() ==
                                    getIntegerBitWidth(task.value.getType())
                                ? maybeSymbol
                                : std::nullopt);
          continue;
        }
      }

      Operation *operation = task.value.getDefiningOp();
      if (!operation || operation->getNumResults() != 1 ||
          !isSupportedIntegerOperation(operation)) {
        cache.try_emplace(task.value, std::nullopt);
        continue;
      }

      if (!activeValues.insert(task.value).second) {
        cache.try_emplace(task.value, std::nullopt);
        continue;
      }
      worklist.push_back({task.value, true});
      for (Value operand : llvm::reverse(operation->getOperands())) {
        worklist.push_back({operand, false});
      }
    }

    auto result = cache.find(value);
    assert(result != cache.end() && "evaluation must cache its result");
    return result->second;
  }

  std::optional<std::uint64_t> getExactRegionInvocationCount(
      ControlFrame frame,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    if (regionInvocationCountEvaluator) {
      if (std::optional<std::uint64_t> maybeCount =
              regionInvocationCountEvaluator(*frame.region)) {
        return maybeCount;
      }
    }

    auto branch = dyn_cast<RegionBranchOpInterface>(frame.parent);
    if (!branch) {
      return std::nullopt;
    }

    SmallVector<Attribute> operands;
    operands.reserve(frame.parent->getNumOperands());
    IntegerEvaluationCache evaluationCache;
    for (Value operand : frame.parent->getOperands()) {
      std::optional<llvm::APInt> maybeInteger =
          operand.getType().isIntOrIndex()
              ? evaluateInteger(operand, inductionValues, evaluationCache)
              : std::nullopt;
      if (maybeInteger) {
        operands.push_back(IntegerAttr::get(operand.getType(), *maybeInteger));
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
    auto maybeUpperBound = regionBounds.getUpperBound();
    SmallVector<RegionSuccessor> entrySuccessors;
    branch.getEntrySuccessorRegions(operands, entrySuccessors);
    if (entrySuccessors.size() == 1) {
      Region *selectedRegion = entrySuccessors.front().getSuccessor();
      if (!selectedRegion) {
        return 0;
      }
      if (selectedRegion != frame.region) {
        if (maybeUpperBound && *maybeUpperBound == 0) {
          return 0;
        }
        if (!isRegionReachable(branch, *selectedRegion, *frame.region)) {
          return 0;
        }
        return std::nullopt;
      }
      if (maybeUpperBound && *maybeUpperBound == 1) {
        return 1;
      }

      SmallVector<RegionSuccessor> regionSuccessors;
      branch.getSuccessorRegions(*selectedRegion, regionSuccessors);
      if (llvm::all_of(regionSuccessors, [](RegionSuccessor successor) {
            return successor.isOperation();
          })) {
        return 1;
      }
      return std::nullopt;
    }

    if (!maybeUpperBound || regionBounds.getLowerBound() != *maybeUpperBound) {
      return std::nullopt;
    }
    return *maybeUpperBound;
  }

  std::optional<llvm::APInt> getSCFForTripCount(
      scf::ForOp forOp,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    if (std::optional<llvm::APInt> maybeTripCount =
            forOp.getStaticTripCount()) {
      return maybeTripCount;
    }

    IntegerEvaluationCache evaluationCache;
    std::optional<llvm::APInt> maybeLowerBound = evaluateInteger(
        forOp.getLowerBound(), inductionValues, evaluationCache);
    std::optional<llvm::APInt> maybeUpperBound = evaluateInteger(
        forOp.getUpperBound(), inductionValues, evaluationCache);
    std::optional<llvm::APInt> maybeStep =
        evaluateInteger(forOp.getStep(), inductionValues, evaluationCache);
    if (!maybeLowerBound || !maybeUpperBound || !maybeStep ||
        maybeStep->isZero()) {
      return std::nullopt;
    }

    IntegerAttr lowerBoundAttr =
        IntegerAttr::get(forOp.getLowerBound().getType(), *maybeLowerBound);
    IntegerAttr upperBoundAttr =
        IntegerAttr::get(forOp.getUpperBound().getType(), *maybeUpperBound);
    IntegerAttr stepAttr =
        IntegerAttr::get(forOp.getStep().getType(), *maybeStep);
    return constantTripCount(lowerBoundAttr, upperBoundAttr, stepAttr,
                             /*isSigned=*/!forOp.getUnsignedCmp(),
                             scf::computeUbMinusLb);
  }

  std::optional<std::uint64_t> getLoopTripCount(
      LoopLikeOpInterface loop,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    std::optional<llvm::APInt> maybeTripCount = loop.getStaticTripCount();
    if (!maybeTripCount) {
      if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
        maybeTripCount = getSCFForTripCount(forOp, inductionValues);
      }
    }
    if (!maybeTripCount || maybeTripCount->getActiveBits() > 64) {
      return std::nullopt;
    }
    return maybeTripCount->getZExtValue();
  }

  std::optional<std::uint64_t>
  countExecutions(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                  llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                  EnumerationBudget &enumerationBudget) const {
    if (frameIndex == frames.size()) {
      return 1;
    }

    ControlFrame frame = frames[frameIndex];
    auto loop = dyn_cast<LoopLikeOpInterface>(frame.parent);
    SmallVector<Region *> loopRegions;
    if (loop) {
      loopRegions = loop.getLoopRegions();
    }
    if (!loop || !llvm::is_contained(loopRegions, frame.region)) {
      std::optional<std::uint64_t> maybeInvocationCount =
          getExactRegionInvocationCount(frame, inductionValues);
      if (!maybeInvocationCount) {
        return std::nullopt;
      }
      if (*maybeInvocationCount == 0) {
        return 0;
      }
      std::optional<std::uint64_t> maybeNestedCount = countExecutions(
          frames, frameIndex + 1, inductionValues, enumerationBudget);
      return maybeNestedCount ? llvm::checkedMulUnsigned(*maybeInvocationCount,
                                                         *maybeNestedCount)
                              : std::nullopt;
    }

    std::optional<std::uint64_t> maybeTripCount =
        getLoopTripCount(loop, inductionValues);
    if (!maybeTripCount) {
      return std::nullopt;
    }

    // A trip count does not define the invocation count of each region in a
    // multi-region loop.
    if (loopRegions.size() != 1) {
      return std::nullopt;
    }
    if (*maybeTripCount == 0) {
      return 0;
    }

    // Try multiplication before enumeration. Passing the same budget charges
    // nested enumeration even when this attempt cannot prove a count.
    std::optional<std::uint64_t> maybeNestedCount = countExecutions(
        frames, frameIndex + 1, inductionValues, enumerationBudget);
    if (maybeNestedCount) {
      return llvm::checkedMulUnsigned(*maybeTripCount, *maybeNestedCount);
    }

    auto forOp = dyn_cast<scf::ForOp>(frame.parent);
    if (!forOp || !enumerationBudget.canConsume(*maybeTripCount)) {
      return std::nullopt;
    }
    IntegerEvaluationCache evaluationCache;
    std::optional<llvm::APInt> maybeInductionValue = evaluateInteger(
        forOp.getLowerBound(), inductionValues, evaluationCache);
    std::optional<llvm::APInt> maybeStep =
        evaluateInteger(forOp.getStep(), inductionValues, evaluationCache);
    if (!maybeInductionValue || !maybeStep) {
      return std::nullopt;
    }

    std::uint32_t inductionWidth =
        getIntegerBitWidth(forOp.getInductionVar().getType());
    *maybeInductionValue = maybeInductionValue->sextOrTrunc(inductionWidth);
    *maybeStep = maybeStep->sextOrTrunc(inductionWidth);
    llvm::scope_exit restoreInductionValue(
        [&] { inductionValues.erase(forOp.getInductionVar()); });
    std::uint64_t total = 0;
    for (std::uint64_t iteration = 0; iteration < *maybeTripCount;
         ++iteration) {
      if (!enumerationBudget.tryConsume()) {
        return std::nullopt;
      }
      inductionValues[forOp.getInductionVar()] = *maybeInductionValue;
      maybeNestedCount = countExecutions(frames, frameIndex + 1,
                                         inductionValues, enumerationBudget);
      if (!maybeNestedCount) {
        return std::nullopt;
      }
      std::optional<std::uint64_t> maybeNextTotal =
          llvm::checkedAddUnsigned(total, *maybeNestedCount);
      if (!maybeNextTotal) {
        return std::nullopt;
      }
      total = *maybeNextTotal;
      *maybeInductionValue += *maybeStep;
    }
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
    RegionInvocationCountEvaluator regionInvocationCountEvaluator)
    : ExecutionCountAnalysis(rootRegion, std::move(symbolValueEvaluator),
                             std::move(regionInvocationCountEvaluator),
                             Options{}) {}

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
