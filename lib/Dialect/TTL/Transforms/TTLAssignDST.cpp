// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL DST Register Assignment Pass
//===----------------------------------------------------------------------===//
//
// This pass performs DST (destination) register assignment for ttl.compute
// operations using interval-based linear scan allocation with unary operation
// merging. The algorithm is based on docs/development/DST_Allocation.md:
//
// Phase 1: Copy Insertion
//   - For values with multiple consumers where any consumer is unary
//   - Insert ttl.copy_dst for all but the last consumer
//   - Prevents unary ops from clobbering values needed by other consumers
//
// Phase 2: Build Live Intervals with Unary Merging
//   - Assign operation indices in block order
//   - Build lifetime intervals [start, end] for each tile value
//   - Merge intervals for unary ops (input and output share DST)
//   - Use union-find to track merged equivalence classes
//
// Phase 3: Linear Scan Allocation
//   - Process intervals by start position (Wimmer & Franz, CGO'10)
//   - Expire intervals when their last use passes
//   - Reuse freed registers for new values
//   - Optional: Separate output region (--separate-output-region flag)
//
// This pass also inserts ttl.copy_tile ops for block arguments and assigns
// dst_idx attributes to all tile compute operations.
//
// Testing: LLVM_DEBUG messages are used extensively for lit test verification.
// Tests use -debug-only=ttl-assign-dst to check intervals, allocations, and
// phase transitions (see *_debug.mlir tests).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <limits>

#define DEBUG_TYPE "ttl-assign-dst"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLASSIGNDST
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Default DST capacity (16-bit, double-buffered).
constexpr std::uint32_t kDefaultDSTCapacity = 8;

/// Sentinel value for placeholder copy_tile indices. Using max value since
/// valid indices are small non-negative integers. This is replaced with proper
/// indices during the copy_tile insertion phase.
constexpr int64_t kPlaceholderIndex = std::numeric_limits<int64_t>::max();

/// Compute DST capacity based on operation types and device config.
/// TODO(#150): Implement dynamic capacity based on datatype and device.
/// Current: Returns default capacity (8 for f16/bf16 with double-buffering).
static std::uint32_t computeDSTCapacity(ComputeOp /*computeOp*/) {
  return kDefaultDSTCapacity;
}

static bool isTileValue(Value v) { return isa<ttcore::TileType>(v.getType()); }

//===----------------------------------------------------------------------===//
// Equivalence Classes for Merged Intervals
//===----------------------------------------------------------------------===//

using MergedClasses = llvm::EquivalenceClasses<Value>;

static Value getLeaderOrInsert(MergedClasses &merged, Value v) {
  return merged.getOrInsertLeaderValue(v);
}

static SmallVector<Value> getAllMerged(MergedClasses &merged, Value v) {
  SmallVector<Value> result;
  if (!merged.contains(v)) {
    result.push_back(v);
    return result;
  }
  for (Value member : merged.members(v)) {
    result.push_back(member);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Live Interval
//===----------------------------------------------------------------------===//

struct Interval {
  int64_t start; // Operation index where value becomes live
  int64_t end;   // Operation index of last use
  Value value;   // SSA value this interval represents
};

/// Compute the AffineMap for linearizing CB tile indices for a block argument.
/// Returns failure if the argument is not found in inputs.
static FailureOr<AffineMapAttr> computeIndexMapAttr(BlockArgument arg,
                                                    ComputeOp computeOp,
                                                    OpBuilder &builder) {
  Block *body = &computeOp.getRegion().front();
  for (auto [idx, input] : llvm::enumerate(computeOp.getInputs())) {
    if (arg == body->getArgument(idx)) {
      auto tensorType = cast<RankedTensorType>(input.getType());
      int64_t rank = tensorType.getRank();

      SmallVector<int64_t> staticShape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      SmallVector<int64_t> strides = mlir::computeStrides(staticShape);

      AffineExpr linearExpr = builder.getAffineConstantExpr(0);
      for (int64_t i = 0; i < rank; ++i) {
        linearExpr = linearExpr + builder.getAffineDimExpr(i) *
                                      builder.getAffineConstantExpr(strides[i]);
      }

      AffineMap indexMap = AffineMap::get(rank, /*numSymbols=*/0, linearExpr);
      return AffineMapAttr::get(indexMap);
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Phase 1: Copy Insertion
//===----------------------------------------------------------------------===//

/// Get consumers of a value sorted by their position in the block.
static SmallVector<Operation *> getSortedConsumers(Value v) {
  SmallVector<Operation *> consumers;
  for (Operation *user : v.getUsers()) {
    consumers.push_back(user);
  }
  // Sort by block position
  llvm::sort(consumers,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return consumers;
}

/// Check if any consumer is a tile unary operation.
static bool hasUnaryConsumer(ArrayRef<Operation *> consumers) {
  return llvm::any_of(consumers,
                      [](Operation *op) { return isTileUnaryOp(op); });
}

/// Phase 1: Insert copy operations for multi-consumer values where any
/// consumer is unary. Copies are inserted for all but the last consumer.
/// Unary ops overwrite their input in-place, so if other ops need that
/// value before the last unary consumer, we must copy it first.
///
/// For block arguments: Insert copy_tile (CB-to-DST copy).
/// For operation results: Insert copy_dst (DST-to-DST copy).
static void insertCopiesForMultiConsumerValues(ComputeOp computeOp,
                                               OpBuilder &builder) {
  Block *body = &computeOp.getRegion().front();

  // Collect values that need copy insertion (avoid modifying while iterating)
  SmallVector<std::pair<Value, SmallVector<Operation *>>> valuesToCopy;

  // Helper to check a value and add it to the list if needed
  auto checkValue = [&](Value v) {
    if (!isTileValue(v)) {
      return;
    }

    auto consumers = getSortedConsumers(v);
    if (consumers.size() <= 1) {
      return; // No multi-consumer
    }

    // Check if any consumer is unary - unary ops overwrite their input
    if (!hasUnaryConsumer(consumers)) {
      return; // All binary consumers - no copies needed
    }

    valuesToCopy.push_back({v, consumers});
  };

  // Check block arguments (e.g., x in "x * sigmoid(x)" pattern)
  for (Value blockArg : body->getArguments()) {
    checkValue(blockArg);
  }

  // Check operation results
  for (Operation &op : *body) {
    for (Value result : op.getResults()) {
      checkValue(result);
    }
  }

  // Insert copies for all but the last consumer
  for (auto &[value, consumers] : valuesToCopy) {
    for (size_t i = 0; i < consumers.size() - 1; ++i) {
      Operation *consumer = consumers[i];
      builder.setInsertionPoint(consumer);
      Location loc = consumer->getLoc();

      Value copyResult;
      if (isa<BlockArgument>(value)) {
        // Block argument: insert copy_tile (CB-to-DST)
        // Use sentinel kPlaceholderIndex for src_index and dst_index - will be
        // replaced later with proper LinearizedIndexOp and allocated DST index
        Value srcIndex =
            builder.create<arith::ConstantIndexOp>(loc, kPlaceholderIndex);
        Value dstIndex =
            builder.create<arith::ConstantIndexOp>(loc, kPlaceholderIndex);
        auto copyOp = builder.create<CopyTileOp>(
            loc,
            TypeRange{DSTRegisterType::get(value.getContext()),
                      value.getType()},
            ValueRange{value, srcIndex, dstIndex});
        copyResult = copyOp.getDstTile();
        LLVM_DEBUG({
          llvm::dbgs()
              << "Phase 1: Inserted placeholder copy_tile for consumer " << i
              << " of block arg " << value << "\n";
        });
      } else {
        // Operation result: insert copy_dst (DST-to-DST)
        auto copyOp = builder.create<CopyDstOp>(loc, value.getType(), value);
        copyResult = copyOp.getResult();
        LLVM_DEBUG({
          llvm::dbgs() << "Phase 1: Inserted copy_dst for consumer " << i
                       << " of value " << value << "\n";
        });
      }

      // Replace this consumer's use of value with the copy
      consumer->replaceUsesOfWith(value, copyResult);
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase 2: Build Live Intervals with Unary Merging
//===----------------------------------------------------------------------===//

/// Build live intervals for all tile values in the compute body.
/// Also performs unary merging: unary op input and output share DST.
static void buildLiveIntervals(Block *body, YieldOp yieldOp,
                               llvm::MapVector<Value, Interval> &intervals,
                               MergedClasses &merged,
                               DenseMap<Operation *, int64_t> &opIndex) {
  // Number operations
  int64_t idx = 0;
  for (Operation &op : *body) {
    opIndex[&op] = idx++;
  }
  int64_t yieldIdx = opIndex[yieldOp];

  // Build initial intervals
  for (Operation &op : *body) {
    int64_t currentIdx = opIndex[&op];

    // Extend input intervals to this use
    for (Value operand : op.getOperands()) {
      if (!isTileValue(operand)) {
        continue;
      }
      if (intervals.find(operand) == intervals.end()) {
        // Block argument: start at (first_use - 1) to enable register reuse.
        // Args consumed at position N get allocated before outputs produced at
        // N, allowing outputs to reuse the consumed args' registers.
        intervals[operand] = {currentIdx - 1, currentIdx, operand};
      } else {
        intervals[operand].end = std::max(intervals[operand].end, currentIdx);
      }
    }

    // Create interval for results
    for (Value result : op.getResults()) {
      if (!isTileValue(result)) {
        continue;
      }
      intervals[result] = {currentIdx, currentIdx, result};
    }
  }

  // Extend intervals for yielded values to the yield operation
  for (Value yielded : yieldOp.getValues()) {
    if (isTileValue(yielded) && intervals.count(yielded)) {
      intervals[yielded].end = yieldIdx;
    }
  }

  // Collect yielded values for quick lookup
  DenseSet<Value> yieldedValues;
  for (Value v : yieldOp.getValues()) {
    yieldedValues.insert(v);
  }

  // Merge intervals for unary ops
  for (Operation &op : *body) {
    if (!isTileUnaryOp(&op)) {
      continue;
    }

    Value input = op.getOperand(0);
    Value output = op.getResult(0);

    if (!intervals.count(input) || !intervals.count(output)) {
      continue;
    }

    // Check if output is yielded or all uses are unary
    bool shouldMerge = yieldedValues.contains(output);
    if (!shouldMerge) {
      // Check if all uses are unary
      shouldMerge = llvm::all_of(output.getUsers(), [](Operation *user) {
        return isTileUnaryOp(user) || isa<YieldOp>(user);
      });
    }

    if (shouldMerge) {
      auto itA = merged.findLeader(merged.insert(input));
      auto itB = merged.findLeader(merged.insert(output));
      if (itA != itB) {
        merged.unionSets(itA, itB);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Phase 2: Merged " << input << " and " << output
                     << "\n";
      });
    }
  }

  // Propagate merged intervals: all values in a merged set get the same
  // interval (the union of all their individual intervals).
  DenseSet<Value> processed;
  for (auto &[value, interval] : intervals) {
    Value root = getLeaderOrInsert(merged, value);
    if (processed.contains(root)) {
      continue;
    }
    processed.insert(root);

    // Find all values in this merged set and compute the union interval
    auto allMerged = getAllMerged(merged, value);
    if (allMerged.size() <= 1) {
      continue;
    }

    int64_t mergedStart = intervals[allMerged[0]].start;
    int64_t mergedEnd = intervals[allMerged[0]].end;
    for (Value v : allMerged) {
      mergedStart = std::min(mergedStart, intervals[v].start);
      mergedEnd = std::max(mergedEnd, intervals[v].end);
    }

    // Update all values in the set to have the merged interval
    for (Value v : allMerged) {
      intervals[v].start = mergedStart;
      intervals[v].end = mergedEnd;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Merged set interval: [" << mergedStart << ", "
                   << mergedEnd << "] for " << allMerged.size() << " values\n";
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "=== Live Intervals ===\n";
    for (auto &[value, interval] : intervals) {
      llvm::dbgs() << "  " << value << ": [" << interval.start << ", "
                   << interval.end << "]\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// Phase 3: Linear Scan Allocation for Inputs/Intermediates
//===----------------------------------------------------------------------===//

/// Helper to check if a value is yielded by the compute operation.
static bool isYieldedValue(Value val, YieldOp yieldOp) {
  return llvm::is_contained(yieldOp.getValues(), val);
}

/// Core linear scan allocation logic (shared by Phase 3 and Phase 4).
/// filterFn determines which intervals to process.
/// Returns the maximum DST index used + 1 (footprint).
template <typename FilterFn>
static FailureOr<std::uint32_t> linearScanAllocateFiltered(
    llvm::MapVector<Value, Interval> &intervals, MergedClasses &merged,
    llvm::SmallBitVector &freeRegs, DenseMap<Value, std::uint32_t> &assignment,
    FilterFn &&shouldProcess, ComputeOp computeOp, StringRef phaseName) {
  // Sort intervals by start position
  SmallVector<Interval *> sortedIntervals;
  for (auto &[val, interval] : intervals) {
    sortedIntervals.push_back(&interval);
  }
  llvm::sort(sortedIntervals,
             [](Interval *a, Interval *b) { return a->start < b->start; });

  SmallVector<Interval *> active;
  DenseSet<Value> processedRoots;
  std::optional<std::uint32_t> maxDstUsed; // Track highest DST index allocated

  for (Interval *interval : sortedIntervals) {
    if (!shouldProcess(interval->value)) {
      continue;
    }

    // Skip if already assigned
    if (assignment.count(interval->value)) {
      continue;
    }

    // Skip if merged set already processed
    Value root = getLeaderOrInsert(merged, interval->value);
    if (processedRoots.contains(root)) {
      continue;
    }

    // Expire old intervals (free registers for reuse)
    SmallVector<Interval *> toRemove;
    for (Interval *activeInterval : active) {
      if (activeInterval->end <= interval->start) {
        auto it = assignment.find(activeInterval->value);
        if (it != assignment.end()) {
          LLVM_DEBUG({
            llvm::dbgs() << phaseName << ": Expired interval for "
                         << activeInterval->value << ", freed DST["
                         << it->second << "]\n";
          });
          freeRegs.set(it->second);
        }
        toRemove.push_back(activeInterval);
      }
    }
    for (Interval *i : toRemove) {
      active.erase(std::find(active.begin(), active.end(), i));
    }

    // Find first free register
    int freeReg = freeRegs.find_first();
    if (freeReg < 0) {
      // TODO: Implement spilling or compute fission for high register pressure
      return failure();
    }

    freeRegs.reset(freeReg);
    std::uint32_t regIdx = static_cast<std::uint32_t>(freeReg);
    maxDstUsed = maxDstUsed ? std::max(*maxDstUsed, regIdx) : regIdx;

    // Assign to all values in the merged set
    auto allMerged = getAllMerged(merged, interval->value);
    for (Value mergedVal : allMerged) {
      if (!assignment.count(mergedVal)) {
        assignment[mergedVal] = regIdx;
      }
    }

    active.push_back(interval);
    processedRoots.insert(root);

    LLVM_DEBUG({
      llvm::dbgs() << phaseName << ": Allocated DST[" << regIdx << "] for "
                   << interval->value
                   << " (merged set size: " << allMerged.size() << ")\n";
    });
  }

  // Return footprint: 0 if nothing allocated, otherwise maxDstUsed + 1
  return maxDstUsed ? *maxDstUsed + 1 : 0;
}

//===----------------------------------------------------------------------===//
// Main Pass Implementation
//===----------------------------------------------------------------------===//

struct TTLAssignDSTPass : public impl::TTLAssignDSTBase<TTLAssignDSTPass> {
  using Base = impl::TTLAssignDSTBase<TTLAssignDSTPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](ComputeOp computeOp) {
      Block *body = &computeOp.getRegion().front();
      if (!body) {
        return;
      }

      auto yieldOp = dyn_cast<YieldOp>(body->getTerminator());
      if (!yieldOp) {
        return;
      }

      std::uint32_t capacity =
          dstCapacity == 0 ? computeDSTCapacity(computeOp) : dstCapacity;

      OpBuilder builder(body, body->begin());

      //=== Phase 1: Copy Insertion ===
      LLVM_DEBUG(llvm::dbgs() << "=== Phase 1: Copy Insertion ===\n");
      insertCopiesForMultiConsumerValues(computeOp, builder);

      //=== Phase 2: Build Live Intervals ===
      LLVM_DEBUG(llvm::dbgs() << "=== Phase 2: Build Live Intervals ===\n");
      llvm::MapVector<Value, Interval> intervals;
      MergedClasses merged;
      DenseMap<Operation *, int64_t> opIndex;
      buildLiveIntervals(body, yieldOp, intervals, merged, opIndex);

      //=== Phase 3 & 4: Linear Scan Allocation ===
      DenseMap<Value, std::uint32_t> dstAssignment;

      if (separateOutputRegion) {
        // Phase 3: Allocate inputs/intermediates (non-yielded values)
        LLVM_DEBUG(llvm::dbgs() << "Using separate output region mode\n");
        LLVM_DEBUG(llvm::dbgs() << "=== Phase 3: Linear Scan Allocation ===\n");
        llvm::SmallBitVector freeRegs(capacity);
        freeRegs.set();
        auto inputsFootprint = linearScanAllocateFiltered(
            intervals, merged, freeRegs, dstAssignment,
            [&](Value val) { return !isYieldedValue(val, yieldOp); }, computeOp,
            "Phase 3");
        if (failed(inputsFootprint)) {
          computeOp.emitOpError()
              << "insufficient DST registers: all " << capacity
              << " registers in use (spilling not yet implemented)";
          signalPassFailure();
          return;
        }
        LLVM_DEBUG({
          llvm::dbgs() << "Phase 3 footprint: " << *inputsFootprint
                       << " registers\n";
        });

        // Phase 4: Allocate outputs (yielded values) starting at
        // inputsFootprint
        LLVM_DEBUG(llvm::dbgs() << "=== Phase 4: Linear Scan Allocation ===\n");
        llvm::SmallBitVector outputRegs(capacity);
        for (std::uint32_t i = *inputsFootprint; i < capacity; ++i) {
          outputRegs.set(i);
        }
        if (failed(linearScanAllocateFiltered(
                intervals, merged, outputRegs, dstAssignment,
                [&](Value val) { return isYieldedValue(val, yieldOp); },
                computeOp, "Phase 4"))) {
          computeOp.emitOpError()
              << "insufficient DST registers for outputs: all " << capacity
              << " registers in use (spilling not yet implemented)";
          signalPassFailure();
          return;
        }
      } else {
        // Single-pass allocation: Outputs can reuse input registers (default)
        LLVM_DEBUG(llvm::dbgs() << "=== Phase 3: Linear Scan Allocation ===\n");
        llvm::SmallBitVector freeRegs(capacity);
        freeRegs.set();
        if (failed(linearScanAllocateFiltered(
                intervals, merged, freeRegs, dstAssignment,
                [](Value) { return true; }, computeOp, "Phase 3"))) {
          computeOp.emitOpError()
              << "insufficient DST registers: all " << capacity
              << " registers in use (spilling not yet implemented)";
          signalPassFailure();
          return;
        }
      }

      // Compute max DST usage for debug output / testing
      std::uint32_t maxDstUsed = 0;
      for (auto &[val, reg] : dstAssignment) {
        maxDstUsed = std::max(maxDstUsed, reg);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "=== Final DST Assignment ===\n";
        for (auto &[val, reg] : dstAssignment) {
          llvm::dbgs() << "  " << val << " -> DST[" << reg << "]\n";
        }
        if (!dstAssignment.empty()) {
          llvm::dbgs() << "Max DST usage: " << (maxDstUsed + 1) << " / "
                       << capacity << " registers\n";
        }
      });

      //=== Insert copy_tile for block arguments and set dst_idx ===
      llvm::SmallBitVector inUse(capacity);
      DenseMap<Value, std::uint32_t> dstIndexForValue;

      // Copy assignments for non-block-args
      for (auto &[val, reg] : dstAssignment) {
        if (!isa<BlockArgument>(val)) {
          dstIndexForValue[val] = reg;
        }
      }

      // Helper to check if a copy_tile has placeholder indices
      auto isPlaceholderCopy = [](CopyTileOp copyTile) {
        if (auto srcConst = copyTile.getSrcIndex()
                                .getDefiningOp<arith::ConstantIndexOp>()) {
          return srcConst.value() == kPlaceholderIndex;
        }
        return false;
      };

      // First: Replace placeholder copy_tile ops with proper copies
      // These were inserted in Phase 1 for block args with multiple consumers
      SmallVector<CopyTileOp> placeholderCopies;
      for (Operation &op : *body) {
        if (auto copyTile = dyn_cast<CopyTileOp>(&op)) {
          if (isPlaceholderCopy(copyTile)) {
            placeholderCopies.push_back(copyTile);
          }
        }
      }

      for (CopyTileOp placeholder : placeholderCopies) {
        auto arg = dyn_cast<BlockArgument>(placeholder.getSrc());
        if (!arg) {
          continue;
        }

        // Get assigned DST index for this copy's result
        std::uint32_t assignedDstIndex = 0;
        auto it = dstAssignment.find(placeholder.getDstTile());
        if (it != dstAssignment.end()) {
          assignedDstIndex = it->second;
        } else {
          // Fall back: find first free register
          int freeReg = inUse.find_first_unset();
          if (freeReg < 0) {
            computeOp.emitOpError(
                "no free DST register for placeholder copy_tile");
            signalPassFailure();
            return;
          }
          assignedDstIndex = static_cast<std::uint32_t>(freeReg);
        }
        inUse.set(assignedDstIndex);

        builder.setInsertionPoint(placeholder);
        Location loc = placeholder.getLoc();

        auto indexMapAttr = computeIndexMapAttr(arg, computeOp, builder);
        if (failed(indexMapAttr)) {
          placeholder.emitOpError("block argument not found in compute inputs");
          signalPassFailure();
          return;
        }
        Value srcIndex = builder.create<LinearizedIndexOp>(loc, *indexMapAttr);
        Value dstIndex =
            builder.create<arith::ConstantIndexOp>(loc, assignedDstIndex);
        auto newCopy = builder.create<CopyTileOp>(
            loc,
            TypeRange{DSTRegisterType::get(arg.getContext()), arg.getType()},
            ValueRange{arg, srcIndex, dstIndex});
        dstIndexForValue[newCopy.getDstTile()] = assignedDstIndex;

        // Replace uses of placeholder with new copy
        placeholder.getDstTile().replaceAllUsesWith(newCopy.getDstTile());
        placeholder.getDstToken().replaceAllUsesWith(newCopy.getDstToken());

        // Erase the placeholder and its constant operands
        Operation *srcIndexDef = placeholder.getSrcIndex().getDefiningOp();
        Operation *dstIndexDef = placeholder.getDstIndex().getDefiningOp();
        placeholder.erase();
        if (srcIndexDef && srcIndexDef->use_empty()) {
          srcIndexDef->erase();
        }
        if (dstIndexDef && dstIndexDef->use_empty()) {
          dstIndexDef->erase();
        }
      }

      // Second: Process remaining block arguments - insert copy_tile at first
      // use
      for (Operation &op : *body) {
        for (OpOperand &operand : op.getOpOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand.get());
          if (!arg || !isTileValue(arg)) {
            continue;
          }

          // Skip if already copied
          if (dstIndexForValue.count(arg)) {
            continue;
          }

          // Get assigned DST index from allocation
          std::uint32_t assignedDstIndex = 0;
          auto it = dstAssignment.find(arg);
          if (it != dstAssignment.end()) {
            assignedDstIndex = it->second;
          } else {
            // Fall back: find first free register
            int freeReg = inUse.find_first_unset();
            if (freeReg < 0) {
              computeOp.emitOpError("no free DST register for block argument");
              signalPassFailure();
              return;
            }
            assignedDstIndex = static_cast<std::uint32_t>(freeReg);
          }
          inUse.set(assignedDstIndex);

          builder.setInsertionPoint(&op);
          Location loc = op.getLoc();

          auto indexMapAttr = computeIndexMapAttr(arg, computeOp, builder);
          if (failed(indexMapAttr)) {
            op.emitOpError("block argument not found in compute inputs");
            signalPassFailure();
            return;
          }
          Value srcIndex =
              builder.create<LinearizedIndexOp>(loc, *indexMapAttr);
          Value dstIndex =
              builder.create<arith::ConstantIndexOp>(loc, assignedDstIndex);
          auto copy = builder.create<CopyTileOp>(
              loc,
              TypeRange{DSTRegisterType::get(arg.getContext()), arg.getType()},
              ValueRange{arg, srcIndex, dstIndex});
          dstIndexForValue[copy.getDstTile()] = assignedDstIndex;

          arg.replaceUsesWithIf(copy.getDstTile(), [&](OpOperand &use) {
            // Don't replace in copy_tile ops - they need the original block arg
            return use.getOwner() != copy && !isa<CopyTileOp>(use.getOwner());
          });
        }
      }

      // Set dst_idx attributes on tile compute ops
      for (Operation &op : *body) {
        if (!isTileComputeOp(&op) && !isa<CopyDstOp>(&op)) {
          continue;
        }

        for (Value res : op.getResults()) {
          if (!isTileValue(res)) {
            continue;
          }

          // Find the DST index
          std::uint32_t dstIdx = 0;
          auto it = dstIndexForValue.find(res);
          if (it != dstIndexForValue.end()) {
            dstIdx = it->second;
          } else {
            // Check in original assignment
            auto assignIt = dstAssignment.find(res);
            if (assignIt != dstAssignment.end()) {
              dstIdx = assignIt->second;
              dstIndexForValue[res] = dstIdx;
            }
          }

          op.setAttr(kDstIdxAttrName,
                     builder.getI32IntegerAttr(static_cast<int32_t>(dstIdx)));
        }
      }

      //=== Post-pass verification: no placeholder copy_tile ops ===
      // A placeholder copy_tile has kPlaceholderIndex as src_index or
      // dst_index. These are inserted in Phase 1 for block args with multiple
      // consumers and should have been replaced with proper copies.
      for (Operation &op : *body) {
        if (auto copyTile = dyn_cast<CopyTileOp>(&op)) {
          if (isPlaceholderCopy(copyTile)) {
            copyTile.emitOpError()
                << "placeholder copy_tile not replaced with proper copy "
                << "(src_index has sentinel value " << kPlaceholderIndex << ")";
            signalPassFailure();
            return;
          }
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
