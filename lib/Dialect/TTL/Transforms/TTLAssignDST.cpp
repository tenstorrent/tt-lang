// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL DST Register Assignment Pass
//===----------------------------------------------------------------------===//
//
// This pass performs DST (destination) register assignment for ttl.compute
// operations using interval-based linear scan allocation with in-place
// operation merging. The algorithm is based on
// docs/development/DST_Allocation.md:
//
// Phase 1: Copy Insertion
//   - For values with multiple consumers where any consumer is in-place
//   - Insert ttl.copy_dst for all but the last consumer
//   - Prevents in-place ops from clobbering values needed by other consumers
//
// Phase 2: Build Live Intervals with In-Place Merging
//   - Assign operation indices in block order
//   - Build lifetime intervals [start, end] for each tile value
//   - Merge intervals for in-place ops (input and output share DST)
//   - Use union-find to track merged equivalence classes
//
// Phase 3: Linear Scan Allocation
//   - Process intervals by start position (Wimmer & Franz, CGO'10)
//   - Expire intervals when their last use passes
//   - Reuse freed registers for new values
//   - Optional: Separate output region (--separate-output-region flag)
//
// This pass also inserts ttl.copy_tile ops for block arguments and assigns
// dst_index SSA operands to all tile compute operations.
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

/// Sentinel value for placeholder copy_tile indices. Using max value since
/// valid indices are small non-negative integers. This is replaced with proper
/// indices during the copy_tile insertion phase.
constexpr int64_t kPlaceholderIndex = std::numeric_limits<int64_t>::max();

static bool isTileValue(Value v) { return isa<ttcore::TileType>(v.getType()); }

//===----------------------------------------------------------------------===//
// Equivalence Classes for Merged Intervals
//===----------------------------------------------------------------------===//

using MergedClasses = llvm::EquivalenceClasses<Value>;

/// Return all values in the merged equivalence class of `v`.
/// If `v` is not in any class, returns just `v` itself.
static SmallVector<Value> getMergedValues(MergedClasses &merged, Value v) {
  if (!merged.contains(v)) {
    return {v};
  }
  return SmallVector<Value>(merged.members(v));
}

//===----------------------------------------------------------------------===//
// Live Interval
//===----------------------------------------------------------------------===//

struct Interval {
  int64_t start; // Operation index where value becomes live
  int64_t end;   // Operation index of last use
  Value value;   // SSA value this interval represents
};

/// Allocate a DST register and create a CopyTileOp for a block argument.
/// Looks up assignment first; falls back to allocating a free register.
/// If dstIndexOverride is provided, it takes precedence over the assignment
/// map.
static FailureOr<CopyTileOp> createCopyTileForArg(
    BlockArgument arg, ComputeOp computeOp, OpBuilder &builder,
    const DenseMap<Value, std::uint32_t> &dstAssignment,
    llvm::SmallBitVector &inUse,
    DenseMap<Value, std::uint32_t> &dstIndexForValue,
    std::optional<std::uint32_t> dstIndexOverride = std::nullopt) {
  std::uint32_t assignedDstIndex = 0;
  if (dstIndexOverride) {
    assignedDstIndex = *dstIndexOverride;
  } else if (auto it = dstAssignment.find(arg); it != dstAssignment.end()) {
    assignedDstIndex = it->second;
  } else {
    int freeReg = inUse.find_first_unset();
    if (freeReg < 0) {
      return computeOp.emitOpError("no free DST register for block argument");
    }
    assignedDstIndex = static_cast<std::uint32_t>(freeReg);
  }
  inUse.set(assignedDstIndex);

  Location loc = builder.getInsertionPoint()->getLoc();
  Value dstIndex =
      arith::ConstantIndexOp::create(builder, loc, assignedDstIndex);
  // src_indices are empty here; populated by the iter_index phase below.
  auto copy = CopyTileOp::create(
      builder, loc,
      TypeRange{DSTRegisterType::get(arg.getContext()), arg.getType()},
      ValueRange{arg, dstIndex});
  dstIndexForValue[copy.getDstTile()] = assignedDstIndex;
  return copy;
}

//===----------------------------------------------------------------------===//
// Phase 1: Copy Insertion
//===----------------------------------------------------------------------===//

/// Get consumers of a value sorted by their position in the block.
/// Excludes CB-reading ops (bcast, etc.) since they don't use DST for input.
static SmallVector<Operation *> getSortedConsumers(Value v) {
  SmallVector<Operation *> consumers;
  for (Operation *user : v.getUsers()) {
    // Skip CB-input ops (bcast, reduce, transpose, FPU binary, etc.)
    if (isCBInputOp(user)) {
      continue;
    }
    consumers.push_back(user);
  }
  // Sort by block position
  llvm::sort(consumers,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return consumers;
}

/// Check if any consumer is an in-place operation (overwrites DST input).
static bool hasInPlaceConsumer(ArrayRef<Operation *> consumers) {
  return llvm::any_of(consumers, [](Operation *op) {
    return op->hasTrait<TTLInPlaceOpTrait>();
  });
}

/// Phase 1: Insert copy operations for multi-consumer values where any
/// consumer is in-place. Copies are inserted for all but the last consumer.
/// In-place ops overwrite their input in DST, so if other ops need that
/// value before the last in-place consumer, we must copy it first.
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

    // Check if any consumer is in-place - in-place ops overwrite their input
    if (!hasInPlaceConsumer(consumers)) {
      return; // No in-place consumers - no copies needed
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
        // Block argument: insert placeholder copy_tile (CB-to-DST).
        // Replaced later with proper DST index allocation. Marked with
        // kPlaceholderCopyAttrName so Phase 2b can identify and replace it.
        Value dstIndex =
            arith::ConstantIndexOp::create(builder, loc, kPlaceholderIndex);
        auto copyOp = CopyTileOp::create(
            builder, loc,
            TypeRange{DSTRegisterType::get(value.getContext()),
                      value.getType()},
            ValueRange{value, dstIndex});
        copyOp->setAttr(kPlaceholderCopyAttrName, builder.getUnitAttr());
        copyResult = copyOp.getDstTile();
        LLVM_DEBUG({
          llvm::dbgs()
              << "Phase 1: Inserted placeholder copy_tile for consumer " << i
              << " of block arg " << value << "\n";
        });
      } else {
        // Operation result: insert copy_dst (DST-to-DST)
        auto copyOp =
            CopyDstOp::create(builder, loc, value.getType(), value,
                              createPlaceholderDstIndex(builder, loc));
        addPlaceholderDstIndexAttr(copyOp.getOperation());
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
// Phase 2: Build Live Intervals with In-Place Merging
//===----------------------------------------------------------------------===//

/// Build live intervals for all tile values in the compute body.
/// Also performs in-place merging: in-place op input and output share DST.
static void buildLiveIntervals(Block *body,
                               llvm::MapVector<Value, Interval> &intervals,
                               MergedClasses &merged,
                               DenseMap<Operation *, int64_t> &opIndex) {
  // Number operations
  int64_t idx = 0;
  for (Operation &op : *body) {
    opIndex[&op] = idx++;
  }

  // Build initial intervals
  for (Operation &op : *body) {
    int64_t currentIdx = opIndex[&op];

    // Extend input intervals to this use (skipping ops with CB inputs)
    if (!isCBInputOp(&op)) {
      for (Value operand : op.getOperands()) {
        if (!isTileValue(operand)) {
          continue;
        }
        if (!intervals.count(operand)) {
          // Block argument: start at (first_use - 1) to enable register reuse.
          // Args consumed at position N get allocated before outputs produced
          // at N, allowing outputs to reuse the consumed args' registers.
          intervals[operand] = {currentIdx - 1, currentIdx, operand};
        } else {
          intervals[operand].end = std::max(intervals[operand].end, currentIdx);
        }
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

  // Merge intervals for in-place ops (input and output share DST slot).
  // In-place ops (exp_tile, abs_tile, etc.) read from and write to the same
  // DST register -- this is a hardware constraint. The merge is unconditional:
  // regardless of what downstream ops consume the result, the input and output
  // must share the same DST index so the lowered instruction (e.g.,
  // exp_tile(dst_index)) operates on the correct register.
  for (Operation &op : *body) {
    if (!op.hasTrait<TTLInPlaceOpTrait>()) {
      continue;
    }

    Value input = op.getOperand(0);
    Value output = op.getResult(0);

    if (!intervals.count(input) || !intervals.count(output)) {
      continue;
    }

    auto itA = merged.findLeader(merged.insert(input));
    auto itB = merged.findLeader(merged.insert(output));
    if (itA != itB) {
      merged.unionSets(itA, itB);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Phase 2: Merged " << input << " and " << output << "\n";
    });
  }

  // Merge intervals for matmul accumulator and output. The accumulator is
  // loaded into DST via copy_tile; matmul_block accumulates into the same
  // register (DST += A*B).
  for (Operation &op : *body) {
    auto matmul = dyn_cast<TileMatmulBlockOp>(&op);
    if (!matmul || !matmul.getAccumulator()) {
      continue;
    }

    Value acc = matmul.getAccumulator();
    Value out = matmul.getResult();

    if (!intervals.contains(acc) || !intervals.contains(out)) {
      continue;
    }

    auto itA = merged.findLeader(merged.insert(acc));
    auto itB = merged.findLeader(merged.insert(out));
    if (itA != itB) {
      merged.unionSets(itA, itB);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Phase 2: Merged matmul accumulator " << acc
                   << " and output " << out << "\n";
    });
  }

  // Propagate merged intervals: all values in a merged set get the same
  // interval (the union of all their individual intervals).
  DenseSet<Value> processed;
  for (auto &[value, interval] : intervals) {
    Value root = merged.getOrInsertLeaderValue(value);
    if (processed.contains(root)) {
      continue;
    }
    processed.insert(root);

    // Find all values in this merged set and compute the union interval
    auto allMerged = getMergedValues(merged, value);
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

  // Prevent DST register reuse between FPU binary ops.
  // FPU binary ops (add_tiles, mul_tiles, sub_tiles) accumulate into their
  // output DST register: result = old_DST_value + computed_value. If two FPU
  // binary ops share the same DST output index, the second reads the first's
  // residual and produces a corrupted result. We prevent this by extending
  // FPU binary result intervals so the linear scan allocator assigns distinct
  // registers.
  //
  // TODO(#343): This wastes DST capacity. The proper fix is to pass
  // acc_to_dest=false to add_tiles_init/sub_tiles_init/mul_tiles_init in
  // tt-mlir's TTKernel dialect (currently has a FIXME in TTKernelOps.td).
  // With explicit overwrite mode, DST reuse between FPU binary ops would be
  // safe and this interval extension could be removed.
  {
    // Include TileMatmulBlockOp alongside FPU binary ops: matmul_block also
    // accumulates into DST and its slot must not be reused by another
    // accumulating op within the same sync region.
    auto isFPUAccumulatingOp = [](Operation &op) {
      return op.hasAttr(kFPUBinaryAttrName) || isa<TileMatmulBlockOp>(&op);
    };

    SmallVector<int64_t> fpuBinaryStarts;
    for (Operation &op : *body) {
      if (isFPUAccumulatingOp(op)) {
        fpuBinaryStarts.push_back(opIndex[&op]);
      }
    }

    if (fpuBinaryStarts.size() > 1) {
      int64_t lastFPUStart = *llvm::max_element(fpuBinaryStarts);
      for (Operation &op : *body) {
        if (!isFPUAccumulatingOp(op)) {
          continue;
        }
        for (Value result : op.getResults()) {
          if (!isTileValue(result) || !intervals.count(result)) {
            continue;
          }
          // Use lastFPUStart + 1 because the linear scan expires intervals
          // with end <= start, so end must be strictly greater than the last
          // FPU binary op's start index to remain active during allocation.
          if (intervals[result].end <= lastFPUStart) {
            LLVM_DEBUG({
              llvm::dbgs() << "Phase 2: Extended FPU binary interval from ["
                           << intervals[result].start << ", "
                           << intervals[result].end << "] to ["
                           << intervals[result].start << ", "
                           << (lastFPUStart + 1) << "] to prevent DST reuse\n";
            });
            intervals[result].end = lastFPUStart + 1;
          }
        }
      }
    }
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

/// Helper to check if a value is stored by a tile_store in the compute body.
static bool isStoredValue(Value val, Block &body) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    return isa<TileStoreOp>(user) && user->getBlock() == &body;
  });
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
    Value root = merged.getOrInsertLeaderValue(interval->value);
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
    auto allMerged = getMergedValues(merged, interval->value);
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
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](ComputeOp computeOp) {
      Block *body = &computeOp.getRegion().front();

      std::uint32_t capacity = dstCapacity;
      if (capacity == 0) {
        auto computed = computeDSTCapacity(computeOp);
        if (failed(computed)) {
          signalPassFailure();
          return;
        }
        capacity = *computed;
      }

      OpBuilder builder(body, body->begin());

      //=== Phase 0: FPU Binary Detection ===
      // Mark add/sub/mul ops as FPU-eligible when both operands are input
      // block arguments (CB-backed). FPU reads from CB, needing 0 DST input
      // slots. Output block arguments are excluded because they may represent
      // accumulation patterns that require DST copy_tile.
      //
      // TODO: Support mixed operands (one CB, one DST) via
      // ttkernel.binary_dest_reuse_tiles with DEST_TO_SRCA/DEST_TO_SRCB.
      // This would allow FPU lowering for patterns like
      // tile_add %arg0, %computed where one operand is already in DST.
      LLVM_DEBUG(llvm::dbgs() << "=== Phase 0: FPU Binary Detection ===\n");
      if (enableFPUBinaryOps) {
        unsigned numInputs = computeOp.getNumInputs();
        auto indexingMaps = computeOp.getIndexingMapsArray();
        for (Operation &op : *body) {
          if (!isa<AddTileOp, SubTileOp, MulTileOp>(&op)) {
            continue;
          }
          Value lhs = op.getOperand(0);
          Value rhs = op.getOperand(1);
          auto lhsArg = dyn_cast<BlockArgument>(lhs);
          auto rhsArg = dyn_cast<BlockArgument>(rhs);
          if (lhsArg && rhsArg && lhsArg.getArgNumber() < numInputs &&
              rhsArg.getArgNumber() < numInputs) {
            // FPU binary ops use a single shared CB tile index for both
            // operands, so the indexing maps must be identical. This is not
            // an error — the op is still valid, it just falls back to the
            // copy_tile + SFPU path which handles each operand independently.
            AffineMap lhsMap = indexingMaps[lhsArg.getArgNumber()];
            AffineMap rhsMap = indexingMaps[rhsArg.getArgNumber()];
            if (lhsMap != rhsMap) {
              LLVM_DEBUG({
                llvm::dbgs()
                    << "Phase 0: Skipping FPU binary (incompatible indexing "
                       "maps): "
                    << op.getName() << "\n";
              });
              continue;
            }
            op.setAttr(kFPUBinaryAttrName, builder.getUnitAttr());
            LLVM_DEBUG({
              llvm::dbgs() << "Phase 0: Marked FPU binary: " << op.getName()
                           << "\n";
            });
          }
        }
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Phase 0: FPU binary ops disabled, skipping\n");
      }

      //=== Phase 1: Copy Insertion ===
      LLVM_DEBUG(llvm::dbgs() << "=== Phase 1: Copy Insertion ===\n");
      insertCopiesForMultiConsumerValues(computeOp, builder);

      //=== Phase 2: Build Live Intervals ===
      LLVM_DEBUG(llvm::dbgs() << "=== Phase 2: Build Live Intervals ===\n");
      llvm::MapVector<Value, Interval> intervals;
      MergedClasses merged;
      DenseMap<Operation *, int64_t> opIndex;
      buildLiveIntervals(body, intervals, merged, opIndex);

      //=== Phase 3 & 4: Linear Scan Allocation ===
      DenseMap<Value, std::uint32_t> dstAssignment;

      if (separateOutputRegion) {
        // Phase 3: Allocate inputs/intermediates (non-stored values)
        LLVM_DEBUG(llvm::dbgs() << "Using separate output region mode\n");
        LLVM_DEBUG(llvm::dbgs() << "=== Phase 3: Linear Scan Allocation ===\n");
        llvm::SmallBitVector freeRegs(capacity);
        freeRegs.set();
        auto inputsFootprint = linearScanAllocateFiltered(
            intervals, merged, freeRegs, dstAssignment,
            [&](Value val) {
              // For separate output region mode, check if any member of the
              // merged set is stored. If so, defer the entire set to Phase 4.
              auto allMerged = getMergedValues(merged, val);
              return llvm::none_of(allMerged, [&](Value member) {
                return isStoredValue(member, *body);
              });
            },
            computeOp, "Phase 3");
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

        // Phase 4: Allocate outputs (stored values) starting at
        // inputsFootprint
        LLVM_DEBUG(llvm::dbgs() << "=== Phase 4: Linear Scan Allocation ===\n");
        llvm::SmallBitVector outputRegs(capacity);
        for (std::uint32_t i = *inputsFootprint; i < capacity; ++i) {
          outputRegs.set(i);
        }
        if (failed(linearScanAllocateFiltered(
                intervals, merged, outputRegs, dstAssignment,
                [&](Value val) { return isStoredValue(val, *body); }, computeOp,
                "Phase 4"))) {
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

      //=== Insert copy_tile for block arguments and set dst_index ===
      llvm::SmallBitVector inUse(capacity);
      DenseMap<Value, std::uint32_t> dstIndexForValue;

      // Copy assignments for non-block-args
      for (auto &[val, reg] : dstAssignment) {
        if (!isa<BlockArgument>(val)) {
          dstIndexForValue[val] = reg;
        }
      }

      auto isPlaceholderCopy = [](CopyTileOp copyTile) {
        return copyTile->hasAttr(kPlaceholderCopyAttrName);
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

        // Use the placeholder's DST assignment if available.
        std::optional<std::uint32_t> dstOverride;
        auto placeholderIt = dstAssignment.find(placeholder.getDstTile());
        if (placeholderIt != dstAssignment.end()) {
          dstOverride = placeholderIt->second;
        }

        builder.setInsertionPoint(placeholder);
        auto newCopy =
            createCopyTileForArg(arg, computeOp, builder, dstAssignment, inUse,
                                 dstIndexForValue, dstOverride);
        if (failed(newCopy)) {
          signalPassFailure();
          return;
        }

        placeholder.getDstTile().replaceAllUsesWith(newCopy->getDstTile());
        placeholder.getDstToken().replaceAllUsesWith(newCopy->getDstToken());

        Operation *dstIndexDef = placeholder.getDstIndex().getDefiningOp();
        placeholder.erase();
        if (dstIndexDef && dstIndexDef->use_empty()) {
          dstIndexDef->erase();
        }
      }

      // Insert copy_tile for remaining block args at first non-CB-reading use.
      // Copies must be inserted at first use (not block start) to match the
      // liveness intervals that DST allocation was computed against.
      for (Operation &op : *body) {
        if (isCBInputOp(&op) || isa<CopyTileOp>(&op)) {
          continue;
        }
        for (OpOperand &operand : op.getOpOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand.get());
          if (!arg || !isTileValue(arg) || dstIndexForValue.count(arg)) {
            continue;
          }

          builder.setInsertionPoint(&op);
          auto copy = createCopyTileForArg(
              arg, computeOp, builder, dstAssignment, inUse, dstIndexForValue);
          if (failed(copy)) {
            signalPassFailure();
            return;
          }

          arg.replaceUsesWithIf(copy->getDstTile(), [&](OpOperand &use) {
            return use.getOwner() != copy->getOperation() &&
                   !isa<CopyTileOp>(use.getOwner()) &&
                   !isCBInputOp(use.getOwner());
          });
        }
      }

      // Set dst_index operands on tile compute ops, copy_tile, and copy_dst.
      for (Operation &op : *body) {
        if (!isTileComputeOp(&op) && !isa<CopyDstOp>(&op) &&
            !isa<CopyTileOp>(&op)) {
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

          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(&op);
          Value dstIdxVal =
              arith::ConstantIndexOp::create(builder, op.getLoc(), dstIdx);
          setTileOpDstIndex(&op, dstIdxVal);
          op.removeAttr(kDstPlaceholderAttrName);
        }
      }

      // Set dst_index on tile_store ops based on their source tile's DST slot.
      for (Operation &op : *body) {
        auto store = dyn_cast<TileStoreOp>(&op);
        if (!store) {
          continue;
        }
        Value tile = store.getTile();
        auto it = dstIndexForValue.find(tile);
        if (it != dstIndexForValue.end()) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(&op);
          Value dstIdxVal =
              arith::ConstantIndexOp::create(builder, op.getLoc(), it->second);
          setTileOpDstIndex(&op, dstIdxVal);
          op.removeAttr(kDstPlaceholderAttrName);
        }
      }

      //=== Post-pass verification: no unassigned dst_index placeholders ===
      for (Operation &op : *body) {
        if (op.hasAttr(kDstPlaceholderAttrName)) {
          llvm_unreachable("dst_index was not assigned by AssignDST");
        }
        if (auto dstVal = getTileOpDstIndex(&op)) {
          if (auto constIdx = getConstantIntValue(*dstVal)) {
            if (*constIdx == kUnassignedDstIndex) {
              llvm_unreachable(
                  "dst_index is still the unassigned sentinel (-1)");
            }
          }
        }
      }

      //=== Post-pass verification: no placeholder copy_tile ops ===
      for (Operation &op : *body) {
        if (auto copyTile = dyn_cast<CopyTileOp>(&op)) {
          if (isPlaceholderCopy(copyTile)) {
            copyTile.emitOpError()
                << "placeholder copy_tile not replaced with proper copy";
            signalPassFailure();
            return;
          }
        }
      }

      //=== Populate CB indices using iter_index ===
      // Create iter_index ops and apply per-operand indexing maps to produce
      // src_indices on copy_tile and indices on tile_store.
      {
        auto indexingMaps = computeOp.getIndexingMapsArray();
        SmallVector<Value> iterIndices =
            getOrCreateIterIndices(builder, computeOp);

        // Populate copy_tile src_indices.
        SmallVector<CopyTileOp> copyTiles;
        for (Operation &op : *body) {
          if (auto ct = dyn_cast<CopyTileOp>(&op)) {
            if (ct.getSrcIndices().empty()) {
              copyTiles.push_back(ct);
            }
          }
        }
        for (CopyTileOp ct : copyTiles) {
          // copy_tile sources inside a compute body are always block arguments.
          auto blockArg = dyn_cast<BlockArgument>(ct.getSrc());
          assert(blockArg &&
                 "copy_tile src must be a block argument inside compute body");

          unsigned argIdx = blockArg.getArgNumber();
          // ComputeOp verifier guarantees block args match indexing maps.
          assert(argIdx < indexingMaps.size() &&
                 "block arg index out of range for indexing maps");
          AffineMap inputMap = indexingMaps[argIdx];

          builder.setInsertionPoint(ct);
          SmallVector<Value> cbIndices =
              applyIndexingMap(builder, ct.getLoc(), inputMap, iterIndices);

          auto newCopy = CopyTileOp::create(
              builder, ct.getLoc(),
              TypeRange{ct.getDstToken().getType(), ct.getDstTile().getType()},
              ct.getSrc(), cbIndices, ct.getDstIndex());
          for (NamedAttribute attr : ct->getAttrs()) {
            newCopy->setAttr(attr.getName(), attr.getValue());
          }
          ct.getDstToken().replaceAllUsesWith(newCopy.getDstToken());
          ct.getDstTile().replaceAllUsesWith(newCopy.getDstTile());
          ct.erase();
        }

        // All copy_tile ops must have populated src_indices at this point.
        for (Operation &op : *body) {
          if (auto ct = dyn_cast<CopyTileOp>(&op)) {
            if (ct.getSrcIndices().empty()) {
              ct.emitOpError()
                  << "copy_tile has empty src_indices after iter_index phase";
              signalPassFailure();
              return;
            }
          }
        }

        LLVM_DEBUG({
          llvm::dbgs() << "=== Populated CB indices using iter_index ("
                       << iterIndices.size() << "D) ===\n";
        });
      }

      //=== Compute and attach unroll_factor ===
      // unroll_factor = how many tiles can be processed per DST sync region.
      // dstPerIteration = DST registers used per single tile iteration.
      // unroll_factor = min(floor(capacity / dstPerIteration), totalTiles).
      if (!dstAssignment.empty()) {
        std::uint32_t dstPerIteration = maxDstUsed + 1;
        std::uint32_t unrollFactor = capacity / dstPerIteration;

        int64_t totalTiles = computeOp.getTotalIterationTiles();

        unrollFactor =
            std::min(unrollFactor, static_cast<std::uint32_t>(totalTiles));

        // Only attach if tiling is beneficial (factor > 1).
        if (unrollFactor > 1) {
          computeOp->setAttr(
              kUnrollFactorAttrName,
              builder.getI64IntegerAttr(static_cast<int64_t>(unrollFactor)));
        }

        LLVM_DEBUG({
          llvm::dbgs() << "DST per iteration: " << dstPerIteration
                       << ", capacity: " << capacity
                       << ", total tiles: " << totalTiles
                       << ", unroll_factor: " << unrollFactor << "\n";
        });
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
