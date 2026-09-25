// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify DFB Lifecycle
//===----------------------------------------------------------------------===//
//
// Verifies the visible reserve/push and wait/pop sequence of each kernel at
// each launch node against DFB capacity, and the visible acquisition totals
// across kernels. Only proven violations are rejected: a sequence the shared
// execution-count analysis cannot resolve is accepted.
//
//===----------------------------------------------------------------------===//

#include "DFBAcquireReleaseAnalysis.h"
#include "DFBProtocolDomainAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYDFBLIFECYCLE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// One producer- or consumer-side protocol effect and where it may execute.
struct DFBTransaction {
  Operation *op = nullptr;
  LaunchNodeDomain domain;
  /// Whether the effect acquires blocks (reserve or wait) rather than
  /// releasing them (push or pop).
  bool acquisition = false;
  /// Blocks transferred per execution; absent when the tile count is not a
  /// whole number of blocks.
  std::optional<int64_t> blocks;
};

using DFBTransactionMap = llvm::MapVector<int64_t, SmallVector<DFBTransaction>>;

ArrayRef<DFBTransaction> getDFBTransactions(const DFBTransactionMap &perDFB,
                                            int64_t logicalId) {
  auto transactionIt = perDFB.find(logicalId);
  if (transactionIt == perDFB.end()) {
    return {};
  }
  return transactionIt->second;
}

bool transactionMayExecuteAt(const DFBTransaction &transaction,
                             LaunchNodeCoord coord) {
  return !transaction.domain.known ||
         knownLaunchNodeDomainContains(transaction.domain, coord);
}

/// Total blocks released (pushed or popped) at one launch node, or absent
/// when unproven. Reserve and wait amounts are readiness thresholds that
/// opaque-call summaries need not pair with their releases, so the releases
/// count the transferred blocks. With `thread`, only that kernel's releases
/// are counted.
std::optional<std::uint64_t> getExactTransactionCount(
    ArrayRef<DFBTransaction> transactions, LaunchNodeCoord coord,
    const DFBProtocolDomainState &state, func::FuncOp thread = {}) {
  std::uint64_t total = 0;
  for (const DFBTransaction &transaction : transactions) {
    if (transaction.acquisition ||
        !transactionMayExecuteAt(transaction, coord)) {
      continue;
    }
    if (thread && getEnclosingKernelThread(transaction.op) != thread) {
      continue;
    }
    std::optional<std::uint64_t> maybeExecutions =
        getExactExecutionCountAtLaunchNode(transaction.op, coord, state);
    if (!transaction.blocks || !maybeExecutions) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> maybeBlocksExecuted = llvm::checkedMulUnsigned(
        static_cast<std::uint64_t>(*transaction.blocks), *maybeExecutions);
    if (!maybeBlocksExecuted) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> maybeNewTotal =
        llvm::checkedAddUnsigned(total, *maybeBlocksExecuted);
    if (!maybeNewTotal) {
      return std::nullopt;
    }
    total = *maybeNewTotal;
  }
  return total;
}

Operation *getTransactionAtNode(ArrayRef<DFBTransaction> producers,
                                ArrayRef<DFBTransaction> consumers,
                                LaunchNodeCoord coord,
                                func::FuncOp thread = {}) {
  Operation *transactionOp = nullptr;
  auto collect = [&](ArrayRef<DFBTransaction> transactions) {
    for (const DFBTransaction &transaction : transactions) {
      if (!transactionMayExecuteAt(transaction, coord)) {
        continue;
      }
      if (thread && getEnclosingKernelThread(transaction.op) != thread) {
        continue;
      }
      transactionOp = pickEarlierBySourceLoc(transactionOp, transaction.op);
    }
  };
  collect(producers);
  collect(consumers);
  assert(transactionOp && "transaction domain must contain the verified node");
  return transactionOp;
}

/// Net change and cumulative extrema of four block counters over one
/// sequential transaction fragment.
struct TransactionSequenceSummary {
  enum Counter : std::size_t {
    ProducerOpen,
    ConsumerOpen,
    Published,
    Capacity,
    Count
  };
  using CounterMask = std::uint8_t;

  static constexpr CounterMask getCounterMask(Counter counter) {
    return static_cast<CounterMask>(1U << counter);
  }

  static constexpr CounterMask getAllCounterMask() {
    return static_cast<CounterMask>((1U << Count) - 1);
  }

  std::array<std::int64_t, Count> net{};
  std::array<std::int64_t, Count> minimum{};
  std::array<std::int64_t, Count> maximum{};

  void setEvent(Counter counter, std::int64_t delta) {
    net[counter] = delta;
    minimum[counter] = std::min<int64_t>(0, delta);
    maximum[counter] = std::max<int64_t>(0, delta);
  }

  bool append(const TransactionSequenceSummary &next) {
    for (std::size_t counter = 0; counter < Count; ++counter) {
      std::optional<std::int64_t> shiftedMinimum =
          llvm::checkedAdd(net[counter], next.minimum[counter]);
      std::optional<std::int64_t> shiftedMaximum =
          llvm::checkedAdd(net[counter], next.maximum[counter]);
      std::optional<std::int64_t> combinedNet =
          llvm::checkedAdd(net[counter], next.net[counter]);
      if (!shiftedMinimum || !shiftedMaximum || !combinedNet) {
        return false;
      }
      minimum[counter] = std::min(minimum[counter], *shiftedMinimum);
      maximum[counter] = std::max(maximum[counter], *shiftedMaximum);
      net[counter] = *combinedNet;
    }
    return true;
  }

  /// Replace this fragment with `count` consecutive copies of itself. The
  /// extremum of copy `j` is `j * net` plus the fragment extremum, so the
  /// overall extremum is attained in the first or the last copy.
  bool repeat(std::uint64_t count) {
    assert(count > 0 && "repeating a fragment requires a positive count");
    if (count > static_cast<std::uint64_t>(INT64_MAX)) {
      return false;
    }
    auto signedCount = static_cast<std::int64_t>(count);
    for (std::size_t counter = 0; counter < Count; ++counter) {
      std::optional<std::int64_t> totalNet =
          llvm::checkedMul(net[counter], signedCount);
      std::optional<std::int64_t> lastCopyShift =
          llvm::checkedMul(net[counter], signedCount - 1);
      if (!totalNet || !lastCopyShift) {
        return false;
      }
      std::optional<std::int64_t> shiftedMinimum = llvm::checkedAdd(
          minimum[counter], std::min<std::int64_t>(0, *lastCopyShift));
      std::optional<std::int64_t> shiftedMaximum = llvm::checkedAdd(
          maximum[counter], std::max<std::int64_t>(0, *lastCopyShift));
      if (!shiftedMinimum || !shiftedMaximum) {
        return false;
      }
      minimum[counter] = *shiftedMinimum;
      maximum[counter] = *shiftedMaximum;
      net[counter] = *totalNet;
    }
    return true;
  }
};

/// Ordered summaries and conservative failures for every DFB in one thread.
struct TransactionSequenceResult {
  llvm::DenseMap<int64_t, TransactionSequenceSummary> summaries;
  llvm::DenseMap<int64_t, TransactionSequenceSummary::CounterMask>
      unknownCounters;

  void markUnknown(int64_t logicalId) {
    unknownCounters[logicalId] |=
        TransactionSequenceSummary::getAllCounterMask();
  }

  void markUnknown(int64_t logicalId,
                   TransactionSequenceSummary::Counter counter) {
    unknownCounters[logicalId] |=
        TransactionSequenceSummary::getCounterMask(counter);
  }

  bool
  isUnknown(int64_t logicalId,
            TransactionSequenceSummary::CounterMask requiredCounters) const {
    auto unknownIt = unknownCounters.find(logicalId);
    return unknownIt != unknownCounters.end() &&
           (unknownIt->second & requiredCounters) != 0;
  }

  void append(const TransactionSequenceResult &next) {
    for (const auto &[logicalId, unknownMask] : next.unknownCounters) {
      unknownCounters[logicalId] |= unknownMask;
    }
    for (const auto &[logicalId, nextSummary] : next.summaries) {
      auto [summaryIt, inserted] =
          summaries.try_emplace(logicalId, nextSummary);
      if (!inserted && !summaryIt->second.append(nextSummary)) {
        markUnknown(logicalId);
      }
    }
  }

  void repeat(std::uint64_t count) {
    for (auto &[logicalId, summary] : summaries) {
      if (!summary.repeat(count)) {
        markUnknown(logicalId);
      }
    }
  }
};

/// Summarizes one kernel's visible transactions at one launch node in program
/// order. Whether and how often a nested region executes comes from the shared
/// launch-location execution counts; structure those counts do not resolve
/// marks the enclosed DFBs unknown.
class DFBTransactionSequenceAnalysis {
public:
  DFBTransactionSequenceAnalysis(func::FuncOp thread, LaunchNodeCoord coord,
                                 const DFBProtocolDomainState &state)
      : thread(thread), coord(coord), state(state) {
    thread.walk([&](Operation *op) {
      if (isa<ResetAllDFBsOp>(op)) {
        allStateDiscarded = true;
        return;
      }
      if (auto reconfiguration = dyn_cast<DFBReconfigurationOp>(op)) {
        if (reconfiguration.getBoundary().getDiscardDfbState()) {
          allStateDiscarded = true;
        }
        return;
      }
      if (auto reset = dyn_cast<ResetDFBsOp>(op)) {
        for (Value dfb : reset.getDfbs()) {
          FailureOr<int64_t> logicalId = getDFBId(dfb);
          assert(succeeded(logicalId) && "DFB identities were verified");
          unmodeledDFBIds.insert(*logicalId);
        }
        return;
      }
      auto access = dyn_cast<DFBAccessOpInterface>(op);
      if (!access) {
        return;
      }
      SmallVector<DFBProtocolEffect> effects = access.getDFBProtocolEffects();
      if (effects.empty()) {
        return;
      }
      protocolOps.insert(op);
      for (const DFBProtocolEffect &effect : effects) {
        FailureOr<int64_t> logicalId = getDFBId(effect.dfb);
        assert(succeeded(logicalId) && "DFB identities were verified");
        allDFBIds.insert(*logicalId);
        // Opaque-call summaries state Metal-level actions whose reserve and
        // wait amounts are thresholds, not paired acquisitions.
        if (isa<OpaqueCallOp>(op)) {
          unmodeledDFBIds.insert(*logicalId);
        }
        for (Operation *ancestor = op->getParentOp();
             ancestor != thread.getOperation();
             ancestor = ancestor->getParentOp()) {
          auto &ids = nestedDFBIds[ancestor];
          if (!llvm::is_contained(ids, *logicalId)) {
            ids.push_back(*logicalId);
          }
        }
      }
      for (Region *region = op->getParentRegion(); region != &thread.getBody();
           region = region->getParentRegion()) {
        regionsWithProtocolOps.insert(region);
      }
    });
  }

  TransactionSequenceResult run() {
    if (!llvm::hasSingleElement(thread.getBody()) || allStateDiscarded) {
      return unknown(allDFBIds);
    }
    TransactionSequenceResult result =
        summarizeBlock(thread.getBody().front(), 1);
    for (int64_t logicalId : unmodeledDFBIds) {
      result.markUnknown(logicalId);
    }
    return result;
  }

private:
  using DFBIdRange = llvm::SmallSetVector<int64_t, 4>;

  /// Executions of the enclosing block at this node; absent inside a loop
  /// whose trip count is unresolved.
  using Executions = std::optional<std::uint64_t>;

  TransactionSequenceResult unknown(const DFBIdRange &logicalIds) const {
    TransactionSequenceResult result;
    for (int64_t logicalId : logicalIds) {
      result.markUnknown(logicalId);
    }
    return result;
  }

  TransactionSequenceResult unknown(Operation *op) const {
    auto idsIt = nestedDFBIds.find(op);
    if (idsIt == nestedDFBIds.end()) {
      return TransactionSequenceResult();
    }
    DFBIdRange logicalIds;
    logicalIds.insert(idsIt->second.begin(), idsIt->second.end());
    return unknown(logicalIds);
  }

  static Executions multiplyExecutions(Executions executions,
                                       std::uint64_t count) {
    if (!executions) {
      return std::nullopt;
    }
    return llvm::checkedMulUnsigned(*executions, count);
  }

  IntegerExpressionEvaluator::ValueEvaluator getValueEvaluator() const {
    return [this](Value value) {
      return evaluateIntegerAtLaunchLocation(
          value, LaunchExecutionLocation(coord), state);
    };
  }

  TransactionSequenceResult summarizeBlock(Block &block,
                                           Executions executions) {
    TransactionSequenceResult result;
    for (Operation &op : block) {
      if (protocolOps.contains(&op)) {
        result.append(summarizeProtocolOp(&op));
      } else if (nestedDFBIds.contains(&op)) {
        result.append(summarizeRegionOp(&op, executions));
      }
    }
    return result;
  }

  TransactionSequenceResult summarizeRegion(Region &region,
                                            Executions executions) {
    if (!llvm::hasSingleElement(region)) {
      return unknown(region.getParentOp());
    }
    return summarizeBlock(region.front(), executions);
  }

  TransactionSequenceResult summarizeProtocolOp(Operation *op) {
    TransactionSequenceResult result;
    for (const DFBProtocolEffect &effect :
         cast<DFBAccessOpInterface>(op).getDFBProtocolEffects()) {
      FailureOr<int64_t> logicalId = getDFBId(effect.dfb);
      assert(succeeded(logicalId) && "DFB identities were verified");
      std::optional<int64_t> blocks = getDFBProtocolEffectBlockCount(effect);
      if (!blocks) {
        result.markUnknown(*logicalId);
        continue;
      }
      TransactionSequenceSummary event;
      switch (effect.kind) {
      case DFBProtocolEffectKind::Reserve:
        event.setEvent(TransactionSequenceSummary::ProducerOpen, *blocks);
        event.setEvent(TransactionSequenceSummary::Capacity, *blocks);
        break;
      case DFBProtocolEffectKind::Push:
        event.setEvent(TransactionSequenceSummary::ProducerOpen, -*blocks);
        event.setEvent(TransactionSequenceSummary::Published, *blocks);
        break;
      case DFBProtocolEffectKind::Wait:
        event.setEvent(TransactionSequenceSummary::ConsumerOpen, *blocks);
        event.setEvent(TransactionSequenceSummary::Published, -*blocks);
        break;
      case DFBProtocolEffectKind::Pop:
        event.setEvent(TransactionSequenceSummary::ConsumerOpen, -*blocks);
        event.setEvent(TransactionSequenceSummary::Capacity, -*blocks);
        break;
      }
      TransactionSequenceResult single;
      single.summaries.try_emplace(*logicalId, event);
      result.append(single);
    }
    return result;
  }

  /// Invocations of `region` per execution of its parent, or absent when
  /// unproven. TTL control operations state the count directly. Otherwise the
  /// region must execute at most once per parent execution, which the ratio
  /// of exact counts proves; a repeated region could distribute unevenly
  /// across parent executions, so it stays unproven.
  std::optional<std::uint64_t>
  getRegionInvocationsPerExecution(Region &region, Executions executions) {
    if (std::optional<std::uint64_t> count =
            getRegionInvocationCountAtLaunchLocation(
                region, LaunchExecutionLocation(coord), state)) {
      return count;
    }
    if (!executions || *executions == 0 || !region.hasOneBlock() ||
        region.front().empty()) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> total = getExactExecutionCountAtLaunchNode(
        &region.front().front(), coord, state);
    if (!total || *total % *executions != 0) {
      return std::nullopt;
    }
    std::uint64_t perExecution = *total / *executions;
    if (perExecution > 1) {
      return std::nullopt;
    }
    return perExecution;
  }

  TransactionSequenceResult summarizeLoop(LoopLikeOpInterface loop,
                                          Executions executions) {
    Operation *op = loop.getOperation();
    if (op->getNumRegions() != 1) {
      return unknown(op);
    }
    Region &body = op->getRegion(0);
    std::optional<std::uint64_t> tripCount = tt::getLoopTripCount(
        loop, LoopInductionBindings(), getValueEvaluator());
    if (!tripCount) {
      // A counter with zero net change per iteration has the same extrema for
      // every nonnegative trip count.
      TransactionSequenceResult oneIteration =
          summarizeRegion(body, std::nullopt);
      for (const auto &[logicalId, summary] : oneIteration.summaries) {
        for (std::size_t counter = 0;
             counter < TransactionSequenceSummary::Count; ++counter) {
          if (summary.net[counter] != 0) {
            oneIteration.markUnknown(
                logicalId,
                static_cast<TransactionSequenceSummary::Counter>(counter));
          }
        }
      }
      return oneIteration;
    }
    if (*tripCount == 0) {
      return TransactionSequenceResult();
    }
    TransactionSequenceResult result =
        summarizeRegion(body, multiplyExecutions(executions, *tripCount));
    result.repeat(*tripCount);
    return result;
  }

  TransactionSequenceResult summarizeRegionOp(Operation *op,
                                              Executions executions) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
      return summarizeLoop(loop, executions);
    }
    Region *activeRegion = nullptr;
    std::uint64_t activeCount = 0;
    for (Region &region : op->getRegions()) {
      if (!regionsWithProtocolOps.contains(&region)) {
        continue;
      }
      std::optional<std::uint64_t> count =
          getRegionInvocationsPerExecution(region, executions);
      if (!count) {
        return unknown(op);
      }
      if (*count == 0) {
        continue;
      }
      // Two active regions have no proven relative order.
      if (activeRegion) {
        return unknown(op);
      }
      activeRegion = &region;
      activeCount = *count;
    }
    if (!activeRegion) {
      return TransactionSequenceResult();
    }
    TransactionSequenceResult result = summarizeRegion(
        *activeRegion, multiplyExecutions(executions, activeCount));
    result.repeat(activeCount);
    return result;
  }

  func::FuncOp thread;
  LaunchNodeCoord coord;
  const DFBProtocolDomainState &state;
  llvm::DenseSet<Operation *> protocolOps;
  llvm::DenseMap<Operation *, SmallVector<int64_t>> nestedDFBIds;
  llvm::DenseSet<Region *> regionsWithProtocolOps;
  DFBIdRange allDFBIds;
  /// DFBs whose counters this kernel does not model: opaque-call effects or
  /// a selective reset discard the counted state.
  DFBIdRange unmodeledDFBIds;
  /// A reset of every DFB or a state-discarding reconfiguration discards the
  /// counted state of every DFB in the kernel.
  bool allStateDiscarded = false;
};

/// Reuse the structured sequence analysis across DFB queries in one thread.
class DFBTransactionSequenceCache {
public:
  explicit DFBTransactionSequenceCache(const DFBProtocolDomainState &state)
      : state(state) {}

  const TransactionSequenceResult &get(func::FuncOp thread,
                                       LaunchNodeCoord coord) {
    auto &resultsByCoord = results[thread.getOperation()];
    auto resultIt = resultsByCoord.find(coord);
    if (resultIt == resultsByCoord.end()) {
      resultIt =
          resultsByCoord
              .emplace(
                  coord,
                  DFBTransactionSequenceAnalysis(thread, coord, state).run())
              .first;
    }
    return resultIt->second;
  }

private:
  const DFBProtocolDomainState &state;
  llvm::DenseMap<Operation *,
                 std::map<LaunchNodeCoord, TransactionSequenceResult>>
      results;
};

std::optional<TransactionSequenceSummary> getTransactionSequenceSummary(
    const TransactionSequenceResult &sequence, int64_t logicalId,
    TransactionSequenceSummary::CounterMask requiredCounters) {
  if (sequence.isUnknown(logicalId, requiredCounters)) {
    return std::nullopt;
  }
  auto summaryIt = sequence.summaries.find(logicalId);
  return summaryIt == sequence.summaries.end() ? TransactionSequenceSummary()
                                               : summaryIt->second;
}

SmallVector<func::FuncOp>
getTransactionThreadsAtNode(ArrayRef<DFBTransaction> transactions,
                            LaunchNodeCoord coord) {
  SmallVector<func::FuncOp> threads;
  for (const DFBTransaction &transaction : transactions) {
    if (!transactionMayExecuteAt(transaction, coord)) {
      continue;
    }
    func::FuncOp transactionThread = getEnclosingKernelThread(transaction.op);
    assert(transactionThread && "recorded transaction must have a thread");
    if (!llvm::is_contained(threads, transactionThread)) {
      threads.push_back(transactionThread);
    }
  }
  return threads;
}

bool countsFitCapacity(std::uint64_t producerCount, std::uint64_t consumerCount,
                       std::uint64_t capacityBlocks) {
  return consumerCount <= producerCount &&
         producerCount - consumerCount <= capacityBlocks;
}

void attachDeclarationNote(InFlightDiagnostic &diagnostic,
                           Operation *bindSite) {
  if (bindSite) {
    diagnostic.attachNote(bindSite->getLoc())
        << "dataflow buffer declared here";
  }
}

/// A compute kernel runs producer effects on PACK and consumer effects on
/// UNPACK; every other kernel executes its effects in program order.
bool isSequentialKernel(func::FuncOp thread) {
  return getKernelThreadType(thread) != ttkernel::ThreadType::Compute;
}

/// Verify one role of a kernel on its own counter. In a compute kernel the
/// producer effects run on PACK and the consumer effects on UNPACK, so each
/// role is a separate sequential endpoint. Returns true when a violation was
/// reported.
bool verifyEndpointTransactionSequence(
    int64_t logicalId, ArrayRef<DFBTransaction> transactions,
    func::FuncOp thread, LaunchNodeCoord coord, std::uint64_t capacityBlocks,
    TransactionSequenceSummary::Counter openCounter, llvm::StringRef role,
    llvm::StringRef acquireName, llvm::StringRef releaseName,
    Operation *bindSite, DFBTransactionSequenceCache &sequenceCache) {
  const TransactionSequenceResult &sequence = sequenceCache.get(thread, coord);
  std::optional<TransactionSequenceSummary> maybeSummary =
      getTransactionSequenceSummary(
          sequence, logicalId,
          TransactionSequenceSummary::getCounterMask(openCounter));
  if (!maybeSummary) {
    return false;
  }
  bool ordered = maybeSummary->minimum[openCounter] >= 0;
  bool closed = maybeSummary->net[openCounter] == 0;
  bool capacitySafe = static_cast<std::uint64_t>(
                          maybeSummary->maximum[openCounter]) <= capacityBlocks;
  if (ordered && closed && capacitySafe) {
    return false;
  }

  Operation *transactionOp = getTransactionAtNode(
      transactions, ArrayRef<DFBTransaction>(), coord, thread);
  InFlightDiagnostic diagnostic = transactionOp->emitError();
  if (!ordered || !closed) {
    diagnostic << "logical DFB " << logicalId << " has an incomplete or "
               << "misordered " << role << " lifecycle on core_x=" << coord.x
               << ", core_y=" << coord.y;
    if (!ordered) {
      diagnostic.attachNote()
          << "the " << role << " kernel performs a " << releaseName
          << " before a matching " << acquireName;
    } else {
      std::int64_t open = maybeSummary->net[openCounter];
      diagnostic.attachNote()
          << "the " << role << " kernel finishes with " << open << " open "
          << acquireName << " block(s) without a " << releaseName;
    }
  } else {
    diagnostic << "logical DFB " << logicalId << " has a " << role
               << " lifecycle that can block on core_x=" << coord.x
               << ", core_y=" << coord.y;
    diagnostic.attachNote()
        << "the " << role << " kernel reaches "
        << maybeSummary->maximum[openCounter] << " open " << acquireName
        << " block(s), exceeding capacity " << capacityBlocks;
  }
  diagnostic.attachNote() << "keep each " << acquireName
                          << " followed by its matching " << releaseName
                          << " before opening more than DFB capacity";
  attachDeclarationNote(diagnostic, bindSite);
  return true;
}

/// Verify a data-movement kernel that performs both roles. Its single RISC
/// executes the effects in program order, so a wait needs an earlier push and
/// a reserve holds capacity until a pop. Returns true when a violation was
/// reported.
bool verifyCombinedTransactionSequence(
    int64_t logicalId, ArrayRef<DFBTransaction> producers,
    ArrayRef<DFBTransaction> consumers, func::FuncOp thread,
    LaunchNodeCoord coord, std::uint64_t capacityBlocks, Operation *bindSite,
    DFBTransactionSequenceCache &sequenceCache) {
  const TransactionSequenceResult &sequence = sequenceCache.get(thread, coord);
  std::optional<TransactionSequenceSummary> maybeSummary =
      getTransactionSequenceSummary(
          sequence, logicalId, TransactionSequenceSummary::getAllCounterMask());
  if (!maybeSummary) {
    return false;
  }
  const TransactionSequenceSummary &summary = *maybeSummary;
  bool prefixesValid = llvm::all_of(
      summary.minimum, [](std::int64_t value) { return value >= 0; });
  bool lifecyclesClosed =
      summary.net[TransactionSequenceSummary::ProducerOpen] == 0 &&
      summary.net[TransactionSequenceSummary::ConsumerOpen] == 0;
  bool capacitySafe =
      static_cast<std::uint64_t>(
          summary.maximum[TransactionSequenceSummary::Capacity]) <=
      capacityBlocks;
  if (prefixesValid && lifecyclesClosed && capacitySafe) {
    return false;
  }

  std::int64_t publishedMinimum =
      summary.minimum[TransactionSequenceSummary::Published];
  bool waitsBeforePublication = publishedMinimum < 0;
  bool incomplete =
      !waitsBeforePublication && (!prefixesValid || !lifecyclesClosed);
  Operation *transactionOp =
      getTransactionAtNode(producers, consumers, coord, thread);
  InFlightDiagnostic diagnostic =
      transactionOp->emitError()
      << "logical DFB " << logicalId
      << (incomplete ? " has an incomplete or misordered "
                       "lifecycle on core_x="
                     : " has capacity-unsafe producer and "
                       "consumer transactions on core_x=")
      << coord.x << ", core_y=" << coord.y;
  if (waitsBeforePublication) {
    std::uint64_t deficit =
        static_cast<std::uint64_t>(-(publishedMinimum + 1)) + 1;
    diagnostic.attachNote() << "the single kernel waits for " << deficit
                            << " block(s) before they are visibly published";
  } else if (incomplete) {
    diagnostic.attachNote() << "the single kernel has an incomplete or "
                               "misordered reserve/push or wait/pop lifecycle";
  } else {
    diagnostic.attachNote()
        << "the single kernel reaches "
        << summary.maximum[TransactionSequenceSummary::Capacity]
        << " outstanding block(s), exceeding capacity " << capacityBlocks;
  }
  diagnostic.attachNote()
      << "keep reserve/push and wait/pop ordered, and pop consumed blocks "
         "before the outstanding count exceeds DFB capacity";
  attachDeclarationNote(diagnostic, bindSite);
  return true;
}

/// Compare visible acquisition totals of distinct producer and consumer
/// kernels. Returns true when a violation was reported.
bool verifyCrossKernelTransactionCounts(int64_t logicalId,
                                        ArrayRef<DFBTransaction> producers,
                                        ArrayRef<DFBTransaction> consumers,
                                        LaunchNodeCoord coord,
                                        std::uint64_t capacityBlocks,
                                        Operation *bindSite,
                                        const DFBProtocolDomainState &state) {
  std::optional<std::uint64_t> producerCount =
      getExactTransactionCount(producers, coord, state);
  std::optional<std::uint64_t> consumerCount =
      getExactTransactionCount(consumers, coord, state);
  if (!producerCount || !consumerCount ||
      countsFitCapacity(*producerCount, *consumerCount, capacityBlocks)) {
    return false;
  }

  Operation *transactionOp = getTransactionAtNode(producers, consumers, coord);
  InFlightDiagnostic diagnostic = transactionOp->emitError()
                                  << "logical DFB " << logicalId
                                  << " has capacity-unsafe producer and "
                                     "consumer transactions on core_x="
                                  << coord.x << ", core_y=" << coord.y;
  if (*consumerCount > *producerCount) {
    diagnostic.attachNote() << "the consumer pops " << *consumerCount
                            << " block(s) per launch, but the producer pushes "
                            << *producerCount << " block(s) per launch";
  } else {
    diagnostic.attachNote()
        << "the producer pushes " << *producerCount
        << " block(s) per launch and the consumer pops " << *consumerCount
        << ", leaving " << *producerCount - *consumerCount
        << " outstanding block(s) for capacity " << capacityBlocks;
  }
  diagnostic.attachNote()
      << "keep waits within visible publications and outstanding "
         "publications within DFB capacity on every active node";
  attachDeclarationNote(diagnostic, bindSite);
  return true;
}

/// Verify one logical DFB on every launch node where a transaction may
/// execute. Nodes where an opaque call may perform protocol actions the IR
/// does not represent are skipped. With `stateDiscarded`, a reset or a
/// state-discarding reconfiguration breaks the launch into intervals the
/// totals do not model, so the cross-kernel comparison is skipped. Returns
/// true when a violation was reported.
bool verifyDFBTransactions(int64_t logicalId,
                           ArrayRef<DFBTransaction> producers,
                           ArrayRef<DFBTransaction> consumers,
                           std::uint64_t capacityBlocks,
                           const LaunchNodeDomain &externalProtocolDomain,
                           bool stateDiscarded, Operation *bindSite,
                           const DFBProtocolDomainState &state,
                           DFBTransactionSequenceCache &sequenceCache) {
  assert((!producers.empty() || !consumers.empty()) &&
         "transaction verification requires a protocol effect");
  LaunchNodeDomain transactionDomain;
  for (const DFBTransaction &transaction : producers) {
    transactionDomain = transactionDomain.unionWith(transaction.domain);
  }
  for (const DFBTransaction &transaction : consumers) {
    transactionDomain = transactionDomain.unionWith(transaction.domain);
  }
  const LaunchNodeDomain &verificationDomain =
      transactionDomain.known ? transactionDomain : state.baseDomain;

  const std::set<LaunchNodeCoord> *externalNodes =
      externalProtocolDomain.getUpperBoundNodes();
  for (LaunchNodeCoord coord : verificationDomain.nodes) {
    if (!externalNodes || externalNodes->count(coord) != 0) {
      continue;
    }
    SmallVector<func::FuncOp> producerThreads =
        getTransactionThreadsAtNode(producers, coord);
    SmallVector<func::FuncOp> consumerThreads =
        getTransactionThreadsAtNode(consumers, coord);
    SmallVector<func::FuncOp> threads = producerThreads;
    for (func::FuncOp consumerThread : consumerThreads) {
      if (!llvm::is_contained(threads, consumerThread)) {
        threads.push_back(consumerThread);
      }
    }
    // A data-movement kernel performing both roles runs them on one RISC and
    // is checked on its combined sequence. A compute kernel runs producer
    // effects on PACK and consumer effects on UNPACK, so each role is its own
    // endpoint, as it is for a kernel performing one role.
    for (func::FuncOp thread : threads) {
      bool producer = llvm::is_contained(producerThreads, thread);
      bool consumer = llvm::is_contained(consumerThreads, thread);
      if (producer && consumer && isSequentialKernel(thread)) {
        if (verifyCombinedTransactionSequence(logicalId, producers, consumers,
                                              thread, coord, capacityBlocks,
                                              bindSite, sequenceCache)) {
          return true;
        }
        continue;
      }
      if (producer && verifyEndpointTransactionSequence(
                          logicalId, producers, thread, coord, capacityBlocks,
                          TransactionSequenceSummary::ProducerOpen, "producer",
                          "reserve", "push", bindSite, sequenceCache)) {
        return true;
      }
      if (consumer && verifyEndpointTransactionSequence(
                          logicalId, consumers, thread, coord, capacityBlocks,
                          TransactionSequenceSummary::ConsumerOpen, "consumer",
                          "wait", "pop", bindSite, sequenceCache)) {
        return true;
      }
    }
    // Exact counts prove that every counted kernel acts on this node, so the
    // totals are compared regardless of ownership relaxation.
    bool singleSequentialKernel =
        threads.size() == 1 && !producerThreads.empty() &&
        !consumerThreads.empty() && isSequentialKernel(threads.front());
    if (!singleSequentialKernel && !stateDiscarded &&
        verifyCrossKernelTransactionCounts(logicalId, producers, consumers,
                                           coord, capacityBlocks, bindSite,
                                           state)) {
      return true;
    }
  }
  return false;
}

struct TTLVerifyDFBLifecyclePass
    : public impl::TTLVerifyDFBLifecycleBase<TTLVerifyDFBLifecyclePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyResolvedDFBIdentities(module, getArgument()))) {
      signalPassFailure();
      return;
    }
    FailureOr<llvm::DenseMap<int64_t, BindCBOp>> bindSites =
        collectFinalizedDFBBindSites(module);
    if (failed(bindSites)) {
      signalPassFailure();
      return;
    }
    if (!hasDFBProtocolEffect(module)) {
      return;
    }

    DFBProtocolDomainState state;
    if (failed(
            initializeDFBProtocolDomainState(module, getArgument(), state)) ||
        failed(analyzeDFBProtocolDomains(module, state))) {
      signalPassFailure();
      return;
    }

    DFBTransactionMap producersByDFB;
    DFBTransactionMap consumersByDFB;
    llvm::SetVector<int64_t> transactionDFBIds;
    module.walk([&](DFBAccessOpInterface access) {
      if (!getEnclosingKernelThread(access)) {
        return;
      }
      LaunchNodeDomain domain = state.getProtocolActionDomain(access).domain;
      for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
        DFBTransactionMap &perDFB = isProducerDFBProtocolEffect(effect.kind)
                                        ? producersByDFB
                                        : consumersByDFB;
        bool acquisition = effect.kind == DFBProtocolEffectKind::Reserve ||
                           effect.kind == DFBProtocolEffectKind::Wait;
        FailureOr<int64_t> dfbId = getDFBId(effect.dfb);
        assert(succeeded(dfbId) && "DFB identities were verified");
        perDFB[*dfbId].push_back({access, domain, acquisition,
                                  getDFBProtocolEffectBlockCount(effect)});
        transactionDFBIds.insert(*dfbId);
      }
    });

    llvm::DenseMap<int64_t, LaunchNodeDomain> externalProtocolDomainsByDFB;
    for (const auto &[dfbId, calls] :
         collectDFBsWithOpaqueProtocolActions(module, *bindSites)) {
      LaunchNodeDomain domain;
      for (OpaqueCallOp call : calls) {
        domain = domain.unionWith(state.getExternalCallDomain(call));
      }
      externalProtocolDomainsByDFB[dfbId] = domain;
    }

    bool allStateDiscarded = false;
    llvm::DenseSet<int64_t> stateDiscardedDFBIds;
    module.walk([&](Operation *op) {
      if (isa<ResetAllDFBsOp>(op)) {
        allStateDiscarded = true;
      } else if (auto reconfiguration = dyn_cast<DFBReconfigurationOp>(op)) {
        allStateDiscarded |= reconfiguration.getBoundary().getDiscardDfbState();
      } else if (auto reset = dyn_cast<ResetDFBsOp>(op)) {
        for (Value dfb : reset.getDfbs()) {
          FailureOr<int64_t> dfbId = getDFBId(dfb);
          assert(succeeded(dfbId) && "DFB identities were verified");
          stateDiscardedDFBIds.insert(*dfbId);
        }
      }
    });

    DFBTransactionSequenceCache sequenceCache(state);
    bool sawError = false;
    for (int64_t logicalId : transactionDFBIds) {
      BindCBOp bindSite = bindSites->lookup(logicalId);
      assert(bindSite && "every transaction must have a DFB declaration");
      auto dfbType = cast<CircularBufferType>(bindSite.getResult().getType());
      sawError |= verifyDFBTransactions(
          logicalId, getDFBTransactions(producersByDFB, logicalId),
          getDFBTransactions(consumersByDFB, logicalId),
          static_cast<std::uint64_t>(dfbType.getBlockCount()),
          externalProtocolDomainsByDFB.lookup(logicalId),
          allStateDiscarded || stateDiscardedDFBIds.contains(logicalId),
          bindSite, state, sequenceCache);
    }
    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
