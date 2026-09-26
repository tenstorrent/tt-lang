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
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
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
#include <functional>
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

/// Total blocks released (pushed or popped) at one launch node, or with
/// `acquisitions` the blocks reserved or waited for; absent when unproven.
/// With `thread`, only that kernel's effects are counted.
std::optional<std::uint64_t>
getExactTransactionCount(ArrayRef<DFBTransaction> transactions,
                         LaunchNodeCoord coord,
                         const DFBProtocolDomainState &state,
                         func::FuncOp thread = {}, bool acquisitions = false) {
  std::uint64_t total = 0;
  for (const DFBTransaction &transaction : transactions) {
    if (transaction.acquisition != acquisitions ||
        !transactionMayExecuteAt(transaction, coord)) {
      continue;
    }
    // An opaque call's reserve or wait is a readiness threshold, not an
    // acquisition that transfers blocks.
    if (acquisitions && isa<OpaqueCallOp>(transaction.op)) {
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

/// Net change and cumulative extrema of the block counters over one sequential
/// transaction fragment.
struct TransactionSequenceSummary {
  enum Counter : std::size_t {
    ProducerOpen,
    ConsumerOpen,
    Published,
    Capacity,
    /// Blocks pushed, popped, and waited for by user operations; only
    /// increase, so their net is the total.
    Pushed,
    Popped,
    Waited,
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

  bool operator==(const TransactionSequenceSummary &other) const {
    return net == other.net && minimum == other.minimum &&
           maximum == other.maximum;
  }

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

/// The summaries of one DFB in one thread, one per interval between
/// synchronized resets (or state-discarding reconfigurations) in program
/// order. A reset restores the empty state, so each interval starts at zero.
using TransactionSegments = SmallVector<TransactionSequenceSummary, 1>;

/// A synchronized reset or state-discarding reconfiguration, by declaration
/// ordinal. The two kinds have separate ordinal spaces.
struct ExecutedStateDiscard {
  bool reconfiguration = false;
  int64_t ordinal = 0;

  bool operator==(const ExecutedStateDiscard &other) const {
    return reconfiguration == other.reconfiguration && ordinal == other.ordinal;
  }
  bool operator<(const ExecutedStateDiscard &other) const {
    return std::tie(reconfiguration, ordinal) <
           std::tie(other.reconfiguration, other.ordinal);
  }
};

/// The value an unresolved `scf.if` condition took on the way to a sequence.
using ConditionDecision = DispatchConditionFormulas::Literal;

/// One possible sequence of a DFB: its segments, the state discards executed
/// on it, and the path condition over dispatch conditions under which it
/// executes. Sequences of one kernel or of different kernels belong to the
/// same execution only when the conjunction of their path conditions is
/// satisfiable.
struct TransactionAlternative {
  TransactionSegments segments;
  SmallVector<ExecutedStateDiscard, 2> executedDiscards;
  DispatchConditionFormulas::FormulaId pathCondition =
      DispatchConditionFormulas::kTrue;

  bool sameSequence(const TransactionAlternative &other) const {
    return segments == other.segments &&
           executedDiscards == other.executedDiscards;
  }
};

/// The possible sequences of one DFB: one entry unless a reset under an
/// unresolved condition makes both its execution and its omission possible.
using TransactionAlternatives = SmallVector<TransactionAlternative, 1>;

/// Ordered segment summaries and conservative failures for every DFB in one
/// thread.
struct TransactionSequenceResult {
  /// Segments per sequence beyond this many are not expanded; the DFB
  /// becomes unknown instead.
  static constexpr std::size_t kMaxSegments = 1024;
  /// Alternatives per DFB beyond this many are not combined; the DFB becomes
  /// unknown instead. Also bounds the cross-kernel combinations checked at a
  /// node.
  static constexpr std::size_t kMaxAlternatives = 64;

  llvm::DenseMap<int64_t, TransactionAlternatives> alternatives;
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

  void markUnknown(int64_t logicalId,
                   TransactionSequenceSummary::CounterMask counters) {
    unknownCounters[logicalId] |= counters;
  }

  bool
  isUnknown(int64_t logicalId,
            TransactionSequenceSummary::CounterMask requiredCounters) const {
    auto unknownIt = unknownCounters.find(logicalId);
    return unknownIt != unknownCounters.end() &&
           (unknownIt->second & requiredCounters) != 0;
  }

  /// Whether nothing about `logicalId` is known any more.
  bool isAbandoned(int64_t logicalId) const {
    auto unknownIt = unknownCounters.find(logicalId);
    return unknownIt != unknownCounters.end() &&
           unknownIt->second == TransactionSequenceSummary::getAllCounterMask();
  }

  /// Give up on `logicalId`: it is unknown and keeps no sequence, so later
  /// appends and repeats do not combine what can no longer be checked.
  void abandon(int64_t logicalId) {
    markUnknown(logicalId);
    alternatives.erase(logicalId);
  }

  /// The alternatives of `logicalId`, creating the single empty sequence.
  TransactionAlternatives &getOrCreate(int64_t logicalId) {
    TransactionAlternatives &own = alternatives[logicalId];
    if (own.empty()) {
      own.push_back({TransactionSegments{TransactionSequenceSummary()},
                     {},
                     DispatchConditionFormulas::kTrue});
    }
    return own;
  }

  /// Add `alternative`, or merge it into the entry with the same sequence,
  /// which then executes under either path condition. Returns false when the
  /// budget is exceeded.
  static bool insertAlternative(TransactionAlternatives &into,
                                TransactionAlternative alternative,
                                DispatchConditionFormulas &formulas) {
    auto *existing = llvm::find_if(into, [&](const TransactionAlternative &e) {
      return e.sameSequence(alternative);
    });
    if (existing == into.end()) {
      into.push_back(std::move(alternative));
    } else {
      existing->pathCondition =
          formulas.disjoin(existing->pathCondition, alternative.pathCondition);
    }
    return into.size() <= kMaxAlternatives;
  }

  /// Start a new segment for `logicalId`: the reset boundary. A DFB without
  /// a sequence yet gets the empty segment that precedes the boundary.
  void startSegment(int64_t logicalId) {
    TransactionAlternatives &own = alternatives[logicalId];
    if (own.empty()) {
      own.push_back({TransactionSegments{TransactionSequenceSummary()},
                     {},
                     DispatchConditionFormulas::kTrue});
      return;
    }
    for (TransactionAlternative &alternative : own) {
      alternative.segments.emplace_back();
    }
  }

  enum class JoinResult { Joined, Contradiction, Overflow };

  /// Continue `own` with `next`: the last segment of `own` joins the first
  /// segment of `next`, the discards accumulate, and the path conditions
  /// conjoin.
  static JoinResult join(TransactionAlternative &own,
                         const TransactionAlternative &next,
                         DispatchConditionFormulas &formulas) {
    DispatchConditionFormulas::FormulaId pathCondition =
        formulas.conjoin(own.pathCondition, next.pathCondition);
    if (!formulas.satisfiable(pathCondition)) {
      return JoinResult::Contradiction;
    }
    own.pathCondition = pathCondition;
    if (!own.segments.back().append(next.segments.front())) {
      return JoinResult::Overflow;
    }
    own.segments.append(std::next(next.segments.begin()), next.segments.end());
    own.executedDiscards.append(next.executedDiscards.begin(),
                                next.executedDiscards.end());
    llvm::sort(own.executedDiscards);
    return own.segments.size() <= kMaxSegments ? JoinResult::Joined
                                               : JoinResult::Overflow;
  }

  /// Record that every sequence of `logicalId` executed `discard`.
  void recordDiscard(int64_t logicalId, ExecutedStateDiscard discard) {
    for (TransactionAlternative &alternative : getOrCreate(logicalId)) {
      alternative.executedDiscards.push_back(discard);
      llvm::sort(alternative.executedDiscards);
    }
  }

  void append(const TransactionSequenceResult &next,
              DispatchConditionFormulas &formulas) {
    for (const auto &[logicalId, unknownMask] : next.unknownCounters) {
      unknownCounters[logicalId] |= unknownMask;
    }
    for (const auto &[logicalId, nextAlternatives] : next.alternatives) {
      if (isAbandoned(logicalId)) {
        alternatives.erase(logicalId);
        continue;
      }
      auto [ownIt, inserted] =
          alternatives.try_emplace(logicalId, nextAlternatives);
      if (inserted) {
        continue;
      }
      TransactionAlternatives combined;
      bool overflow = false;
      for (const TransactionAlternative &own : ownIt->second) {
        for (const TransactionAlternative &following : nextAlternatives) {
          TransactionAlternative joined = own;
          JoinResult joinResult = join(joined, following, formulas);
          if (joinResult == JoinResult::Contradiction) {
            continue;
          }
          if (joinResult == JoinResult::Overflow ||
              !insertAlternative(combined, std::move(joined), formulas)) {
            overflow = true;
            break;
          }
        }
        if (overflow) {
          break;
        }
      }
      // Every combination contradicting itself means the decisions were
      // recorded inconsistently; nothing is known then.
      if (overflow || combined.empty()) {
        abandon(logicalId);
        continue;
      }
      ownIt->second = std::move(combined);
    }
  }

  /// Replace every DFB's sequence with `count` consecutive copies. The last
  /// segment of one copy continues into the first segment of the next. A DFB
  /// with several alternatives could choose differently per copy; it becomes
  /// unknown.
  void repeat(std::uint64_t count) {
    if (count == 1) {
      return;
    }
    SmallVector<int64_t> abandoned;
    for (auto &[logicalId, own] : alternatives) {
      if (own.size() != 1) {
        abandoned.push_back(logicalId);
        continue;
      }
      TransactionSegments &segments = own.front().segments;
      if (segments.size() == 1) {
        if (!segments.front().repeat(count)) {
          abandoned.push_back(logicalId);
        }
        continue;
      }
      std::size_t inner = segments.size() - 1;
      if (count > kMaxSegments || inner * count + 1 > kMaxSegments) {
        abandoned.push_back(logicalId);
        continue;
      }
      TransactionSegments expanded;
      expanded.push_back(segments.front());
      for (std::uint64_t copy = 1; copy <= count; ++copy) {
        expanded.append(std::next(segments.begin()), std::prev(segments.end()));
        TransactionSequenceSummary joined = segments.back();
        if (copy < count) {
          if (!joined.append(segments.front())) {
            abandoned.push_back(logicalId);
          }
        }
        expanded.push_back(joined);
      }
      segments = std::move(expanded);
    }
    for (int64_t logicalId : abandoned) {
      abandon(logicalId);
    }
  }
};

/// The DFBs whose protocol state a reset or reconfiguration discards: every
/// DFB, or the listed ones.
struct StateDiscardTargets {
  bool all = false;
  SmallVector<int64_t> logicalIds;
  ExecutedStateDiscard discard;
};

std::optional<StateDiscardTargets> getStateDiscardTargets(Operation *op) {
  if (auto resetAll = dyn_cast<ResetAllDFBsOp>(op)) {
    return StateDiscardTargets{
        true, {}, {false, resetAll.getReset().getOrdinal()}};
  }
  if (auto reconfiguration = dyn_cast<DFBReconfigurationOp>(op)) {
    if (!reconfiguration.getBoundary().getDiscardDfbState()) {
      return std::nullopt;
    }
    return StateDiscardTargets{
        true, {}, {true, reconfiguration.getBoundary().getOrdinal()}};
  }
  auto reset = dyn_cast<ResetDFBsOp>(op);
  if (!reset) {
    return std::nullopt;
  }
  StateDiscardTargets targets;
  targets.discard = {false, reset.getReset().getOrdinal()};
  for (Value dfb : reset.getDfbs()) {
    FailureOr<int64_t> logicalId = getDFBId(dfb);
    assert(succeeded(logicalId) && "DFB identities were verified");
    targets.logicalIds.push_back(*logicalId);
  }
  return targets;
}

/// Summarizes one kernel's visible transactions at one launch node in program
/// order. Whether and how often a nested region executes comes from the shared
/// launch-location execution counts; structure those counts do not resolve
/// marks the enclosed DFBs unknown.
class DFBTransactionSequenceAnalysis {
public:
  DFBTransactionSequenceAnalysis(func::FuncOp thread, LaunchNodeCoord coord,
                                 const DFBProtocolDomainState &state,
                                 DispatchConditionFormulas &conditionFormulas)
      : thread(thread), coord(coord), state(state),
        conditionFormulas(conditionFormulas) {
    thread.walk([&](Operation *op) {
      SmallVector<int64_t> touched;
      SmallVector<TransactionSequenceSummary::CounterMask> touchedCounters;
      if (std::optional<StateDiscardTargets> targets =
              getStateDiscardTargets(op)) {
        if (targets->all) {
          discardsEveryDFB.insert(op);
        }
        discardTargetIds[op] = targets->logicalIds;
        discards[op] = targets->discard;
        touched = targets->logicalIds;
        touchedCounters.assign(touched.size(),
                               TransactionSequenceSummary::getAllCounterMask());
      } else if (auto access = dyn_cast<DFBAccessOpInterface>(op)) {
        SmallVector<DFBProtocolEffect> effects = access.getDFBProtocolEffects();
        if (effects.empty()) {
          return;
        }
        protocolOps.insert(op);
        for (const DFBProtocolEffect &effect : effects) {
          FailureOr<int64_t> logicalId = getDFBId(effect.dfb);
          assert(succeeded(logicalId) && "DFB identities were verified");
          touched.push_back(*logicalId);
          touchedCounters.push_back(
              getEffectCounterMask(effect.kind, isa<OpaqueCallOp>(op)));
          if (isa<OpaqueCallOp>(op) &&
              effect.kind == DFBProtocolEffectKind::Pop) {
            opaquePopDFBIds.insert(*logicalId);
          }
        }
      } else {
        return;
      }
      allDFBIds.insert(touched.begin(), touched.end());
      for (Operation *ancestor = op->getParentOp();
           ancestor != thread.getOperation();
           ancestor = ancestor->getParentOp()) {
        auto &ids = nestedDFBIds[ancestor];
        auto &counters = nestedCounters[ancestor];
        for (auto [logicalId, mask] : llvm::zip(touched, touchedCounters)) {
          if (!llvm::is_contained(ids, logicalId)) {
            ids.push_back(logicalId);
          }
          counters[logicalId] |= mask;
        }
        if (discardsEveryDFB.contains(op)) {
          nestedEveryDFBDiscard.insert(ancestor);
        }
        if (discardTargetIds.contains(op)) {
          auto &resetIds = nestedDiscardIds[ancestor];
          for (int64_t logicalId : touched) {
            if (!llvm::is_contained(resetIds, logicalId)) {
              resetIds.push_back(logicalId);
            }
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
    if (!llvm::hasSingleElement(thread.getBody())) {
      return unknown(allDFBIds);
    }
    TransactionSequenceResult result =
        summarizeBlock(thread.getBody().front(), 1);
    // Waits stand in for pops only where every pop is a user pop; an opaque
    // pop has no user wait.
    for (int64_t logicalId : opaquePopDFBIds) {
      result.markUnknown(logicalId, TransactionSequenceSummary::Waited);
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

  /// Every DFB an operation nested in `op` touches; an operation discarding
  /// every DFB's state touches every DFB of the kernel.
  DFBIdRange getNestedDFBIds(Operation *op) const {
    DFBIdRange logicalIds;
    if (nestedEveryDFBDiscard.contains(op)) {
      logicalIds.insert(allDFBIds.begin(), allDFBIds.end());
      return logicalIds;
    }
    auto idsIt = nestedDFBIds.find(op);
    if (idsIt != nestedDFBIds.end()) {
      logicalIds.insert(idsIt->second.begin(), idsIt->second.end());
    }
    return logicalIds;
  }

  /// The counters a user effect of `kind` changes; an opaque call's reserve
  /// and wait thresholds change none.
  static TransactionSequenceSummary::CounterMask
  getEffectCounterMask(DFBProtocolEffectKind kind, bool opaque) {
    using Summary = TransactionSequenceSummary;
    switch (kind) {
    case DFBProtocolEffectKind::Reserve:
      return opaque ? 0
                    : Summary::getCounterMask(Summary::ProducerOpen) |
                          Summary::getCounterMask(Summary::Capacity);
    case DFBProtocolEffectKind::Push:
      return Summary::getCounterMask(Summary::ProducerOpen) |
             Summary::getCounterMask(Summary::Published) |
             Summary::getCounterMask(Summary::Pushed);
    case DFBProtocolEffectKind::Wait:
      return opaque ? 0
                    : Summary::getCounterMask(Summary::ConsumerOpen) |
                          Summary::getCounterMask(Summary::Published) |
                          Summary::getCounterMask(Summary::Waited);
    case DFBProtocolEffectKind::Pop:
      return Summary::getCounterMask(Summary::ConsumerOpen) |
             Summary::getCounterMask(Summary::Capacity) |
             Summary::getCounterMask(Summary::Popped);
    }
    llvm_unreachable("unknown protocol effect kind");
  }

  /// The DFB counters that the effects nested in `op` change become unknown;
  /// a counter no nested effect changes stays exact, so exact wait totals
  /// survive conditional pops.
  TransactionSequenceResult unknown(Operation *op) const {
    if (nestedEveryDFBDiscard.contains(op)) {
      return unknown(allDFBIds);
    }
    TransactionSequenceResult result;
    auto countersIt = nestedCounters.find(op);
    if (countersIt != nestedCounters.end()) {
      for (const auto &[logicalId, counters] : countersIt->second) {
        result.markUnknown(logicalId, counters);
      }
    }
    return result;
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
        appendProtocolOp(&op, result);
      } else if (discardTargetIds.contains(&op)) {
        result.append(summarizeStateDiscard(&op), conditionFormulas);
      } else if (nestedDFBIds.contains(&op)) {
        result.append(summarizeRegionOp(&op, executions), conditionFormulas);
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

  /// A reset ends the current segment of each DFB it restores and labels the
  /// DFB's sequences with itself.
  TransactionSequenceResult summarizeStateDiscard(Operation *op) {
    TransactionSequenceResult result;
    DFBIdRange logicalIds;
    if (discardsEveryDFB.contains(op)) {
      logicalIds.insert(allDFBIds.begin(), allDFBIds.end());
    } else {
      const SmallVector<int64_t> &targets = discardTargetIds.lookup(op);
      logicalIds.insert(targets.begin(), targets.end());
    }
    // Two segments: the empty tail of the interval before the reset, which
    // `append` joins to the preceding summary, and the empty start of the
    // interval after it.
    ExecutedStateDiscard discard = discards.lookup(op);
    for (int64_t logicalId : logicalIds) {
      result.startSegment(logicalId);
      result.startSegment(logicalId);
      result.recordDiscard(logicalId, discard);
    }
    return result;
  }

  /// The counter changes of one effect; `counted` is false for the
  /// acquisition an opaque release implies, which is no user wait.
  static TransactionSequenceSummary makeEvent(DFBProtocolEffectKind kind,
                                              std::int64_t blocks,
                                              bool counted = true) {
    TransactionSequenceSummary event;
    switch (kind) {
    case DFBProtocolEffectKind::Reserve:
      event.setEvent(TransactionSequenceSummary::ProducerOpen, blocks);
      event.setEvent(TransactionSequenceSummary::Capacity, blocks);
      break;
    case DFBProtocolEffectKind::Push:
      event.setEvent(TransactionSequenceSummary::ProducerOpen, -blocks);
      event.setEvent(TransactionSequenceSummary::Published, blocks);
      event.setEvent(TransactionSequenceSummary::Pushed, blocks);
      break;
    case DFBProtocolEffectKind::Wait:
      event.setEvent(TransactionSequenceSummary::ConsumerOpen, blocks);
      event.setEvent(TransactionSequenceSummary::Published, -blocks);
      if (counted) {
        event.setEvent(TransactionSequenceSummary::Waited, blocks);
      }
      break;
    case DFBProtocolEffectKind::Pop:
      event.setEvent(TransactionSequenceSummary::ConsumerOpen, -blocks);
      event.setEvent(TransactionSequenceSummary::Capacity, -blocks);
      event.setEvent(TransactionSequenceSummary::Popped, blocks);
      break;
    }
    return event;
  }

  /// Open acquisitions of `counter` on `logicalId` in the block summarized
  /// so far, from `sofar` (the block before this operation) and `partial`
  /// (this operation's earlier effects).
  /// Append the effects of one protocol operation to `result`. Opaque-call
  /// summaries state Metal-level actions: a declared reserve or wait is a
  /// readiness threshold rather than an acquisition that a later release
  /// closes, so only the declared pushes and pops count. On each alternative
  /// of the block so far, an opaque push or pop closes a user acquisition of
  /// its kind that is still open there, as automatic synchronization pairs
  /// them; otherwise it is a transfer that acquires and releases its blocks.
  void appendProtocolOp(Operation *op, TransactionSequenceResult &result) {
    bool opaque = isa<OpaqueCallOp>(op);
    for (const DFBProtocolEffect &effect :
         cast<DFBAccessOpInterface>(op).getDFBProtocolEffects()) {
      FailureOr<int64_t> logicalId = getDFBId(effect.dfb);
      assert(succeeded(logicalId) && "DFB identities were verified");
      std::optional<int64_t> blocks = getDFBProtocolEffectBlockCount(effect);
      if (!blocks) {
        result.abandon(*logicalId);
        continue;
      }
      if (result.isAbandoned(*logicalId)) {
        continue;
      }
      if (!opaque) {
        TransactionSequenceResult single;
        single.getOrCreate(*logicalId).front().segments =
            TransactionSegments{makeEvent(effect.kind, *blocks)};
        result.append(single, conditionFormulas);
        continue;
      }
      if (effect.kind != DFBProtocolEffectKind::Push &&
          effect.kind != DFBProtocolEffectKind::Pop) {
        continue;
      }
      bool push = effect.kind == DFBProtocolEffectKind::Push;
      TransactionSequenceSummary::Counter openCounter =
          push ? TransactionSequenceSummary::ProducerOpen
               : TransactionSequenceSummary::ConsumerOpen;
      bool overflow = false;
      for (TransactionAlternative &alternative :
           result.getOrCreate(*logicalId)) {
        TransactionSequenceSummary &last = alternative.segments.back();
        // The release closes the open user acquisition first; only the
        // remainder is a self-contained transfer.
        std::int64_t open = std::max<std::int64_t>(0, last.net[openCounter]);
        if (open < *blocks &&
            !last.append(makeEvent(push ? DFBProtocolEffectKind::Reserve
                                        : DFBProtocolEffectKind::Wait,
                                   *blocks - open, /*counted=*/false))) {
          overflow = true;
        }
        if (!last.append(makeEvent(effect.kind, *blocks))) {
          overflow = true;
        }
      }
      if (overflow) {
        result.abandon(*logicalId);
      }
    }
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
      // every nonnegative trip count; a reset inside the body does not.
      TransactionSequenceResult oneIteration =
          summarizeRegion(body, std::nullopt);
      for (const auto &[logicalId, own] : oneIteration.alternatives) {
        if (own.size() != 1 || own.front().segments.size() != 1) {
          oneIteration.markUnknown(logicalId);
          continue;
        }
        const TransactionSegments &segments = own.front().segments;
        for (std::size_t counter = 0;
             counter < TransactionSequenceSummary::Count; ++counter) {
          if (segments.front().net[counter] != 0) {
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

  /// Whether every execution of `op` enters one of its regions.
  static bool coversEveryPath(Operation *op) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return !ifOp.getElseRegion().empty();
    }
    return isa<scf::IndexSwitchOp, scf::ExecuteRegionOp>(op);
  }

  /// The decision an `scf.if` makes by entering `region`, or by entering no
  /// region when `region` is null. Other operations decide nothing.
  std::optional<ConditionDecision> getDecision(Operation *op, Region *region) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp) {
      return std::nullopt;
    }
    return ConditionDecision{conditionFormulas.get(ifOp.getCondition()),
                             region == &ifOp.getThenRegion()};
  }

  /// Restricts `alternative` to executions taking `decision`. Returns false
  /// when none of its executions does, so the path is dead.
  bool decide(TransactionAlternative &alternative,
              std::optional<ConditionDecision> decision) {
    if (!decision) {
      return true;
    }
    DispatchConditionFormulas::FormulaId literal =
        decision->value ? decision->formula
                        : conditionFormulas.negate(decision->formula);
    DispatchConditionFormulas::FormulaId pathCondition =
        conditionFormulas.conjoin(alternative.pathCondition, literal);
    if (!conditionFormulas.satisfiable(pathCondition)) {
      return false;
    }
    alternative.pathCondition = pathCondition;
    return true;
  }

  /// An at-most-once operation whose region activity is unproven and that
  /// resets a DFB: the DFB's sequence is one of the regions' sequences or,
  /// when no region covers the remaining executions, the empty sequence. The
  /// regions' sequences already carry the discards they execute; each gains
  /// the decision that selects it, and a sequence whose decisions exclude it
  /// is dead. DFBs the operation does not reset stay unknown.
  TransactionSequenceResult summarizeUnprovenResetOp(Operation *op) {
    TransactionSequenceResult result = unknown(op);
    DFBIdRange resetIds;
    if (nestedEveryDFBDiscard.contains(op)) {
      resetIds.insert(allDFBIds.begin(), allDFBIds.end());
    } else if (auto resetIt = nestedDiscardIds.find(op);
               resetIt != nestedDiscardIds.end()) {
      resetIds.insert(resetIt->second.begin(), resetIt->second.end());
    }
    if (resetIds.empty()) {
      return result;
    }
    // The empty sequence of a region (or of skipping every region); absent
    // when the decision selecting it is dead.
    auto emptyAlternative =
        [&](Region *region) -> std::optional<TransactionAlternative> {
      TransactionAlternative alternative{
          TransactionSegments{TransactionSequenceSummary()},
          {},
          DispatchConditionFormulas::kTrue};
      if (!decide(alternative, getDecision(op, region))) {
        return std::nullopt;
      }
      return alternative;
    };
    SmallVector<std::pair<TransactionSequenceResult, Region *>> branches;
    for (Region &region : op->getRegions()) {
      if (!region.empty()) {
        branches.push_back({summarizeRegion(region, std::nullopt), &region});
      }
    }
    for (int64_t logicalId : resetIds) {
      TransactionAlternatives own;
      bool unknownBranch = false;
      for (auto &[branch, region] : branches) {
        if (branch.isUnknown(logicalId,
                             TransactionSequenceSummary::getAllCounterMask())) {
          unknownBranch = true;
          break;
        }
        auto branchIt = branch.alternatives.find(logicalId);
        if (branchIt == branch.alternatives.end()) {
          if (std::optional<TransactionAlternative> empty =
                  emptyAlternative(region)) {
            unknownBranch = !TransactionSequenceResult::insertAlternative(
                own, std::move(*empty), conditionFormulas);
          }
        } else {
          for (const TransactionAlternative &alternative : branchIt->second) {
            TransactionAlternative decided = alternative;
            if (!decide(decided, getDecision(op, region))) {
              continue;
            }
            if (!TransactionSequenceResult::insertAlternative(
                    own, std::move(decided), conditionFormulas)) {
              unknownBranch = true;
              break;
            }
          }
        }
        if (unknownBranch) {
          break;
        }
      }
      if (!unknownBranch && !coversEveryPath(op)) {
        if (std::optional<TransactionAlternative> empty =
                emptyAlternative(nullptr)) {
          unknownBranch = !TransactionSequenceResult::insertAlternative(
              own, std::move(*empty), conditionFormulas);
        }
      }
      if (unknownBranch) {
        continue;
      }
      result.unknownCounters.erase(logicalId);
      result.alternatives[logicalId] = std::move(own);
    }
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
        return executesRegionsAtMostOnce(op) ? summarizeUnprovenResetOp(op)
                                             : unknown(op);
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
  llvm::DenseMap<Operation *, SmallVector<int64_t>> discardTargetIds;
  llvm::DenseMap<Operation *, ExecutedStateDiscard> discards;
  llvm::DenseSet<Operation *> discardsEveryDFB;
  llvm::DenseSet<Operation *> nestedEveryDFBDiscard;
  llvm::DenseMap<Operation *, SmallVector<int64_t>> nestedDFBIds;
  llvm::DenseMap<
      Operation *,
      llvm::DenseMap<int64_t, TransactionSequenceSummary::CounterMask>>
      nestedCounters;
  llvm::DenseSet<int64_t> opaquePopDFBIds;
  llvm::DenseMap<Operation *, SmallVector<int64_t>> nestedDiscardIds;
  llvm::DenseSet<Region *> regionsWithProtocolOps;
  DFBIdRange allDFBIds;
  DispatchConditionFormulas &conditionFormulas;
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
              .emplace(coord, DFBTransactionSequenceAnalysis(
                                  thread, coord, state, conditionFormulas)
                                  .run())
              .first;
    }
    return resultIt->second;
  }

  DispatchConditionFormulas &getConditionFormulas() {
    return conditionFormulas;
  }

private:
  const DFBProtocolDomainState &state;
  DispatchConditionFormulas conditionFormulas;
  llvm::DenseMap<Operation *,
                 std::map<LaunchNodeCoord, TransactionSequenceResult>>
      results;
};

/// The alternative sequences of `logicalId` in `sequence`, or absent when a
/// required counter is unknown. A DFB without transactions has one empty
/// segment.
std::optional<TransactionAlternatives> getTransactionAlternatives(
    const TransactionSequenceResult &sequence, int64_t logicalId,
    TransactionSequenceSummary::CounterMask requiredCounters) {
  if (sequence.isUnknown(logicalId, requiredCounters)) {
    return std::nullopt;
  }
  auto alternativesIt = sequence.alternatives.find(logicalId);
  if (alternativesIt == sequence.alternatives.end()) {
    return TransactionAlternatives{
        {TransactionSegments{TransactionSequenceSummary()},
         {},
         DispatchConditionFormulas::kTrue}};
  }
  return alternativesIt->second;
}

SmallVector<func::FuncOp>
getTransactionThreadsAtNode(ArrayRef<DFBTransaction> transactions,
                            LaunchNodeCoord coord) {
  SmallVector<func::FuncOp> threads;
  for (const DFBTransaction &transaction : transactions) {
    if (!transactionMayExecuteAt(transaction, coord)) {
      continue;
    }
    func::FuncOp thread = getEnclosingKernelThread(transaction.op);
    if (!llvm::is_contained(threads, thread)) {
      threads.push_back(thread);
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
  diagnostic.attachNote(bindSite->getLoc()) << "dataflow buffer declared here";
}

/// A data-movement kernel runs its producer and consumer effects on one RISC
/// in program order; a compute kernel splits them across PACK and UNPACK.
bool isSequentialKernel(func::FuncOp thread) {
  return getKernelThreadType(thread) != ttkernel::ThreadType::Compute;
}

/// Verify one role of one kernel at one launch node: within every reset
/// interval the releases may not exceed the acquisitions and the open
/// acquisitions may not exceed capacity, and the last interval must close.
/// Returns true when a violation was reported.
bool verifyEndpointTransactionSequence(
    int64_t logicalId, ArrayRef<DFBTransaction> transactions,
    func::FuncOp thread, LaunchNodeCoord coord, std::uint64_t capacityBlocks,
    TransactionSequenceSummary::Counter openCounter, llvm::StringRef role,
    llvm::StringRef acquireName, llvm::StringRef releaseName,
    Operation *bindSite, DFBTransactionSequenceCache &sequenceCache) {
  const TransactionSequenceResult &sequence = sequenceCache.get(thread, coord);
  std::optional<TransactionAlternatives> maybeAlternatives =
      getTransactionAlternatives(
          sequence, logicalId,
          TransactionSequenceSummary::getCounterMask(openCounter));
  if (!maybeAlternatives) {
    return false;
  }
  for (const TransactionAlternative &alternative : *maybeAlternatives) {
    const TransactionSegments &segments = alternative.segments;
    for (const auto &[index, summary] : llvm::enumerate(segments)) {
      bool ordered = summary.minimum[openCounter] >= 0;
      bool closed =
          index + 1 < segments.size() || summary.net[openCounter] == 0;
      bool capacitySafe = static_cast<std::uint64_t>(
                              summary.maximum[openCounter]) <= capacityBlocks;
      if (ordered && closed && capacitySafe) {
        continue;
      }

      Operation *transactionOp = getTransactionAtNode(
          transactions, ArrayRef<DFBTransaction>(), coord, thread);
      InFlightDiagnostic diagnostic = transactionOp->emitError();
      if (!ordered || !closed) {
        diagnostic << "logical DFB " << logicalId << " has an incomplete or "
                   << "misordered " << role
                   << " lifecycle on core_x=" << coord.x
                   << ", core_y=" << coord.y;
        if (!ordered) {
          diagnostic.attachNote()
              << "the " << role << " kernel performs a " << releaseName
              << " before a matching " << acquireName;
        } else {
          std::int64_t open = summary.net[openCounter];
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
            << summary.maximum[openCounter] << " open " << acquireName
            << " block(s), exceeding capacity " << capacityBlocks;
      }
      diagnostic.attachNote()
          << "keep each " << acquireName << " followed by its matching "
          << releaseName << " before opening more than DFB capacity";
      attachDeclarationNote(diagnostic, bindSite);
      return true;
    }
  }
  return false;
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
  std::optional<TransactionAlternatives> maybeAlternatives =
      getTransactionAlternatives(
          sequence, logicalId,
          TransactionSequenceSummary::getCounterMask(
              TransactionSequenceSummary::ProducerOpen) |
              TransactionSequenceSummary::getCounterMask(
                  TransactionSequenceSummary::ConsumerOpen) |
              TransactionSequenceSummary::getCounterMask(
                  TransactionSequenceSummary::Capacity) |
              TransactionSequenceSummary::getCounterMask(
                  TransactionSequenceSummary::Published));
  if (!maybeAlternatives) {
    return false;
  }
  for (const TransactionAlternative &alternative : *maybeAlternatives) {
    const TransactionSegments &segments = alternative.segments;
    for (const auto &[index, summary] : llvm::enumerate(segments)) {
      bool prefixesValid = llvm::all_of(
          summary.minimum, [](std::int64_t value) { return value >= 0; });
      bool lifecyclesClosed =
          index + 1 < segments.size() ||
          (summary.net[TransactionSequenceSummary::ProducerOpen] == 0 &&
           summary.net[TransactionSequenceSummary::ConsumerOpen] == 0);
      bool capacitySafe =
          static_cast<std::uint64_t>(
              summary.maximum[TransactionSequenceSummary::Capacity]) <=
          capacityBlocks;
      if (prefixesValid && lifecyclesClosed && capacitySafe) {
        continue;
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
        diagnostic.attachNote()
            << "the single kernel waits for " << deficit
            << " block(s) before they are visibly published";
      } else if (incomplete) {
        diagnostic.attachNote()
            << "the single kernel has an incomplete or "
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
  }
  return false;
}

/// Blocks a role transfers per launch on one node: released blocks when
/// every release count is exact, otherwise the acquired blocks for a consumer
/// (a wait beyond the pushes never completes). Absent when unproven.
struct RoleTotal {
  std::uint64_t blocks = 0;
  bool acquisitions = false;
};

std::optional<RoleTotal> getRoleTotal(ArrayRef<DFBTransaction> transactions,
                                      LaunchNodeCoord coord,
                                      const DFBProtocolDomainState &state,
                                      bool allowAcquisitions) {
  if (std::optional<std::uint64_t> released =
          getExactTransactionCount(transactions, coord, state)) {
    return RoleTotal{*released, false};
  }
  // Waits stand in for pops only where every pop is a user pop; an opaque
  // pop has no user wait.
  if (!allowAcquisitions ||
      llvm::any_of(transactions, [&](const DFBTransaction &transaction) {
        return !transaction.acquisition && isa<OpaqueCallOp>(transaction.op) &&
               transactionMayExecuteAt(transaction, coord);
      })) {
    return std::nullopt;
  }
  if (std::optional<std::uint64_t> acquired = getExactTransactionCount(
          transactions, coord, state, /*thread=*/{}, /*acquisitions=*/true)) {
    return RoleTotal{*acquired, true};
  }
  return std::nullopt;
}

void reportCountsMismatch(int64_t logicalId, Operation *transactionOp,
                          LaunchNodeCoord coord, RoleTotal producer,
                          RoleTotal consumer, std::uint64_t capacityBlocks,
                          std::optional<std::size_t> interval,
                          Operation *bindSite) {
  InFlightDiagnostic diagnostic = transactionOp->emitError()
                                  << "logical DFB " << logicalId
                                  << " has capacity-unsafe producer and "
                                     "consumer transactions on core_x="
                                  << coord.x << ", core_y=" << coord.y;
  llvm::StringRef consumerVerb = consumer.acquisitions ? "waits for" : "pops";
  if (consumer.blocks > producer.blocks) {
    diagnostic.attachNote()
        << "the consumer " << consumerVerb << " " << consumer.blocks
        << " block(s) per launch, but the producer pushes " << producer.blocks
        << " block(s) per launch";
  } else {
    diagnostic.attachNote()
        << "the producer pushes " << producer.blocks
        << " block(s) per launch and the consumer " << consumerVerb << " "
        << consumer.blocks << ", leaving " << producer.blocks - consumer.blocks
        << " outstanding block(s) for capacity " << capacityBlocks;
  }
  if (interval && *interval == 0) {
    diagnostic.attachNote()
        << "in the interval before the first synchronized reset";
  } else if (interval) {
    diagnostic.attachNote()
        << "in the interval after synchronized reset " << *interval;
  }
  diagnostic.attachNote()
      << "keep pops within pushes and unpopped blocks within DFB capacity "
         "on every active node";
  attachDeclarationNote(diagnostic, bindSite);
}

/// Compare the pushed and popped totals of the kernels at one node, per
/// interval between synchronized resets when the DFB is reset anywhere and
/// otherwise over the launch. Returns true when a violation was reported.
bool verifyCrossKernelTransactionCounts(
    int64_t logicalId, ArrayRef<DFBTransaction> producers,
    ArrayRef<DFBTransaction> consumers, ArrayRef<func::FuncOp> threads,
    LaunchNodeCoord coord, std::uint64_t capacityBlocks, bool segmented,
    Operation *bindSite, const DFBProtocolDomainState &state,
    DFBTransactionSequenceCache &sequenceCache) {
  if (!segmented) {
    std::optional<RoleTotal> producer =
        getRoleTotal(producers, coord, state, /*allowAcquisitions=*/false);
    std::optional<RoleTotal> consumer =
        getRoleTotal(consumers, coord, state, /*allowAcquisitions=*/true);
    if (!producer || !consumer ||
        countsFitCapacity(producer->blocks, consumer->blocks, capacityBlocks)) {
      return false;
    }
    reportCountsMismatch(
        logicalId, getTransactionAtNode(producers, consumers, coord), coord,
        *producer, *consumer, capacityBlocks, std::nullopt, bindSite);
    return true;
  }

  // Every participant executes the same discards and decides the same
  // dispatch conditions, so alternatives of different kernels belong to one
  // execution when their discards are equal and their decisions are jointly
  // satisfiable. A kernel with one alternative matches every choice. Every
  // consistent combination is checked; more of them than `kMaxAlternatives`
  // leaves the DFB unchecked at this node.
  // A kernel whose pops are conditional contributes its exact wait total
  // instead, as in the unsegmented comparison.
  using Summary = TransactionSequenceSummary;
  constexpr Summary::CounterMask pushedMask =
      Summary::getCounterMask(Summary::Pushed);
  SmallVector<TransactionAlternatives, 4> perThread;
  SmallVector<bool, 4> waitsStandIn;
  for (func::FuncOp thread : threads) {
    const TransactionSequenceResult &sequence =
        sequenceCache.get(thread, coord);
    bool waits = false;
    std::optional<TransactionAlternatives> alternatives =
        getTransactionAlternatives(
            sequence, logicalId,
            pushedMask | Summary::getCounterMask(Summary::Popped));
    if (!alternatives) {
      alternatives = getTransactionAlternatives(
          sequence, logicalId,
          pushedMask | Summary::getCounterMask(Summary::Waited));
      waits = true;
    }
    if (!alternatives) {
      return false;
    }
    perThread.push_back(std::move(*alternatives));
    waitsStandIn.push_back(waits);
  }

  DispatchConditionFormulas &formulas = sequenceCache.getConditionFormulas();
  SmallVector<const TransactionAlternative *> chosen;
  SmallVector<DispatchConditionFormulas::FormulaId> pathConditions{
      DispatchConditionFormulas::kTrue};
  auto isConsistent = [&](const TransactionAlternative &candidate,
                          bool candidateAlone) {
    for (auto [index, earlier] : llvm::enumerate(chosen)) {
      bool earlierAlone = perThread[index].size() == 1;
      if (!earlierAlone && !candidateAlone &&
          earlier->executedDiscards != candidate.executedDiscards) {
        return false;
      }
    }
    return formulas.satisfiable(
        formulas.conjoin(pathConditions.back(), candidate.pathCondition));
  };
  // Every participant executes the same resets; a kernel that touches the
  // DFB without resetting it keeps one segment and is left out. Returns true
  // when a violation was reported.
  auto checkCombination = [&]() {
    std::size_t intervals = 0;
    for (const TransactionAlternative *alternative : chosen) {
      intervals = std::max(intervals, alternative->segments.size());
    }
    bool aligned =
        llvm::all_of(chosen, [&](const TransactionAlternative *alternative) {
          return alternative->segments.size() == 1 ||
                 alternative->segments.size() == intervals;
        });
    if (!aligned) {
      return false;
    }
    for (std::size_t interval = 0; interval < intervals; ++interval) {
      std::optional<std::uint64_t> pushed = 0;
      std::optional<std::uint64_t> popped = 0;
      bool anyWaits = false;
      for (auto [index, alternative] : llvm::enumerate(chosen)) {
        const TransactionSegments &segments = alternative->segments;
        if (segments.size() != intervals || !pushed || !popped) {
          continue;
        }
        pushed = llvm::checkedAddUnsigned(
            *pushed, static_cast<std::uint64_t>(
                         segments[interval].net[Summary::Pushed]));
        Summary::Counter consumed =
            waitsStandIn[index] ? Summary::Waited : Summary::Popped;
        anyWaits |=
            waitsStandIn[index] && segments[interval].net[Summary::Waited] != 0;
        popped = llvm::checkedAddUnsigned(
            *popped,
            static_cast<std::uint64_t>(segments[interval].net[consumed]));
      }
      // A total beyond the representable range is not compared.
      if (!pushed || !popped) {
        continue;
      }
      if (countsFitCapacity(*pushed, *popped, capacityBlocks)) {
        continue;
      }
      reportCountsMismatch(
          logicalId, getTransactionAtNode(producers, consumers, coord), coord,
          RoleTotal{*pushed, false}, RoleTotal{*popped, anyWaits},
          capacityBlocks, interval, bindSite);
      return true;
    }
    return false;
  };
  std::size_t combinations = 0;
  bool overflow = false;
  bool violation = false;
  std::function<void(std::size_t)> visit = [&](std::size_t index) {
    if (overflow || violation) {
      return;
    }
    if (index == perThread.size()) {
      if (++combinations > TransactionSequenceResult::kMaxAlternatives) {
        overflow = true;
        return;
      }
      violation = checkCombination();
      return;
    }
    bool alone = perThread[index].size() == 1;
    for (const TransactionAlternative &candidate : perThread[index]) {
      if (!isConsistent(candidate, alone)) {
        continue;
      }
      chosen.push_back(&candidate);
      pathConditions.push_back(
          formulas.conjoin(pathConditions.back(), candidate.pathCondition));
      visit(index + 1);
      pathConditions.pop_back();
      chosen.pop_back();
    }
  };
  visit(0);
  return violation;
}

/// Verify one logical DFB on every launch node where a transaction may
/// execute. Nodes where an opaque call may perform protocol actions the IR
/// does not represent are skipped. Returns true when a violation was reported.
bool verifyDFBTransactions(int64_t logicalId,
                           ArrayRef<DFBTransaction> producers,
                           ArrayRef<DFBTransaction> consumers,
                           std::uint64_t capacityBlocks,
                           const LaunchNodeDomain &externalProtocolDomain,
                           bool resetAnywhere, Operation *bindSite,
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
    if (verifyCrossKernelTransactionCounts(
            logicalId, producers, consumers, threads, coord, capacityBlocks,
            resetAnywhere, bindSite, state, sequenceCache)) {
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

    bool everyDFBReset = false;
    llvm::DenseSet<int64_t> resetDFBIds;
    module.walk([&](Operation *op) {
      if (std::optional<StateDiscardTargets> targets =
              getStateDiscardTargets(op)) {
        everyDFBReset |= targets->all;
        resetDFBIds.insert(targets->logicalIds.begin(),
                           targets->logicalIds.end());
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
          everyDFBReset || resetDFBIds.contains(logicalId), bindSite, state,
          sequenceCache);
    }
    if (sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
