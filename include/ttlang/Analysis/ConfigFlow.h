// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Configuration Flow
//===----------------------------------------------------------------------===//
//
// A forward analysis of one implicit hardware configuration slot over
// structured control flow. Each operation either preserves the slot or
// replaces it with a constant state, so region transfers stay in a closed
// three-element algebra and every region is summarized once. The analysis is
// linear in the number of operations at any loop depth.
//
// See docs/development/MathHardwareConfiguration.md for the correctness
// argument, insertion interaction, conservative behavior, and limitations.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_ANALYSIS_CONFIGFLOW_H
#define TTLANG_ANALYSIS_CONFIGFLOW_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <utility>

namespace mlir::tt {

/// Transfer function of a region for slot `Slot`: the identity, `s -> c`, or
/// `s -> join(s, c)`. The set is closed under sequencing and join, and the
/// loop fixed point `x = join(s, f(x))` is `join(s, f(s))`.
/// `Slot::join` must be associative, commutative, and idempotent.
template <typename Slot>
struct ConfigTransfer {
  using State = typename Slot::State;
  enum class Kind { Identity, Assign, Join };

  Kind kind = Kind::Identity;
  State value{};

  static ConfigTransfer assign(State state) {
    return {Kind::Assign, std::move(state)};
  }

  /// Transfer from the entry of a loop to the entry of a body with transfer
  /// `body`.
  static ConfigTransfer loopEntry(const ConfigTransfer &body, Slot &slot) {
    return ConfigTransfer().join(body, slot);
  }

  State apply(const State &state, Slot &slot) const {
    switch (kind) {
    case Kind::Identity:
      return state;
    case Kind::Assign:
      return value;
    case Kind::Join:
      return slot.join(state, value);
    }
    llvm_unreachable("unknown transfer kind");
  }

  /// Returns the transfer that runs this one and then `next`.
  ConfigTransfer then(const ConfigTransfer &next, Slot &slot) const {
    if (next.kind == Kind::Identity) {
      return *this;
    }
    if (next.kind == Kind::Assign || kind == Kind::Identity) {
      return next;
    }
    return {kind, slot.join(value, next.value)};
  }

  /// Returns the transfer of a control-flow merge.
  ConfigTransfer join(const ConfigTransfer &other, Slot &slot) const {
    if (other.kind == Kind::Identity) {
      return kind == Kind::Identity ? *this : ConfigTransfer{Kind::Join, value};
    }
    if (kind == Kind::Identity) {
      return {Kind::Join, other.value};
    }
    Kind joined = kind == Kind::Assign && other.kind == Kind::Assign
                      ? Kind::Assign
                      : Kind::Join;
    return {joined, slot.join(value, other.value)};
  }
};

/// Forward flow of one configuration slot through `scf.for`, `scf.if`, and
/// `scf.while`. A static trip count of 0 contributes the incoming state and
/// does not visit the body. A trip count of 1 runs the body on the incoming
/// state and does not join the backedge. Other operations with regions, and
/// regions with several blocks, analyze each nested block from the unknown
/// state and join its exit with the incoming state and unknown.
///
/// `Slot` provides:
///   - `State`, default-constructible and copyable;
///   - `static State unknown()`;
///   - `State join(const State &, const State &)`;
///   - `std::optional<State> getWrite(Operation *)`, the state after an
///     operation without regions, or std::nullopt when it preserves the slot.
template <typename Slot>
class ConfigFlow {
public:
  using State = typename Slot::State;
  using Transfer = ConfigTransfer<Slot>;
  using Visitor = llvm::function_ref<void(Operation *, const State &)>;

  explicit ConfigFlow(Slot &slot) : slot(slot) {}

  /// Calls `visitor` for every non-terminator operation nested in `region`,
  /// in program order, with the state on entry to that operation. `scf.for`,
  /// `scf.if`, and `scf.while` are visited with the state before the region.
  /// Returns the state on exit from `region`.
  State run(Region &region, const State &entry, Visitor visitor) {
    if (region.empty()) {
      return entry;
    }
    if (!region.hasOneBlock()) {
      State exit = slot.join(entry, Slot::unknown());
      for (Block &block : region) {
        exit = slot.join(exit, runBlock(block, Slot::unknown(), visitor));
      }
      return exit;
    }
    return runBlock(region.front(), entry, visitor);
  }

private:
  static iterator_range<Block::iterator> body(Block &block) {
    if (block.mightHaveTerminator()) {
      return block.without_terminator();
    }
    return {block.begin(), block.end()};
  }

  enum class TripCount { Zero, One, AtLeastTwo, Dynamic };

  static TripCount tripCount(scf::ForOp forOp) {
    std::optional<llvm::APInt> count = forOp.getStaticTripCount();
    if (!count) {
      return TripCount::Dynamic;
    }
    if (count->isZero()) {
      return TripCount::Zero;
    }
    if (count->isOne()) {
      return TripCount::One;
    }
    return TripCount::AtLeastTwo;
  }

  State runBlock(Block &block, State state, Visitor visitor) {
    for (Operation &op : body(block)) {
      state = runOp(&op, state, visitor);
    }
    return state;
  }

  State runOp(Operation *op, const State &state, Visitor visitor) {
    // Region operations are visited with the state before the region so a
    // caller can see the pre-header, which the body fixed point can hide.
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
      visitor(op, state);
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Region &loopBody = forOp.getRegion();
      switch (tripCount(forOp)) {
      case TripCount::Zero:
        return state;
      case TripCount::One:
        return run(loopBody, state, visitor);
      case TripCount::AtLeastTwo:
        return run(
            loopBody,
            Transfer::loopEntry(summarize(loopBody), slot).apply(state, slot),
            visitor);
      case TripCount::Dynamic: {
        State exit = run(
            loopBody,
            Transfer::loopEntry(summarize(loopBody), slot).apply(state, slot),
            visitor);
        return slot.join(state, exit);
      }
      }
      llvm_unreachable("unknown trip count");
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      State thenExit = run(ifOp.getThenRegion(), state, visitor);
      return slot.join(thenExit, run(ifOp.getElseRegion(), state, visitor));
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      Transfer iteration = summarize(whileOp.getBefore())
                               .then(summarize(whileOp.getAfter()), slot);
      // Every exit leaves through the before region.
      State beforeExit =
          run(whileOp.getBefore(),
              Transfer::loopEntry(iteration, slot).apply(state, slot), visitor);
      run(whileOp.getAfter(), beforeExit, visitor);
      return beforeExit;
    }
    if (op->getNumRegions() != 0) {
      State exit = slot.join(state, Slot::unknown());
      for (Region &region : op->getRegions()) {
        exit = slot.join(exit, run(region, Slot::unknown(), visitor));
      }
      return exit;
    }
    visitor(op, state);
    if (std::optional<State> written = slot.getWrite(op)) {
      return *written;
    }
    return state;
  }

  Transfer summarize(Region &region) {
    if (region.empty()) {
      return Transfer();
    }
    auto it = summaries.find(&region);
    if (it != summaries.end()) {
      return it->second;
    }
    Transfer transfer;
    if (region.hasOneBlock()) {
      for (Operation &op : body(region.front())) {
        transfer = transfer.then(summarize(&op), slot);
      }
    } else {
      State exit = Slot::unknown();
      for (Block &block : region) {
        Transfer blockTransfer;
        for (Operation &op : body(block)) {
          blockTransfer = blockTransfer.then(summarize(&op), slot);
        }
        exit = slot.join(exit, blockTransfer.apply(Slot::unknown(), slot));
      }
      transfer = transfer.join(Transfer::assign(exit), slot);
    }
    summaries[&region] = transfer;
    return transfer;
  }

  Transfer summarize(Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      switch (tripCount(forOp)) {
      case TripCount::Zero:
        return Transfer();
      case TripCount::One:
        return summarize(forOp.getRegion());
      case TripCount::AtLeastTwo:
        return Transfer::loopEntry(summarize(forOp.getRegion()), slot)
            .then(summarize(forOp.getRegion()), slot);
      case TripCount::Dynamic: {
        Transfer loopBody = summarize(forOp.getRegion());
        return Transfer().join(
            Transfer::loopEntry(loopBody, slot).then(loopBody, slot), slot);
      }
      }
      llvm_unreachable("unknown trip count");
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return summarize(ifOp.getThenRegion())
          .join(summarize(ifOp.getElseRegion()), slot);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      Transfer before = summarize(whileOp.getBefore());
      return Transfer::loopEntry(
                 before.then(summarize(whileOp.getAfter()), slot), slot)
          .then(before, slot);
    }
    if (op->getNumRegions() != 0) {
      State exit = Slot::unknown();
      for (Region &region : op->getRegions()) {
        exit = slot.join(exit, summarize(region).apply(Slot::unknown(), slot));
      }
      return Transfer().join(Transfer::assign(exit), slot);
    }
    if (std::optional<State> written = slot.getWrite(op)) {
      return Transfer::assign(*written);
    }
    return Transfer();
  }

  Slot &slot;
  llvm::DenseMap<Region *, Transfer> summaries;
};

} // namespace mlir::tt

#endif
