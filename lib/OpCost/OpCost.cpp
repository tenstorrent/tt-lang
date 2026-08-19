// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/OpCost/OpCost.h"

#include <cmath>
#include <iterator>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::opcost {
namespace {

/// One measured cost from an LLK perf sweep.
///
/// The key is everything the measurement depends on, because a cost is only
/// meaningful for the configuration it was taken in. A lookup that cannot match
/// every field it needs has to miss rather than borrow a neighbouring
/// configuration.
///
/// Carries neither the operation nor the engine: a row is reached only through
/// the engine slot that slices it, so both are implied by where it sits.
///
/// `variant` holds the benchmark-specific knobs as a `k=v;k=v` string, in the
/// order gen_cost_table.py declared them, so a benchmark that sweeps something
/// unusual does not silently average unlike measurements together.
struct MeasuredCost {
  llvm::StringRef inFormat;
  llvm::StringRef outFormat;
  bool destAcc;
  unsigned faces;
  llvm::StringRef variant;
  Unit unit;

  /// Kept as measured; rounding is the caller's decision. `fixed` is the
  /// intercept of a fit against a block dimension, and zero for a plain
  /// measurement.
  double cost;
  double fixed;
};

/// Which measured rows describe one operation on one engine.
///
/// A slice of `kMeasured`, and nothing more: the table carries measurements
/// alone, so a slot has no number of its own to fall back on.
///
/// `count == 0` means the operation occupies the engine and nothing has timed
/// it, which is distinct from a measurement existing that a given kernel cannot
/// match: the first is missing data, the second a configuration no sweep
/// presented.
struct EngineCost {
  unsigned first = 0;
  unsigned count = 0;
};

/// One operation's entry: the engines it occupies, and what it costs on each.
///
/// An engine with no value is one the operation does not run on. Every engine
/// empty is an operation that runs nowhere -- known and free, as opposed to
/// absent from the table, which is unknown.
struct OpCost {
  llvm::StringRef op;
  std::optional<EngineCost> dm;
  std::optional<EngineCost> unpack;
  std::optional<EngineCost> math;
  std::optional<EngineCost> pack;
};

#include "CostTableBlackhole.inc"

/// The measured rows belonging to one engine slot; empty when it has none.
llvm::ArrayRef<MeasuredCost> measuredRows(const EngineCost &slot) {
  return llvm::ArrayRef(kMeasured).slice(slot.first, slot.count);
}

/// The slot an operation uses for `engine`, or nullopt when it does not run
/// there.
const std::optional<EngineCost> &engineSlot(const OpCost &entry,
                                            Engine engine) {
  switch (engine) {
  case Engine::Dm:
    return entry.dm;
  case Engine::Unpack:
    return entry.unpack;
  case Engine::Math:
    return entry.math;
  case Engine::Pack:
    return entry.pack;
  }
  llvm_unreachable("unhandled engine");
}

/// The cost table for one architecture, keyed by operation name.
///
/// Built once instead of binary-searched, because a lookup happens per
/// placement and a kernel whose loops unroll places hundreds of thousands of
/// them. The entries point at static data; nothing is copied.
///
/// Wormhole has no table yet, and answers empty rather than borrowing
/// Blackhole's -- costs do not transfer between architectures.
const llvm::StringMap<const OpCost *> &getTable(Arch arch) {
  static const llvm::StringMap<const OpCost *> blackhole = [] {
    llvm::StringMap<const OpCost *> t;
    for (const OpCost &entry : kCostTable) {
      t[entry.op] = &entry;
    }
    return t;
  }();
  static const llvm::StringMap<const OpCost *> empty;
  return arch == Arch::Blackhole ? blackhole : empty;
}

const OpCost *findOp(llvm::StringRef op, Arch arch) {
  const llvm::StringMap<const OpCost *> &table = getTable(arch);
  auto it = table.find(op);
  return it == table.end() ? nullptr : it->second;
}

/// Whether the caller can answer for every knob an entry was measured under.
///
/// Purely a string match against `OpKey::knobs`; nothing here knows what any
/// knob means. A row naming one the caller did not supply cannot be matched,
/// which is what stops a measurement taken in a configuration we cannot
/// describe from answering anyway.
bool variantMatches(llvm::StringRef variant, llvm::ArrayRef<Knob> knobs) {
  while (!variant.empty()) {
    auto [entry, rest] = variant.split(';');
    variant = rest;
    auto [knob, value] = entry.split('=');
    const Knob *supplied = nullptr;
    for (const Knob &k : knobs) {
      if (k.name == knob) {
        supplied = &k;
        break;
      }
    }
    if (!supplied || supplied->value != value) {
      return false;
    }
  }
  return true;
}

/// Every field has to agree, `variant` included.
///
/// An empty field on the row matches any value: it is how a benchmark that
/// never varied something stays reachable, rather than demanding a value it
/// never measured.
bool keyMatches(const MeasuredCost &row, const OpKey &opKey,
                const KernelConfig &config) {
  return row.inFormat == opKey.inFormat && row.outFormat == config.outFormat &&
         row.destAcc == config.destAcc && row.faces == config.faces &&
         variantMatches(row.variant, opKey.knobs);
}

/// The one measured row matching this configuration, or nothing.
///
/// Rows agreeing to within a hair collapse to one answer; rows that disagree
/// are treated as no match, since two rows matching one key means the key is
/// missing a field the measurement depended on.
std::optional<Cost> matchRow(const EngineCost &slot, const OpKey &opKey,
                             const KernelConfig &config) {
  std::optional<Cost> found;
  for (const MeasuredCost &row : measuredRows(slot)) {
    if (!keyMatches(row, opKey, config)) {
      continue;
    }
    if (found && std::abs(found->value - row.cost) > 0.01 * found->value) {
      return std::nullopt;
    }
    found = Cost{row.cost, row.fixed, row.unit};
  }
  return found;
}

} // namespace

bool isKnownOp(llvm::StringRef op, Arch arch) {
  return findOp(op, arch) != nullptr;
}

bool runsOnEngine(llvm::StringRef op, Engine engine, Arch arch) {
  const OpCost *entry = findOp(op, arch);
  return entry && engineSlot(*entry, engine).has_value();
}

bool runsNowhere(llvm::StringRef op, Arch arch) {
  const OpCost *entry = findOp(op, arch);
  return entry && !entry->dm && !entry->unpack && !entry->math && !entry->pack;
}

std::optional<Cost> lookup(llvm::StringRef op, Engine engine,
                           const OpKey &opKey, const KernelConfig &config,
                           Arch arch) {
  const OpCost *entry = findOp(op, arch);
  if (!entry) {
    return std::nullopt;
  }
  const std::optional<EngineCost> &slot = engineSlot(*entry, engine);
  if (!slot) {
    return std::nullopt;
  }
  return matchRow(*slot, opKey, config);
}

TableStats getTableStats(Arch arch) {
  if (arch != Arch::Blackhole) {
    return {};
  }
  return {static_cast<unsigned>(std::size(kCostTable)),
          static_cast<unsigned>(std::size(kMeasured))};
}

llvm::ArrayRef<llvm::StringRef> getOperations(Arch arch) {
  static const std::vector<llvm::StringRef> blackhole = [] {
    std::vector<llvm::StringRef> names;
    names.reserve(std::size(kCostTable));
    for (const OpCost &entry : kCostTable) {
      names.push_back(entry.op);
    }
    return names;
  }();
  static const std::vector<llvm::StringRef> empty;
  return arch == Arch::Blackhole ? blackhole : empty;
}

unsigned getMeasurementCount(llvm::StringRef op, Engine engine, Arch arch) {
  const OpCost *entry = findOp(op, arch);
  if (!entry) {
    return 0;
  }
  const std::optional<EngineCost> &slot = engineSlot(*entry, engine);
  return slot ? slot->count : 0;
}

llvm::StringRef getEngineName(Engine engine) {
  switch (engine) {
  case Engine::Dm:
    return "dm";
  case Engine::Unpack:
    return "unpack";
  case Engine::Math:
    return "math";
  case Engine::Pack:
    return "pack";
  }
  llvm_unreachable("unhandled engine");
}

} // namespace mlir::tt::opcost
