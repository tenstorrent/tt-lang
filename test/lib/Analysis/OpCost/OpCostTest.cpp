// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Checks the operation-cost library against whatever table is compiled into it.
//
// Deliberately asserts no cycle counts. The table is regenerated from device
// sweeps whenever the measurements are re-taken, so a test that pinned its
// numbers would fail on every re-sweep and teach the reader to update it
// without looking -- which is how a real loss of data gets waved through. What
// is pinned here is what the library promises about *any* table: which
// questions it answers, which it refuses, and that every row it holds can be
// found by the key it was measured at.
//
// Each check derives its own probes from the table rather than naming an
// operation or a format, so nothing here needs touching when the data changes.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/OpCost.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <optional>

using namespace mlir::tt;
using opcost::Arch;
using opcost::Cost;
using opcost::Engine;
using opcost::KernelConfig;
using opcost::Knob;
using opcost::Measurement;
using opcost::OpKey;

namespace {

constexpr Engine kEngines[] = {Engine::Dm, Engine::Unpack, Engine::Math,
                               Engine::Pack};

/// Accumulates failures so one run reports every check rather than the first.
class Checks {
public:
  void expect(bool condition, const llvm::Twine &what) {
    if (condition) {
      return;
    }
    ++failures;
    llvm::errs() << "  FAIL " << what << "\n";
  }

  void finish(llvm::StringRef name) {
    if (failures == reported) {
      llvm::outs() << "PASS " << name << "\n";
    } else {
      llvm::outs() << "FAIL " << name << "\n";
      reported = failures;
    }
  }

  bool failed() const { return failures != 0; }

private:
  unsigned failures = 0;
  unsigned reported = 0;
};

/// The knobs of a measurement, split out of its packed `name=value;...` string.
///
/// The values point into the table's own storage, so a list outlives the string
/// it came from only because that string is static.
llvm::SmallVector<Knob> splitKnobs(llvm::StringRef packed) {
  llvm::SmallVector<Knob> knobs;
  while (!packed.empty()) {
    auto [entry, rest] = packed.split(';');
    packed = rest;
    auto [name, value] = entry.split('=');
    knobs.push_back({name, value});
  }
  return knobs;
}

/// The key a measurement was taken at, in the shape a lookup takes it.
std::pair<OpKey, KernelConfig> keyOf(const Measurement &row,
                                     llvm::ArrayRef<Knob> knobs) {
  OpKey opKey;
  opKey.inFormat = row.inFormat;
  opKey.knobs = knobs;

  KernelConfig config;
  config.outFormat = row.outFormat;
  config.destAcc = row.destAcc;
  config.faces = row.faces;
  return {opKey, config};
}

std::string describe(const Measurement &row) {
  return (row.op + "/" + opcost::getEngineName(row.engine) + " " +
          row.inFormat + "->" + row.outFormat +
          " destAcc=" + (row.destAcc ? "1" : "0") +
          " faces=" + std::to_string(row.faces) + " [" + row.knobs + "]")
      .str();
}

bool sameCost(const Cost &a, const Cost &b) {
  // The tolerance the library itself uses to collapse agreeing rows.
  return a.unit == b.unit && std::abs(a.fixed - b.fixed) < 0.005 &&
         std::abs(a.value - b.value) <= 0.01 * std::abs(a.value) + 0.005;
}

/// Every row can be found by the key it was measured at.
///
/// The one check that needs the data, and the one that catches the failure the
/// data can produce on its own: two rows sharing a key and disagreeing are
/// treated as no match, so a sweep that emits a conflicting pair makes a cost
/// disappear from every report rather than announce itself. A row that cannot
/// find itself is that, or a loader that reads the table differently from the
/// generator that wrote it.
void checkEveryRowIsReachable(Checks &checks) {
  unsigned rows = 0;
  opcost::forEachMeasurement(Arch::Blackhole, [&](const Measurement &row) {
    ++rows;
    llvm::SmallVector<Knob> knobs = splitKnobs(row.knobs);
    auto [opKey, config] = keyOf(row, knobs);
    std::optional<Cost> found =
        opcost::lookup(row.op, row.engine, opKey, config, Arch::Blackhole);
    if (!found) {
      checks.expect(false, "no answer for its own key: " + describe(row));
      return;
    }
    checks.expect(sameCost(*found, row.cost),
                  "answered a different cost than the row holds: " +
                      describe(row));
  });
  checks.expect(rows > 0, "the table holds no measurements at all");
  checks.expect(rows == opcost::getTableStats(Arch::Blackhole).measuredRows,
                "enumeration and getTableStats disagree on the row count");
  checks.finish("every measured row is reachable by its own key");
}

/// A key missing a knob the row names cannot match it, an extra knob is
/// ignored, and a wrong value misses.
///
/// The first two are what lets a caller answer every knob it can rather than
/// predict which rows a lookup will reach, and stop answering when it cannot
/// determine one.
void checkKnobDiscipline(Checks &checks) {
  bool probed = false;
  opcost::forEachMeasurement(Arch::Blackhole, [&](const Measurement &row) {
    if (probed || row.knobs.empty()) {
      return;
    }
    llvm::SmallVector<Knob> knobs = splitKnobs(row.knobs);
    auto [opKey, config] = keyOf(row, knobs);
    if (!opcost::lookup(row.op, row.engine, opKey, config, Arch::Blackhole)) {
      // A conflicted row: the reachability check above reports it. Anything
      // derived from it would report the same fault twice.
      return;
    }
    probed = true;

    // One knob dropped.
    llvm::SmallVector<Knob> fewer(knobs.begin(), knobs.end() - 1);
    OpKey dropped = opKey;
    dropped.knobs = fewer;
    checks.expect(
        !opcost::lookup(row.op, row.engine, dropped, config, Arch::Blackhole),
        "answered without a knob the row is keyed on: " + describe(row));

    // No knobs at all.
    OpKey none = opKey;
    none.knobs = {};
    checks.expect(
        !opcost::lookup(row.op, row.engine, none, config, Arch::Blackhole),
        "answered with no knobs supplied: " + describe(row));

    // One knob's value changed to something no sweep measured.
    llvm::SmallVector<Knob> wrong(knobs.begin(), knobs.end());
    wrong.back().value = "not-a-measured-value";
    OpKey mismatched = opKey;
    mismatched.knobs = wrong;
    checks.expect(
        !opcost::lookup(row.op, row.engine, mismatched, config,
                        Arch::Blackhole),
        "answered with a knob set to a value it was not measured at: " +
            describe(row));

    // A knob the row does not name, alongside the ones it does.
    llvm::SmallVector<Knob> extra(knobs.begin(), knobs.end());
    extra.push_back({"a-knob-no-row-names", "true"});
    OpKey padded = opKey;
    padded.knobs = extra;
    std::optional<Cost> withExtra =
        opcost::lookup(row.op, row.engine, padded, config, Arch::Blackhole);
    checks.expect(withExtra && sameCost(*withExtra, row.cost),
                  "an unrelated knob changed the answer: " + describe(row));
  });
  checks.expect(probed,
                "no row carries knobs, so knob matching went unchecked");
  checks.finish("knob matching needs every knob a row names, and no more");
}

/// The rest of the key has to agree too, and `dstSync` is not part of it.
void checkKeyDiscipline(Checks &checks) {
  bool probed = false;
  opcost::forEachMeasurement(Arch::Blackhole, [&](const Measurement &row) {
    if (probed) {
      return;
    }
    llvm::SmallVector<Knob> knobs = splitKnobs(row.knobs);
    auto [opKey, config] = keyOf(row, knobs);
    std::optional<Cost> found =
        opcost::lookup(row.op, row.engine, opKey, config, Arch::Blackhole);
    if (!found) {
      return;
    }
    probed = true;

    // Perturbed to values no row can hold, so a miss cannot be another row
    // legitimately matching instead.
    OpKey otherIn = opKey;
    otherIn.inFormat = "NotAFormat";
    checks.expect(
        !opcost::lookup(row.op, row.engine, otherIn, config, Arch::Blackhole),
        "answered for an input format it was not measured at: " +
            describe(row));

    KernelConfig otherOut = config;
    otherOut.outFormat = "NotAFormat";
    checks.expect(
        !opcost::lookup(row.op, row.engine, opKey, otherOut, Arch::Blackhole),
        "answered for an output format it was not measured at: " +
            describe(row));

    KernelConfig otherFaces = config;
    otherFaces.faces = 4096;
    checks.expect(
        !opcost::lookup(row.op, row.engine, opKey, otherFaces, Arch::Blackhole),
        "answered for a face count it was not measured at: " + describe(row));

    // Documented as not consulted: the sync mode changes DST capacity and the
    // math-to-pack handoff, neither of which alters an operation's isolated
    // work time, so no row is keyed on it.
    for (llvm::StringRef sync : {"Half", "Full", ""}) {
      KernelConfig synced = config;
      synced.dstSync = sync;
      std::optional<Cost> answer =
          opcost::lookup(row.op, row.engine, opKey, synced, Arch::Blackhole);
      checks.expect(answer && sameCost(*answer, *found),
                    "dstSync changed the answer: " + describe(row));
    }
  });
  checks.expect(probed, "no row was reachable, so key matching went unchecked");
  checks.finish("formats and face count are part of the key, dstSync is not");
}

/// A slot the table knows but nothing timed answers nothing, however the caller
/// asks.
///
/// Missing data, as against a measurement this configuration cannot key: the
/// two call for opposite responses, so a report has to keep them apart. A slot
/// that invented a number here would erase both.
void checkUntimedSlotsStaySilent(Checks &checks) {
  unsigned untimed = 0;
  for (llvm::StringRef op : opcost::getOperations(Arch::Blackhole)) {
    for (Engine engine : kEngines) {
      if (!opcost::runsOnEngine(op, engine, Arch::Blackhole) ||
          opcost::getMeasurementCount(op, engine, Arch::Blackhole) != 0) {
        continue;
      }
      ++untimed;
      const Knob knobs[] = {{"math_fidelity", "HiFi4"},
                            {"approx_mode", "false"},
                            {"unpack_to_dest", "false"}};
      for (const OpKey &opKey :
           {OpKey{}, OpKey{"Float16_b", knobs}, OpKey{"Float32", {}}}) {
        for (const KernelConfig &config :
             {KernelConfig{}, KernelConfig{"Float16_b", false, "Half", 4},
              KernelConfig{"Float32", true, "Full", 2}}) {
          checks.expect(
              !opcost::lookup(op, engine, opKey, config, Arch::Blackhole),
              "an untimed slot answered: " + op + "/" +
                  opcost::getEngineName(engine));
        }
      }
    }
  }
  checks.expect(
      untimed > 0,
      "no slot is untimed, so silence on missing data went unchecked");
  checks.finish("a slot with no measurements answers nothing");
}

/// The three predicates and the operation list agree with each other.
void checkSlotsAgree(Checks &checks) {
  llvm::ArrayRef<llvm::StringRef> ops = opcost::getOperations(Arch::Blackhole);
  checks.expect(ops.size() == opcost::getTableStats(Arch::Blackhole).operations,
                "getOperations and getTableStats disagree on the count");

  unsigned rows = 0;
  for (llvm::StringRef op : ops) {
    checks.expect(opcost::isKnownOp(op, Arch::Blackhole),
                  "an operation the table lists is not known: " + op);

    bool anyEngine = false;
    for (Engine engine : kEngines) {
      bool runs = opcost::runsOnEngine(op, engine, Arch::Blackhole);
      anyEngine |= runs;
      rows += opcost::getMeasurementCount(op, engine, Arch::Blackhole);
      checks.expect(
          runs || opcost::getMeasurementCount(op, engine, Arch::Blackhole) == 0,
          "measurements on an engine the operation does not run on: " + op +
              "/" + opcost::getEngineName(engine));
    }
    checks.expect(opcost::runsNowhere(op, Arch::Blackhole) == !anyEngine,
                  "runsNowhere disagrees with the engine slots: " + op);
  }
  checks.expect(rows == opcost::getTableStats(Arch::Blackhole).measuredRows,
                "per-slot counts do not sum to getTableStats");
  checks.finish("the operation list, the predicates and the counts agree");
}

/// An operation the table does not know is unknown rather than free.
void checkUnknownOperation(Checks &checks) {
  constexpr llvm::StringLiteral kAbsent("not_a_ttkernel_operation");
  checks.expect(!opcost::isKnownOp(kAbsent, Arch::Blackhole),
                "an absent operation is known");
  checks.expect(!opcost::runsNowhere(kAbsent, Arch::Blackhole),
                "an absent operation reads as running nowhere, which is what a "
                "known and free operation reads as");
  for (Engine engine : kEngines) {
    checks.expect(!opcost::runsOnEngine(kAbsent, engine, Arch::Blackhole),
                  "an absent operation runs on an engine");
    checks.expect(
        opcost::getMeasurementCount(kAbsent, engine, Arch::Blackhole) == 0,
        "an absent operation has measurements");
    checks.expect(!opcost::lookup(kAbsent, engine, OpKey{}, KernelConfig{},
                                  Arch::Blackhole),
                  "an absent operation answered a cost");
  }
  checks.finish("an unknown operation answers nothing on every engine");
}

/// An architecture with no table borrows nothing from the one that has data.
void checkOtherArchIsEmpty(Checks &checks) {
  checks.expect(opcost::hasTable(Arch::Blackhole),
                "Blackhole has no table, so nothing here means anything");
  checks.expect(!opcost::hasTable(Arch::Wormhole),
                "Wormhole reports a table it does not have");
  checks.expect(opcost::getOperations(Arch::Wormhole).empty(),
                "Wormhole lists operations");
  opcost::TableStats stats = opcost::getTableStats(Arch::Wormhole);
  checks.expect(stats.operations == 0 && stats.measuredRows == 0,
                "Wormhole reports table statistics");

  unsigned visited = 0;
  opcost::forEachMeasurement(Arch::Wormhole,
                             [&](const Measurement &) { ++visited; });
  checks.expect(visited == 0, "Wormhole enumerated measurements");

  // Asked with a key that Blackhole answers, so a miss is the architecture and
  // not the key.
  opcost::forEachMeasurement(Arch::Blackhole, [&](const Measurement &row) {
    if (visited++ > 0) {
      return;
    }
    llvm::SmallVector<Knob> knobs = splitKnobs(row.knobs);
    auto [opKey, config] = keyOf(row, knobs);
    checks.expect(!opcost::isKnownOp(row.op, Arch::Wormhole),
                  "Wormhole knows a Blackhole operation");
    checks.expect(
        !opcost::lookup(row.op, row.engine, opKey, config, Arch::Wormhole),
        "Wormhole answered a cost");
  });
  checks.finish("an architecture with no table answers nothing");
}

} // namespace

int main() {
  Checks checks;
  checkEveryRowIsReachable(checks);
  checkKnobDiscipline(checks);
  checkKeyDiscipline(checks);
  checkUntimedSlotsStaySilent(checks);
  checkSlotsAgree(checks);
  checkUnknownOperation(checks);
  checkOtherArchIsEmpty(checks);

  if (checks.failed()) {
    llvm::outs() << "opcost: checks failed\n";
    return 1;
  }
  llvm::outs() << "opcost: all checks passed\n";
  return 0;
}
