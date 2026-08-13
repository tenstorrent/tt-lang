// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Prints what the operation-cost table contains, for opcost_coverage.mlir to
// check.
//
// This is the record of the measurement data. `perf_data/` is not in the tree --
// the CSVs are large, regenerable from llk-perf/, and tied to the tt-metal
// revision that produced them -- so the coverage this prints, pinned by the lit
// test, is what makes a regeneration's effect on the data reviewable. A sweep
// that silently loses measurements changes these numbers and fails.
//
// It also exercises the query API against the real table, which is otherwise
// only reached through the cost estimator.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#include "ttlang/OpCost/OpCost.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::tt;

namespace {

constexpr opcost::Engine kEngines[] = {
    opcost::Engine::Dm, opcost::Engine::Unpack, opcost::Engine::Math,
    opcost::Engine::Pack};

/// Knobs a representative tt-lang kernel supplies.
///
/// Three different origins, which is the point of the mechanism: `approx_mode`
/// and `iterations` are kernel-wide, `unpack_to_dest` belongs to a circular
/// buffer, and a real caller would add the operation's own attributes here too.
/// The library treats all of them alike.
constexpr opcost::Knob kKernelKnobs[] = {
    {"unpack_to_dest", "false"},
    {"approx_mode", "false"},
    {"iterations", "8"},
};

/// A representative tt-lang kernel: bf16 in and out, no dest accumulation, full
/// 32x32 tiles, at ttnn's defaults for the knobs the pipeline never sets.
///
/// Reachability is quoted against one configuration because it is a property of
/// the pair, not of the data alone: a slot can hold measurements that no kernel
/// of this shape can match. Both numbers are printed for exactly that reason.
opcost::KernelConfig representativeKernel() {
  opcost::KernelConfig config;
  config.outFormat = "Float16_b";
  config.destAcc = false;
  config.fidelity = "HiFi4";
  config.dstSync = "Half";
  config.faces = 4;
  config.knobs = kKernelKnobs;
  return config;
}

void printCoverage(llvm::raw_ostream &out) {
  const opcost::KernelConfig config = representativeKernel();
  constexpr llvm::StringLiteral kInFormat("Float16_b");

  unsigned occupied[4] = {};
  unsigned measured[4] = {};
  unsigned reachable[4] = {};
  unsigned runsNowhere = 0;

  for (llvm::StringRef op : opcost::getOperations()) {
    if (opcost::runsNowhere(op)) {
      ++runsNowhere;
    }
    for (unsigned i = 0; i < 4; ++i) {
      opcost::Engine engine = kEngines[i];
      if (!opcost::runsOnEngine(op, engine)) {
        continue;
      }
      ++occupied[i];
      if (opcost::getMeasurementCount(op, engine) > 0) {
        ++measured[i];
      }
      if (opcost::lookupMeasured(op, engine, kInFormat, config)) {
        ++reachable[i];
      }
    }
  }

  opcost::TableStats stats = opcost::getTableStats();
  out << "operations " << stats.operations << "\n";
  out << "measured-rows " << stats.measuredRows << "\n";
  out << "runs-nowhere " << runsNowhere << "\n";

  // "slots" is what the table could describe, "measured" what any perf sweep
  // reached, "reachable" what this kernel shape can actually be answered for.
  // measured >= reachable always: the gap is data taken in a configuration
  // tt-lang does not generate.
  unsigned totalSlots = 0, totalMeasured = 0, totalReachable = 0;
  for (unsigned i = 0; i < 4; ++i) {
    out << "engine " << opcost::getEngineName(kEngines[i]) << " slots "
        << occupied[i] << " measured " << measured[i] << " reachable "
        << reachable[i] << "\n";
    totalSlots += occupied[i];
    totalMeasured += measured[i];
    totalReachable += reachable[i];
  }
  out << "total slots " << totalSlots << " measured " << totalMeasured
      << " reachable " << totalReachable << "\n";
}

/// One resolved lookup, in the form a caller sees it.
void printLookup(llvm::raw_ostream &out, llvm::StringRef op,
                 opcost::Engine engine) {
  const opcost::KernelConfig config = representativeKernel();
  std::optional<opcost::Cost> cost =
      opcost::lookupOrPlaceholder(op, engine, "Float16_b", config);
  out << "lookup " << op << "/" << opcost::getEngineName(engine) << " ";
  if (!cost) {
    out << "not-on-engine\n";
    return;
  }
  out << (cost->measured ? "measured" : "placeholder") << " "
      << llvm::format("%.2f", cost->value) << " "
      << (cost->unit == opcost::Unit::PerTile ? "per-tile" : "per-call") << "\n";
}

} // namespace

int main() {
  llvm::raw_ostream &out = llvm::outs();
  printCoverage(out);

  // A measured slot, an operation that runs on an engine it has no measurement
  // for, and one that runs nowhere: the three outcomes a caller must handle.
  printLookup(out, "pack_tile", opcost::Engine::Pack);
  printLookup(out, "add_tiles", opcost::Engine::Unpack);
  printLookup(out, "matmul_block", opcost::Engine::Math);
  printLookup(out, "pack_tile", opcost::Engine::Math);
  printLookup(out, "get_compile_time_arg_val", opcost::Engine::Math);

  // `lookupMeasured` refuses where `lookupOrPlaceholder` invents. This is the
  // distinction the two-call split exists for, so it is pinned rather than
  // left to the reader.
  const opcost::KernelConfig config = representativeKernel();
  out << "measured-only matmul_block/math "
      << (opcost::lookupMeasured("matmul_block", opcost::Engine::Math,
                                 "Float16_b", config)
              ? "some"
              : "none")
      << "\n";

  // A reduce needs its own attributes to be answerable: the operation's cost
  // depends on the reduce dimension and pool type, which no kernel-wide setting
  // can supply. Without them the row cannot be matched; with them it can. This
  // is the whole reason knobs are a caller-supplied list rather than fields.
  opcost::KernelConfig reduceQuery = representativeKernel();
  const opcost::Knob reduceKnobs[] = {
      {"unpack_to_dest", "false"},
      {"mathop", "ReduceColumn"},
      {"reduce_pool_type", "Max"},
  };
  reduceQuery.knobs = reduceKnobs;
  out << "reduce without op attrs "
      << (opcost::lookupMeasured("reduce_tile", opcost::Engine::Math,
                                 "Float16_b", config)
              ? "some"
              : "none")
      << "\n";
  std::optional<opcost::Cost> reduce = opcost::lookupMeasured(
      "reduce_tile", opcost::Engine::Math, "Float16_b", reduceQuery);
  out << "reduce with op attrs "
      << (reduce ? "measured" : "none");
  if (reduce) {
    out << " " << llvm::format("%.2f", reduce->value);
  }
  out << "\n";

  // Costs do not transfer across architectures, so an architecture with no
  // table answers nothing rather than borrowing Blackhole's.
  out << "wormhole operations "
      << opcost::getTableStats(opcost::Arch::Wormhole).operations << "\n";
  out << "wormhole known-op pack_tile "
      << (opcost::isKnownOp("pack_tile", opcost::Arch::Wormhole) ? "yes" : "no")
      << "\n";
  return 0;
}
