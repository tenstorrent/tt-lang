// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What one TTKernel operation costs on one execution engine.
//
// The data comes from LLK perf sweeps scoped to the configurations tt-lang can
// generate (llk-perf/), turned into a table by scripts/gen_cost_table.py. This
// header is the whole public surface; the table itself is generated and private.
//
// Deliberately depends on LLVM Support alone -- no MLIR, no dialect. Recovering
// a KernelConfig from IR is the caller's job, which keeps this library usable
// from anything and keeps a dependency edge from TTKernel back to here from ever
// existing.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_OPCOST_OPCOST_H
#define TTLANG_OPCOST_OPCOST_H

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::opcost {

/// Architecture a cost was measured on.
///
/// Part of every query rather than a build-time constant: costs are not
/// transferable between architectures, so a caller must say which one it means.
/// Only Blackhole has a table today; asking for another returns no data rather
/// than the wrong data.
enum class Arch { Blackhole, Wormhole };

/// The execution engines a TTKernel operation can occupy.
///
/// Not the same thing as the estimator's five timelines. Those are instruction
/// streams (NCRISC, TRISC0-2, BRISC); these are the units that do the work. The
/// mapping is the caller's: NCRISC and BRISC both read `Dm`, because an
/// operation does not choose which data-movement core runs it -- `ttl.noc_index`
/// on the enclosing function does.
enum class Engine { Dm, Unpack, Math, Pack };

/// What a cost is charged per.
///
/// `PerTile` costs come from a benchmark's tile loop and `PerCall` from its init
/// zone. The two are not interchangeable and the distinction cannot be recovered
/// once dropped, so it is carried rather than resolved here: only the caller
/// knows how many tiles an operation processes.
enum class Unit : uint8_t { PerCall, PerTile };

/// One benchmark knob a measurement was taken under, as a name and a value.
///
/// The generator normalises values so a caller does not have to know each knob's
/// spelling: booleans are "true"/"false", enums their bare case name, numbers
/// themselves.
struct Knob {
  llvm::StringRef name;
  llvm::StringRef value;
};

/// The configuration a measured cost is keyed on.
///
/// A cost is only meaningful for the configuration it was taken in -- the same
/// LLK at 2 faces against 4 differs by roughly half, and an FPU multiply spans
/// 22 to 88 cycles across math fidelities -- so a query that cannot supply a
/// field it needs misses rather than borrowing a neighbouring configuration.
///
/// An empty string matches any value: it is how a caller says "this kernel does
/// not constrain that", and how rows measured by benchmarks that never varied a
/// field stay reachable.
struct KernelConfig {
  /// Format the packer wrote. Empty when the kernel packs more than one, so a
  /// mixed-format kernel misses rather than picking one of them.
  llvm::StringRef outFormat;

  bool destAcc = false;

  /// Kernel-wide, from the compute descriptor. Empty matches any.
  llvm::StringRef fidelity;

  /// "Half" or "Full". Empty matches any.
  llvm::StringRef dstSync;

  /// Faces per tile. 4 is a full 32x32 tile.
  unsigned faces = 4;

  /// Everything else a row can be keyed on, whatever its origin.
  ///
  /// The library does not know what these mean and deliberately does not: some
  /// are kernel-wide (`approx_mode`, `iterations`), some belong to a circular
  /// buffer (`unpack_to_dest`), and some are attributes of the operation being
  /// asked about (`mathop` and `reduce_pool_type` for a reduce, whose math cost
  /// spans 19 to 133 cycles across their six combinations). Only the caller
  /// knows where each comes from, and a table that gains a knob then needs no
  /// change here.
  ///
  /// `unpack_to_dest` is the one most easily mistaken for kernel-wide. It is a
  /// per-buffer decision (`ttl.unpack_to_dest_fp32` holds a list of CB indices),
  /// and it is not a small effect: the same `copy_tile` reads 120.78 cycles/tile
  /// on unpack for a listed buffer against 42.20 for an unlisted one, at an
  /// otherwise identical configuration.
  ///
  /// A row naming a knob absent from this list cannot be matched, which is the
  /// point rather than a limitation: it is what stops a measurement taken in a
  /// configuration the caller cannot describe from answering anyway.
  llvm::ArrayRef<Knob> knobs;
};

/// One operation's cost on one engine.
///
/// The full measured form is `value * tiles + fixed` for `PerTile`, and `value +
/// fixed` for `PerCall`. `fixed` is the intercept of a fit against a block
/// dimension and is zero for a plain measurement; it is reported rather than
/// folded away, because a caller that scales by tile count needs it and one that
/// does not needs to know it is discarding something.
struct Cost {
  double value = 0.0;
  double fixed = 0.0;
  Unit unit = Unit::PerCall;

  /// True when this came from a perf measurement, false when it is the table's
  /// invented placeholder.
  ///
  /// Worth checking rather than assuming. 125 of 333 slots are measured, so a
  /// caller that ignores this is often reporting a guess, and the guesses are not
  /// close: `copy_tile` on math was invented at 150 and measures 19,
  /// `compute_kernel_hw_startup` on pack at 140 against 207.
  bool measured = false;

  /// Whether this can be charged as a flat number. False when `fixed` is
  /// non-zero, which a caller with no tile count has nowhere to put.
  bool isScalar() const { return fixed == 0.0; }
};

/// Whether the table knows this operation at all.
///
/// False means the operation is absent, which is different from known and free:
/// a caller should fail rather than assume zero, since the table covers every
/// operation the dialect defines and an absence means the two have drifted.
bool isKnownOp(llvm::StringRef op, Arch arch = Arch::Blackhole);

/// Whether `op` occupies `engine`.
bool runsOnEngine(llvm::StringRef op, Engine engine,
                  Arch arch = Arch::Blackhole);

/// Whether `op` occupies no engine at all: known, and costing nothing.
bool runsNowhere(llvm::StringRef op, Arch arch = Arch::Blackhole);

/// The measured cost, or nothing.
///
/// Returns a value only when a perf row matches this configuration exactly. Use
/// this when reporting an absolute number to a user, where "unknown" is a better
/// answer than a guess.
///
/// Rows that disagree by more than a hair are treated as no match: two rows
/// matching one key means the key is missing a field the measurement depended
/// on, which is how an unverifiable number would otherwise reach a report.
std::optional<Cost> lookupMeasured(llvm::StringRef op, Engine engine,
                                   llvm::StringRef inFormat,
                                   const KernelConfig &config,
                                   Arch arch = Arch::Blackhole);

/// The measured cost if one matches and is scalar, otherwise the placeholder.
///
/// Returns nothing only when the operation does not run on this engine. Check
/// `Cost::measured` before treating the result as fact -- with most slots
/// unmeasured, a caller that ignores it is usually reporting a guess.
///
/// Use this when ranking candidates against each other, where a consistent
/// invented number beats a hole.
std::optional<Cost> lookupOrPlaceholder(llvm::StringRef op, Engine engine,
                                        llvm::StringRef inFormat,
                                        const KernelConfig &config,
                                        Arch arch = Arch::Blackhole);

/// How many operations and measured rows back an architecture's table, for
/// reports that want to state their own provenance.
struct TableStats {
  unsigned operations = 0;
  unsigned measuredRows = 0;
};
TableStats getTableStats(Arch arch = Arch::Blackhole);

/// Every operation the table defines, in table order.
///
/// The table covers the whole TTKernel dialect -- generation reads the operation
/// list out of TTKernelOps.td and fails if the two disagree -- so this is also
/// the dialect's operation list as of the last regeneration.
llvm::ArrayRef<llvm::StringRef> getOperations(Arch arch = Arch::Blackhole);

/// How many measured rows back one (operation, engine), across every
/// configuration.
///
/// Zero means no measurement exists for that slot at all, which is a different
/// thing from one existing that a particular kernel cannot match -- the first is
/// missing data, the second is a key mismatch, and only the caller's config
/// decides the second. `lookupMeasured` answers the second; this answers the
/// first, and is what a coverage report needs.
unsigned getMeasurementCount(llvm::StringRef op, Engine engine,
                             Arch arch = Arch::Blackhole);

/// Name of an engine, for diagnostics.
llvm::StringRef getEngineName(Engine engine);

} // namespace mlir::tt::opcost

#endif // TTLANG_OPCOST_OPCOST_H
