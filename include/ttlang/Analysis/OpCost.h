// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What one TTKernel operation costs on one execution engine.
//
// The data comes from LLK perf sweeps scoped to the configurations tt-lang can
// generate, run and turned into a table by llk-perf/ in the
// tt-lang-ops-and-models repository -- the measurement pipeline needs a device
// and a tt-llk harness, so it lives outside this tree and this table is one of
// its outputs. This header is the whole public surface; the table itself is
// generated and private.
//
// Deliberately depends on LLVM Support alone -- no MLIR, no dialect. Recovering
// an OpKey and a KernelConfig from IR is the caller's job, which keeps this
// library usable from anything and keeps a dependency edge from TTKernel back
// to here from ever existing.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_ANALYSIS_OPCOST_H
#define TTLANG_ANALYSIS_OPCOST_H

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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
/// Not the instruction streams that issue the work -- NCRISC, TRISC0-2 and
/// BRISC -- but the units that do it. Mapping one to the other is the caller's
/// job: NCRISC and BRISC both read `Dm`, because an operation does not choose
/// which data-movement core runs it -- `ttl.noc_index` on the enclosing
/// function does.
enum class Engine { Dm, Unpack, Math, Pack };

/// What a cost is charged per.
///
/// `PerTile` costs come from a benchmark's tile loop and `PerCall` from its
/// init zone. The two are not interchangeable and the distinction cannot be
/// recovered once dropped, so it is carried rather than resolved here: only the
/// caller knows how many tiles an operation processes.
enum class Unit : uint8_t { PerCall, PerTile };

/// One benchmark knob a measurement was taken under, as a name and a value.
///
/// The generator normalises values so a caller does not have to know each
/// knob's spelling: booleans are "true"/"false", enums their bare case name,
/// numbers themselves.
struct Knob {
  llvm::StringRef name;
  llvm::StringRef value;
};

/// The half of a measurement's key that holds for a whole kernel.
///
/// A cost is only meaningful for the configuration it was taken in -- the same
/// LLK at 2 faces against 4 differs by roughly half, and an FPU multiply spans
/// 22 to 88 cycles across math fidelities -- so a query that cannot supply a
/// field it needs misses rather than borrowing a neighbouring configuration.
///
/// Every field here is a property of the kernel rather than of any operation in
/// it, so a caller recovers this once and reuses it across every lookup that
/// kernel makes. What one operation supplies is `OpKey`; keeping the two apart
/// is what stops a value belonging to one operation being asked about another.
///
/// An empty string matches any value: it is how a caller says "this kernel does
/// not constrain that", and how rows measured by benchmarks that never varied a
/// field stay reachable.
struct KernelConfig {
  /// Format the packer wrote. Empty when the kernel packs more than one, so a
  /// mixed-format kernel misses rather than picking one of them.
  ///
  /// The kernel's rather than the operation's because it cannot be anything
  /// else: a compute operation like `add_tiles` has no output buffer among its
  /// operands, while the measurement was keyed on what the packer wrote.
  llvm::StringRef outFormat;

  bool destAcc = false;

  /// "Half" or "Full".
  ///
  /// Not consulted: the mode changes DST capacity and the granularity of the
  /// math-to-pack handoff, neither of which alters an operation's isolated work
  /// time, so no measurement is keyed on it.
  llvm::StringRef dstSync;

  /// Faces per tile. 4 is a full 32x32 tile.
  unsigned faces = 4;
};

/// The half of a measurement's key that belongs to the operation being asked
/// about.
///
/// Separate from `KernelConfig` because these change from one query to the next
/// inside a single kernel, and a stale value here is a wrong answer rather than
/// a miss: charging one operation's format or one buffer's route to another
/// produces a number that looks like data.
struct OpKey {
  /// Format the engine read, from the buffer this operation reads. Empty when
  /// the caller cannot tell, which misses rather than guessing.
  ///
  /// Unlike `KernelConfig::outFormat` this one is on the operation: `copy_tile`
  /// and `add_tiles` name the buffers they unpack from. An operation that names
  /// none -- every SFPU operation, which reads DST -- leaves the caller to say
  /// what reached DST, or to leave this empty and get no answer.
  llvm::StringRef inFormat;

  /// Everything else a row can be keyed on, whatever its origin.
  ///
  /// The library does not know what these mean and deliberately does not: some
  /// are kernel-wide (`math_fidelity`, `approx_mode`), some belong to a
  /// circular buffer (`unpack_to_dest`), and some are attributes of the
  /// operation being asked about (`mathop` and `reduce_pool_type` for a reduce,
  /// whose math cost spans 19 to 133 cycles across their six combinations).
  /// Only the caller knows where each comes from, and a table that gains a knob
  /// then needs no change here.
  ///
  /// They live on this side because the ones that vary do so per operation, and
  /// a knob whose value is fixed for the kernel costs a caller nothing to
  /// repeat.
  ///
  /// `unpack_to_dest` is the one most easily mistaken for kernel-wide. It is a
  /// per-buffer decision (`ttl.unpack_to_dest_fp32` holds a list of CB
  /// indices), and it is not a small effect: the same `copy_tile` reads 120.78
  /// cycles/tile on unpack for a listed buffer against 42.20 for an unlisted
  /// one, at an otherwise identical configuration.
  ///
  /// A row naming a knob absent from this list cannot be matched, which is the
  /// point rather than a limitation: it is what stops a measurement taken in a
  /// configuration the caller cannot describe from answering anyway.
  llvm::ArrayRef<Knob> knobs;
};

/// One operation's cost on one engine.
///
/// The full measured form is `value * tiles + fixed` for `PerTile`, and `value
/// + fixed` for `PerCall`. `fixed` is the intercept of a fit against a block
/// dimension and is zero for a plain measurement; it is reported rather than
/// folded away, because a caller that scales by tile count needs it and one
/// that does not needs to know it is discarding something.
struct Cost {
  double value = 0.0;
  double fixed = 0.0;
  Unit unit = Unit::PerCall;

  /// Whether this can be charged as a flat number. False when `fixed` is
  /// non-zero, which a caller with no tile count has nowhere to put.
  bool isScalar() const { return fixed == 0.0; }
};

/// Whether any measurements exist for this architecture at all.
///
/// False means no sweep has run on it, so every other query answers nothing. A
/// caller checks this once and says so, rather than reporting each operation in
/// turn as unknown and leaving the reader to infer that the table is missing.
bool hasTable(Arch arch);

/// Whether the table knows this operation at all.
///
/// False means the operation is absent, which is different from known and free:
/// a caller should fail rather than assume zero, since the table covers every
/// operation the dialect defines and an absence means the two have drifted.
bool isKnownOp(llvm::StringRef op, Arch arch);

/// Whether `op` occupies `engine`.
bool runsOnEngine(llvm::StringRef op, Engine engine, Arch arch);

/// Whether `op` occupies no engine at all: known, and costing nothing.
bool runsNowhere(llvm::StringRef op, Arch arch);

/// The cost of this operation on this engine in this configuration, or nothing.
///
/// The only query the library has, because there is only one kind of answer it
/// can give. The table holds measurements alone: an operation on an engine
/// nothing timed, or in a configuration no sweep presented, answers nothing
/// rather than an invented number. A caller that needs a figure for every
/// operation supplies its own fallback and knows that it did.
///
/// Rows that disagree by more than a hair are treated as no match: two rows
/// matching one key means the key is missing a field the measurement depended
/// on, which is how an unverifiable number would otherwise reach a report.
std::optional<Cost> lookup(llvm::StringRef op, Engine engine,
                           const OpKey &opKey, const KernelConfig &config,
                           Arch arch);

/// How many operations and measured rows back an architecture's table, for
/// reports that want to state their own provenance.
struct TableStats {
  unsigned operations = 0;
  unsigned measuredRows = 0;
};
TableStats getTableStats(Arch arch);

/// Every operation the table defines, in table order.
///
/// The table covers the whole TTKernel dialect -- generation reads the
/// operation list out of TTKernelOps.td and fails if the two disagree -- so
/// this is also the dialect's operation list as of the last regeneration.
llvm::ArrayRef<llvm::StringRef> getOperations(Arch arch);

/// How many measured rows back one (operation, engine), across every
/// configuration.
///
/// Zero means no measurement exists for that slot at all, which is a different
/// thing from one existing that a particular kernel cannot match -- the first
/// is missing data, the second is a key mismatch, and only the caller's config
/// decides the second. `lookup` answers the second; this answers the
/// first, and is what a coverage report needs.
unsigned getMeasurementCount(llvm::StringRef op, Engine engine, Arch arch);

/// Name of an engine, for diagnostics.
llvm::StringRef getEngineName(Engine engine);

/// One measured row, for a caller checking the table rather than costing a
/// kernel: the cost, and the key it was measured at.
///
/// The key is handed back in the shape a lookup takes it, except for `knobs`,
/// which stays the packed `name=value;name=value` string the table stores. A
/// caller that wants to query with it splits it; one that wants to report on
/// the data does not have to.
struct Measurement {
  llvm::StringRef op;
  Engine engine;
  llvm::StringRef inFormat;
  llvm::StringRef outFormat;
  bool destAcc = false;
  unsigned faces = 4;
  llvm::StringRef knobs;
  Cost cost;
};

/// Every measured row backing one architecture, in table order.
///
/// The data behind every answer `lookup` gives, which is what makes the table
/// checkable from outside: a row rebuilt into a key has to find itself, and two
/// rows that share a key have to agree. Neither property can be seen through
/// `lookup` alone, because a caller cannot ask a question it cannot spell.
///
/// Enumeration rather than an array so the packed rows stay private; nothing is
/// copied but the row being visited.
void forEachMeasurement(Arch arch,
                        llvm::function_ref<void(const Measurement &)> visit);

} // namespace mlir::tt::opcost

#endif // TTLANG_ANALYSIS_OPCOST_H
