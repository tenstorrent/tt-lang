// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_ANALYSIS_COSTESTIMATOR_H
#define TTLANG_ANALYSIS_COSTESTIMATOR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace mlir::tt {

/// Estimates the cost of one TT-Lang device launch from TTKernel IR.
///
/// Expects a module lowered to TTKernel but not yet to EmitC. TTKernel ops map
/// one-to-one onto the emitted compute-API calls at that point, and the
/// constant operands carrying tile counts are already folded. Running earlier
/// misses operations that later passes insert (`ttkernel-insert-inits`,
/// `ttkernel-combine-pack-tiles`); running later leaves circular-buffer calls
/// as opaque `emitc.verbatim` strings.
class CostEstimator {
public:
  /// A hardware timeline that serializes the work placed on it. One Tensix
  /// core has five: two data-movement RISCs and the three compute TRISCs.
  ///
  /// Declaration order is dataflow order, so a forward dependence runs from a
  /// lower lane to a higher one and back-pressure runs the other way.
  ///
  /// The engines a TRISC drives (unpacker, FPU, SFPU, packer) are folded into
  /// their processor lane. That is sound only while the processor is not on the
  /// limiting dependence chain. The unpack processor is known to decouple from
  /// its engine through a two-credit config-context counter
  /// (`wait_for_next_context(2)`), so this assumption must be revisited before
  /// the estimator is trusted for unpack-bound kernels.
  enum class Lane : unsigned {
    Ncrisc = 0,   ///< RISCV_1, `ttl.noc_index` = 0, the DRAM reader preset.
    Trisc0Unpack, ///< L1 -> SrcA/SrcB.
    Trisc1Math,   ///< SrcA/SrcB -> DST.
    Trisc2Pack,   ///< DST -> L1.
    Brisc,        ///< RISCV_0, `ttl.noc_index` = 1, the DRAM writer preset.
  };

  static constexpr unsigned kNumLanes = 5;

  /// Every lane, in dataflow order.
  static llvm::ArrayRef<Lane> getAllLanes();

  /// Short display name, e.g. "TRISC0 unpack".
  static llvm::StringRef getLaneName(Lane lane);

  static constexpr unsigned getLaneIndex(Lane lane) {
    return static_cast<unsigned>(lane);
  }

  /// What an operation does to a shared resource.
  ///
  /// Every mechanism modelled here is a credit counter, so double buffering is
  /// never represented directly: a circular buffer holds `capacity` tiles and
  /// DST holds one or two halves, and whether a given acquire blocks is an
  /// outcome of the counter at that point. Raising `block_count` from 2 to 3
  /// changes no code.
  struct ResourceEffect {
    enum class Kind {
      None,
      CbReserve,  ///< producer claims tiles; blocks until that many are free
      CbPush,     ///< producer publishes its claimed tiles to the consumer
      CbWait,     ///< consumer blocks until that many tiles are published
      CbPop,      ///< consumer frees tiles back to the producer
      DstAcquire, ///< MATH blocks until a DST half is free
      DstCommit,  ///< MATH hands its half to PACK
      DstWait,    ///< PACK blocks until MATH has committed a half
      DstRelease, ///< PACK returns the half to MATH
      /// MATH re-initializes the DST pipeline: it blocks until the packer has
      /// drained every committed half, then re-seeds the handshake and resets
      /// the section base. Stricter than DstAcquire, which needs only one free
      /// half. This is a boundary rather than a stall better scheduling could
      /// hide -- sections cannot pipeline across a reset that redefines where
      /// they are.
      DstSyncInit,
      /// UNPACK fills a SrcA/SrcB bank and sets dvalid. Blocks when no bank is
      /// free, which is what stops the unpacker running further ahead of MATH
      /// than the bank count allows.
      SrcProduce,
      /// MATH consumes a bank; the math MOP's end op clears dvalid and returns
      /// it. Blocks until UNPACK has produced one.
      SrcConsume,
    };

    Kind kind = Kind::None;
    /// Compile-time arg index of the circular buffer, for the `Cb*` kinds. The
    /// same index in two different kernels is the same buffer, which is what
    /// links a reader's push to a compute kernel's wait.
    unsigned cb = 0;
    uint64_t tiles = 0;
  };

  /// One operation placed on a lane.
  struct PlacedOp {
    /// Fully qualified name, e.g. "ttkernel.cb_wait_front".
    std::string name;
    Location loc;

    /// Cost of this operation on this lane. One operation can land on several
    /// lanes with a different cost on each: `binary_op_init_common` configures
    /// the unpacker, the math sync and the packer, and the three are not equal.
    ///
    /// Zero when nothing measured stands behind this placement, which is how
    /// the circular-buffer and DST handshake operations land: the table carries
    /// measurements alone and no perf source isolates a handshake, so they are
    /// modelled as pure synchronization. An operation the table does not know
    /// at all fails the estimate instead of landing here at zero.
    uint64_t cost = 0;

    /// Where this placement's cost came from.
    ///
    /// Recorded per placement because a kernel is usually a mixture, and a
    /// mixed total should not be read as if it were measured throughout. Kept
    /// on the placement rather than recomputed when rendering, so a report
    /// answers for itself without asking the cost library again.
    enum class Provenance {
      /// A measurement keyed to this kernel's exact configuration.
      Measured,
      /// Measurements exist for this operation and engine, in no configuration
      /// this kernel can supply. Charged nothing.
      NoMatchingKey,
      /// Nothing has timed this operation on this engine. Charged nothing.
      Untimed,
    };

    Provenance provenance = Provenance::Untimed;

    bool isMeasured() const { return provenance == Provenance::Measured; }

    ResourceEffect effect;

    /// Filled in by scheduling. `stall` is the gap between the lane becoming
    /// free and this operation actually starting, so it attributes waiting to
    /// the operation that waited.
    uint64_t start = 0;
    uint64_t finish = 0;
    uint64_t stall = 0;
  };

  /// What lands on one lane, in program order.
  ///
  /// These are static occurrences, so the list is bounded by the size of the IR
  /// rather than by loop trip counts. Weighting by execution count is a
  /// separate step.
  struct LaneReport {
    llvm::SmallVector<PlacedOp> ops;
  };

  /// Result of estimating one module.
  struct Report {
    /// Kernel-thread functions the estimator recognized, by symbol name.
    llvm::SmallVector<std::string> kernels;

    /// The architecture the costs were measured on, recovered from the module.
    std::string arch;
    uint64_t tableRows = 0;
    uint64_t tableOperations = 0;
    std::array<LaneReport, kNumLanes> lanes = {};

    /// Accumulated cost at which the last lane retires. Zero when scheduling
    /// did not run.
    uint64_t totalCost = 0;

    /// How the placements split by provenance, counted per (operation, lane)
    /// placement rather than per distinct key, so the figures say how much of
    /// *this* module's work is measured rather than how much of the table was
    /// used.
    ///
    /// The three are different answers rather than degrees of one: a measured
    /// placement carries a number, an unmatched one means a measurement exists
    /// that this configuration cannot match -- a key mismatch, closed by
    /// supplying the field it is missing -- and an untimed one means no sweep
    /// timed that engine at all. Only the first contributes cost; the other two
    /// are charged nothing and occupy their lane for the one-cost floor
    /// scheduling gives every placement, so their ordering and their resource
    /// effects still hold.
    uint64_t measuredPlacements = 0;
    uint64_t unmatchedPlacements = 0;
    uint64_t untimedPlacements = 0;

    /// Summary: per-lane totals and overall latency, all in cost. This is the
    /// default output, because the per-operation views below are debugging aids
    /// that run to hundreds of thousands of rows on a real kernel.
    std::string render() const;

    /// Per-lane operation tables with start, end, cost, wait and source line.
    /// Capped per lane, and says how many it omitted.
    std::string renderDetail() const;

    /// Five-column timeline: one row per event boundary, one column per lane.
    ///
    /// Rows are the distinct starts, finishes and wait-interval boundaries, not
    /// a fixed sample rate, so no interval is invented and nothing is aliased
    /// away. The trade is that row height carries no meaning: the `gap` column
    /// is the distance to the next event on *any* lane. Empty if scheduling did
    /// not run.
    std::string renderTimeline() const;
  };

  struct Options {
    /// Fold TRISC engine time into the processor lane; see Lane.
    bool foldEngineIntoProcessor = true;

    /// SrcA/SrcB banks the unpacker can fill before it has to wait for MATH.
    /// Two on Wormhole and Blackhole, which is what lets the unpacker work one
    /// tile ahead. Exposed so a caller can check whether the credit binds at
    /// all: if 1 and 2 give the same answer, it is slack rather than a
    /// bottleneck.
    unsigned srcBanks = 2;

    /// Math fidelity as the cost table spells it: "LoFi", "HiFi2", "HiFi3" or
    /// "HiFi4". Empty when the caller does not know.
    ///
    /// The one key field the IR cannot answer, because fidelity is a runtime
    /// compute-config field that never reaches TTKernel. The rows keyed on it
    /// -- `mul_tiles` and `reduce_tile` on math, whose cost spans a factor of
    /// four across the four settings -- stay unmatched until it is supplied,
    /// which the report counts rather than papering over.
    std::string mathFidelity;
  };

  explicit CostEstimator(ModuleOp module);
  CostEstimator(ModuleOp module, Options options);
  ~CostEstimator();

  CostEstimator(CostEstimator &&) noexcept;
  CostEstimator &operator=(CostEstimator &&) noexcept;

  CostEstimator(const CostEstimator &) = delete;
  CostEstimator &operator=(const CostEstimator &) = delete;

  /// Estimate the whole module, or fail if anything in it cannot be accounted
  /// for: a module at the wrong pipeline stage, an operation the cost table
  /// does not cover, control flow whose outcome is not resolved, or a loop that
  /// cannot be unrolled. A returned report describes
  /// every operation in the module; there is no partial result, because a
  /// latency computed from an incomplete program is neither an upper nor a
  /// lower bound on the real one. Diagnostics name each gap at its own
  /// operation.
  FailureOr<Report> estimate();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_COSTESTIMATOR_H
