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

  /// Work the estimator could not account for.
  ///
  /// Any unknown makes the estimate untrustworthy. The estimator reports the
  /// gap rather than assuming a cost, so that an unmodelled kernel is never
  /// mistaken for a cheap one.
  struct Unknown {
    std::string message;
    Location loc;
  };

  /// One operation placed on a lane.
  struct PlacedOp {
    /// Fully qualified name, e.g. "ttkernel.cb_wait_front".
    std::string name;
    Location loc;

    /// Functions this operation expands to on this lane, in emission order:
    /// `llk_*` on the TRISCs, dataflow-API names on the data-movement cores.
    ///
    /// A compute-API call can expand unevenly across the TRISCs, which is why
    /// this is a list rather than a flag: `binary_op_init_common` is two calls
    /// on UNPACK, two on MATH and three on PACK.
    ///
    /// These reference string literals in the affinity table, so they are valid
    /// for the life of the program.
    llvm::SmallVector<llvm::StringRef> calls;

    unsigned llkCalls() const { return calls.size(); }
  };

  /// What lands on one lane, in program order. Cycle terms arrive with the cost
  /// table; this records placement only.
  ///
  /// These are static occurrences, so the list is bounded by the size of the IR
  /// rather than by loop trip counts. Weighting by execution count is a
  /// separate step.
  struct LaneReport {
    llvm::SmallVector<PlacedOp> ops;

    /// Total LLK calls on this lane.
    uint64_t llkCalls() const {
      uint64_t total = 0;
      for (const PlacedOp &op : ops) {
        total += op.llkCalls();
      }
      return total;
    }
  };

  /// Result of estimating one module.
  struct Report {
    /// Kernel-thread functions the estimator recognized, by symbol name.
    llvm::SmallVector<std::string> kernels;
    std::array<LaneReport, kNumLanes> lanes = {};
    llvm::SmallVector<Unknown> unknowns;

    /// True when every operation was accounted for.
    bool isComplete() const { return unknowns.empty(); }

    /// Deterministic human-readable report.
    std::string render() const;
  };

  struct Options {
    /// Fold TRISC engine time into the processor lane; see Lane.
    bool foldEngineIntoProcessor = true;
  };

  explicit CostEstimator(ModuleOp module);
  CostEstimator(ModuleOp module, Options options);
  ~CostEstimator();

  CostEstimator(CostEstimator &&) noexcept;
  CostEstimator &operator=(CostEstimator &&) noexcept;

  CostEstimator(const CostEstimator &) = delete;
  CostEstimator &operator=(const CostEstimator &) = delete;

  /// Estimate the whole module. Fails only when the module is not at the
  /// expected pipeline stage; per-operation gaps land in `Report::unknowns` so
  /// a partial result stays inspectable.
  FailureOr<Report> estimate();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_COSTESTIMATOR_H
