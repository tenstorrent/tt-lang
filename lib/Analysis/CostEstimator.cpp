// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/CostEstimator.h"

#include "ttlang/Analysis/LoopIterationUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelTraits.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <optional>

namespace mlir::tt {

namespace {

constexpr llvm::StringLiteral kThreadAttrName("ttkernel.thread");
constexpr llvm::StringLiteral kTTKernelDialect("ttkernel");

/// Loop iterations a single kernel may unroll, shared across the whole nest.
/// Exceeding it fails the estimate rather than truncating the program; see
/// placeLoop.
///
/// Counted in iterations, which is not the quantity that matters: a body of N
/// operations costs trip * N placements, and placements are what the report
/// holds and what the scheduler walks, so this bounds time and memory only
/// loosely. A 16x16 block nest over 4x4-tile blocks needs a few tens of
/// thousands of iterations and places 59k operations, which the estimator
/// handles in a few hundred milliseconds and no measurable memory over the rest
/// of a compile. This value is set well above such a kernel while still
/// bounding a pathological one, but a loop whose body is far larger can still
/// cost more than the count suggests.
constexpr std::uint64_t kUnrollBudget = 1ull << 20;

/// Output caps. A kernel whose loops unroll to tens of thousands of operations
/// would otherwise emit a report longer than anyone will read. Truncation is
/// always announced, never silent.
constexpr size_t kMaxTimelineRows = 200;
constexpr size_t kMaxListedOpsPerLane = 30;

using Lane = CostEstimator::Lane;

/// What a cost is charged per.
///
/// Kept because the two are not interchangeable and the distinction cannot be
/// recovered once dropped: the perf run measures a tile loop per tile and an
/// init zone per call. Nothing consumes it yet -- every cost is charged once per
/// call, which is right only for an operation processing one tile per call.
enum class Unit : uint8_t { PerCall, PerTile };

/// One measured cost from the nightly tt-llk perf run.
///
/// The key is everything the measurement depends on, because a cost is only
/// meaningful for the configuration it was taken in: the same LLK at 2 faces
/// against 4 differs by roughly half, and an FPU multiply spans 22 to 88 cycles
/// across math fidelities. A lookup that cannot match every field it needs has
/// to miss rather than borrow a neighbouring configuration.
///
/// Carries neither the operation nor the lane: a row is reached only through the
/// lane slot that slices it, so both are implied by where it sits.
///
/// `variant` holds the benchmark-specific knobs as a `k=v;k=v` string, in the
/// order gen_cost_table.py declared them, so a benchmark that sweeps something
/// unusual does not silently average unlike measurements together.
struct MeasuredCost {
  llvm::StringRef inFormat;
  llvm::StringRef outFormat;
  bool destAcc;
  llvm::StringRef fidelity;
  llvm::StringRef dstSync;
  unsigned faces;
  llvm::StringRef variant;

  /// What `cost` is charged per. Recorded but not yet consumed: the estimator
  /// charges every cost once per call, which is right only for an operation
  /// processing one tile per call. Operand-dependent scaling is what makes the
  /// distinction load-bearing.
  Unit unit;

  /// Kept as measured; rounding to the estimator's integer cost is the caller's
  /// decision. `fixed` is the intercept of a fit against a block dimension, and
  /// zero for a plain measurement. A row carrying one is refused rather than
  /// used as a scalar, since the intercept has nowhere to go until scaling
  /// lands.
  double cost;
  double fixed;
};

/// What one operation costs on one lane.
///
/// `placeholder` is always present and always invented -- measurements live only
/// in `kMeasured`, so there is never a question which of the two a number is. It
/// answers whenever no measured row matches the kernel's configuration, which
/// for most of the table is always.
///
/// `count == 0` means no measurement exists for this (operation, lane) at all,
/// as distinct from one existing that this kernel cannot match.
struct LaneCost {
  uint64_t placeholder;
  Unit unit;
  unsigned first = 0;
  unsigned count = 0;
};

/// One operation's entry: the lanes it runs on, and what it costs on each.
///
/// A lane with no value is a lane the operation does not run on. Every lane
/// empty is an operation that runs nowhere -- known and free, as opposed to
/// absent from the table, which is unknown and fails the estimate.
///
/// `dm` is one slot rather than one per data-movement RISC, because an operation
/// does not choose which core it runs on: `ttl.noc_index` on the enclosing
/// function does. NCRISC and BRISC read the same slot.
struct OpCost {
  llvm::StringRef op;
  std::optional<LaneCost> dm;
  std::optional<LaneCost> unpack;
  std::optional<LaneCost> math;
  std::optional<LaneCost> pack;
};

#include "CostTableBlackhole.inc"

/// The measured rows belonging to one lane slot; empty when it has none.
llvm::ArrayRef<MeasuredCost> measuredRows(const LaneCost &slot) {
  return llvm::ArrayRef(kMeasured).slice(slot.first, slot.count);
}

/// The slot an operation uses for `lane`, or nullopt when it does not run there.
const std::optional<LaneCost> &laneSlot(const OpCost &entry, Lane lane) {
  switch (lane) {
  case Lane::Ncrisc:
  case Lane::Brisc:
    return entry.dm;
  case Lane::Trisc0Unpack:
    return entry.unpack;
  case Lane::Trisc1Math:
    return entry.math;
  case Lane::Trisc2Pack:
    return entry.pack;
  }
  llvm_unreachable("unhandled lane");
}

/// Whether the operation runs on no lane at all: known, and costing nothing.
bool runsNowhere(const OpCost &entry) {
  return !entry.dm && !entry.unpack && !entry.math && !entry.pack;
}

/// The cost table, keyed by operation name.
///
/// Built once instead of binary-searched, because a lookup happens per placement
/// and a kernel whose loops unroll places hundreds of thousands of them. 296
/// entries pointing at static data; nothing is copied.
const llvm::StringMap<const OpCost *> &getCostTable() {
  static const llvm::StringMap<const OpCost *> table = [] {
    llvm::StringMap<const OpCost *> t;
    for (const OpCost &entry : kCostTable) {
      t[entry.op] = &entry;
    }
    return t;
  }();
  return table;
}

using ResourceEffect = CostEstimator::ResourceEffect;

/// Resource effect of one TTKernel operation, read from its operands.
///
/// Nothing here is hand-maintained data: the buffer identity, the tile count
/// and the capacity all come from the module. Only the op-name-to-kind mapping
/// is fixed, and that follows from the operation's own semantics. Operations
/// that unpack into SrcA/SrcB without carrying `TTKernelFPUOpTrait`. The trait
/// covers the six FPU ops; these are the datacopy and (un)tilize paths, which
/// feed Src exactly the same way but are not FPU operations. Missing one means
/// MATH is free to run before the data exists, so the assert below cross-checks
/// this list against the lanes each operation runs on.
const llvm::StringSet<> &getNonFpuSrcCouplingOps() {
  static const llvm::StringSet<> ops = {"copy_tile",
                                        "copy_block_matmul_partials",
                                        "tilize_block", "untilize_block"};
  return ops;
}

/// Whether an operation's lanes are consistent with a Src handshake: it has to
/// fill a bank on UNPACK and drain one on MATH. Used only to validate the set
/// above.
///
/// This is the half of the old per-call cross-check that survives at lane
/// granularity, and it is the half that catches the mistake that matters: an
/// operation named as Src coupled but not running on both lanes would leave
/// MATH waiting on a bank nothing fills. The converse -- an operation on both
/// lanes that only configures the unpacker -- can no longer be checked, because
/// `add_tiles` and `add_tiles_init` are indistinguishable once the call names
/// are gone.
bool lanesAllowSrcCoupling(const OpCost &entry) {
  return entry.unpack && entry.math;
}

/// Operations whose MATH half re-initializes the DST pipeline.
///
/// Their MATH half calls `llk_math_pack_sync_init`, which spins until the
/// MATH/PACK semaphore reads zero -- every committed half drained -- before
/// re-seeding that semaphore and resetting the DST section base
/// (tt_llk_blackhole/llk_lib/llk_math_common.h:130). The wait is the price of
/// the reset: the section base cannot move under a packer still reading a half.
///
/// Keyed on what convert-ttkernel-to-emitc emits rather than on the compute-API
/// wrapper of the same name, because the two disagree here. `mm_init` and
/// `mm_block_init` lower to `compute_kernel_hw_startup` plus
/// `matmul_init`/`matmul_block_init` (TTKernelToEmitC.cpp:1188-1203), and it is
/// the startup call that carries the sync; `mm_block_init_short` emits
/// `matmul_block_init` alone and so does not.
///
/// The per-operation inits that stay inside compiler-generated tile and
/// subblock loops -- add_tiles_init, copy_tile_init, matmul_block_init -- carry
/// no sync, which is what leaves the DST halves free to pipeline there. Only a
/// common init re-executed inside a user loop pays this.
///
/// tilize_init, transpose_wh_init and the bcast inits carry it too, and belong
/// here once the work table covers them.
const llvm::StringSet<> &getDstSyncInitOps() {
  static const llvm::StringSet<> ops = {"binary_op_init_common",
                                        "unary_op_init_common",
                                        "init_sfpu",
                                        "compute_kernel_hw_startup",
                                        "mm_init",
                                        "mm_block_init"};
  return ops;
}

ResourceEffect getResourceEffect(Operation *op, Lane lane) {
  llvm::StringRef name = op->getName().stripDialect();

  // The Src handshake is the one effect that depends on which lane the
  // placement is for: the UNPACK half fills a bank and the MATH half drains
  // one. Every other effect below lands on exactly one lane, so they ignore
  // `lane`.
  bool coupled = op->hasTrait<ttkernel::TTKernelFPUOpTrait>() ||
                 getNonFpuSrcCouplingOps().contains(name);
  const llvm::StringMap<const OpCost *> &table = getCostTable();
  auto entry = table.find(name);
  assert((!coupled || entry == table.end() ||
          lanesAllowSrcCoupling(*entry->second)) &&
         "Src coupling disagrees with the operation's lanes: an op that feeds "
         "SrcA/SrcB has to run on both UNPACK and MATH");
  if (coupled) {
    if (lane == Lane::Trisc0Unpack) {
      return {ResourceEffect::Kind::SrcProduce, 0, 0};
    }
    if (lane == Lane::Trisc1Math) {
      return {ResourceEffect::Kind::SrcConsume, 0, 0};
    }
    return {};
  }

  // Only the MATH half re-initializes DST. The same operation's unpack and pack
  // halves configure their own engines and leave the handshake alone.
  if (lane == Lane::Trisc1Math && getDstSyncInitOps().contains(name)) {
    return {ResourceEffect::Kind::DstSyncInit, 0, 0};
  }

  ResourceEffect::Kind kind = ResourceEffect::Kind::None;
  if (name == "cb_reserve_back") {
    kind = ResourceEffect::Kind::CbReserve;
  } else if (name == "cb_push_back") {
    kind = ResourceEffect::Kind::CbPush;
  } else if (name == "cb_wait_front") {
    kind = ResourceEffect::Kind::CbWait;
  } else if (name == "cb_pop_front") {
    kind = ResourceEffect::Kind::CbPop;
  } else if (name == "tile_regs_acquire") {
    return {ResourceEffect::Kind::DstAcquire, 0, 0};
  } else if (name == "tile_regs_commit") {
    return {ResourceEffect::Kind::DstCommit, 0, 0};
  } else if (name == "tile_regs_wait") {
    return {ResourceEffect::Kind::DstWait, 0, 0};
  } else if (name == "tile_regs_release") {
    return {ResourceEffect::Kind::DstRelease, 0, 0};
  } else {
    return {};
  }

  // Circular-buffer ops carry the buffer as operand 0 and the tile count as
  // operand 1. By this point in the pipeline canonicalization has folded the
  // count, so it is a plain constant.
  if (op->getNumOperands() < 2) {
    return {};
  }
  auto argVal = op->getOperand(0).getDefiningOp<ttkernel::GetCompileArgValOp>();
  std::optional<int64_t> tiles = getConstantIntValue(op->getOperand(1));
  if (!argVal || !tiles || *tiles <= 0) {
    return {};
  }
  return {kind, static_cast<unsigned>(argVal.getArgIndex()),
          static_cast<uint64_t>(*tiles)};
}

/// Capacity in tiles of the circular buffer an operation touches.
std::optional<uint64_t> getCbCapacity(Operation *op) {
  if (op->getNumOperands() < 1) {
    return std::nullopt;
  }
  auto cbType = mlir::dyn_cast<ttkernel::CBType>(op->getOperand(0).getType());
  if (!cbType) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(cbType.getNumTiles());
}

/// Short display name for the timeline columns.
///
/// Presentation only: a name missing from the map falls back to the operation
/// name without its dialect prefix, so an unlisted operation renders wide
/// rather than wrong. Mechanical truncation is not an option because
/// `tile_regs_*` would collide.
llvm::StringRef getShortName(llvm::StringRef opName) {
  static const llvm::StringMap<llvm::StringRef> names = [] {
    llvm::StringMap<llvm::StringRef> n;
    n["cb_wait_front"] = "cbwait";
    n["cb_pop_front"] = "cbpop";
    n["cb_reserve_back"] = "cbresv";
    n["cb_push_back"] = "cbpush";
    n["tile_regs_acquire"] = "dstacq";
    n["tile_regs_commit"] = "dstcmt";
    n["tile_regs_wait"] = "dstwait";
    n["tile_regs_release"] = "dstrel";
    n["binary_op_init_common"] = "binit";
    n["add_tiles_init"] = "addinit";
    n["add_tiles"] = "add";
    n["pack_tile"] = "pack1";
    n["pack_tile_block"] = "pack";
    n["noc_async_read_tile"] = "read";
    n["noc_async_write_tile"] = "write";
    n["noc_async_read_barrier"] = "rbar";
    n["noc_async_write_barrier"] = "wbar";
    n["get_common_arg_val"] = "arg";
    n["get_write_ptr"] = "wptr";
    n["get_read_ptr"] = "rptr";
    n["TensorAccessor"] = "tacc";
    return n;
  }();

  llvm::StringRef bare = opName;
  bare.consume_front("ttkernel.");
  auto entry = names.find(bare);
  return entry == names.end() ? bare : entry->second;
}

/// Compact source position for the report, e.g. "eltwise_add.py:34". Empty when
/// the location is not a file/line, which keeps the report readable for IR that
/// carries no debug info (hand-written test cases, for instance).
std::string formatLoc(Location loc) {
  auto fileLine = dyn_cast<FileLineColLoc>(loc);
  if (!fileLine) {
    return "";
  }
  llvm::StringRef path = fileLine.getFilename().getValue();
  llvm::StringRef base = path.rsplit('/').second;
  if (base.empty()) {
    base = path;
  }
  return (base + ":" + std::to_string(fileLine.getLine())).str();
}

/// Lane a data-movement kernel runs on. `ttl.noc_index` 0 is the reader on
/// RISCV_1 (NCRISC) and 1 is the writer on RISCV_0 (BRISC), matching
/// `getNocIndex` in TTLOpsUtils.h and the Metalium reader/writer presets.
/// A function without the attribute takes the reader default, as lowering does.
Lane getDataMovementLane(func::FuncOp funcOp) {
  auto nocIndex = funcOp->getAttrOfType<IntegerAttr>(ttl::kNocIndexAttrName);
  if (nocIndex && nocIndex.getInt() == 1) {
    return Lane::Brisc;
  }
  return Lane::Ncrisc;
}

/// Architecture the compiled-in measurements were taken on. Costs do not
/// transfer across architectures, so a caller that does not know its target must
/// not use them.
constexpr llvm::StringLiteral kPerfTableArch("blackhole");

/// Data format as the perf CSVs spell it.
///
/// The benchmarks name formats the way the LLK does, so the table's keys are
/// those spellings and a lookup has to translate. An unmapped type returns empty,
/// which makes the lookup miss rather than match some other format's cost.
llvm::StringRef getPerfFormatName(ttcore::DataType dataType) {
  switch (dataType) {
  case ttcore::DataType::BFloat16:
    return "Float16_b";
  case ttcore::DataType::Float16:
    return "Float16";
  case ttcore::DataType::Float32:
    return "Float32";
  case ttcore::DataType::BFP_BFloat8:
    return "Bfp8_b";
  case ttcore::DataType::BFP_Float8:
    return "Bfp8";
  case ttcore::DataType::Int32:
    return "Int32";
  default:
    return {};
  }
}

/// Format of the tiles a circular buffer holds, or empty when the operand is not
/// a circular buffer of tiles.
llvm::StringRef getCbFormatName(Value value) {
  auto cbType = mlir::dyn_cast<ttkernel::CBType>(value.getType());
  if (!cbType) {
    return {};
  }
  auto tileType = mlir::dyn_cast<ttcore::TileType>(cbType.getElementType());
  if (!tileType) {
    return {};
  }
  return getPerfFormatName(tileType.getDataType());
}

/// Math fidelity ttnn defaults a ComputeConfigDescriptor to. Assumed rather than
/// read; see KernelConfig.
constexpr llvm::StringLiteral kDefaultFidelity("HiFi4");

/// The configuration a measured cost is keyed on, recovered from one kernel
/// function.
///
/// Everything here is read from the module except `fidelity` and `approxMode`,
/// which the pipeline never sets. Both are kernel-wide constants that tt-metal
/// generates into the kernel's descriptor header from the compute config
/// (jit_build/genfiles.cpp:776-783 emits `MATH_FIDELITY` and `APPROX` together),
/// and ttl_api sets neither, so both are whatever ttnn defaults to. Recorded
/// here rather than guessed per lookup: the same FPU multiply spans 22 to 88
/// cycles across fidelities, so a wrong guess is a 4x error.
///
/// `faces` is 4: the estimator has no subtile path, and the only measurements
/// taken at 2 faces are the partial-face matmuls, which a 4-face key correctly
/// fails to match.
struct KernelConfig {
  llvm::StringRef outFormat;
  bool destAcc = false;
  llvm::StringRef fidelity = kDefaultFidelity;
  llvm::StringRef dstSync;
  unsigned faces = 4;

  /// ttnn's `math_approx_mode` default (kernel_types.hpp:111). Every SFPU
  /// wrapper passes this same `APPROX` constant, so it is one value for the
  /// whole kernel rather than a per-operation choice. It moves the math lane by
  /// 40-93% on the six approximation-sensitive operations and 0% on the other
  /// 21, and moves unpack and pack by 0.1-0.4%.
  bool approxMode = false;

  /// Trip count of the SFPU kernel's inner loop. tt-metal compiles 8 at 87 of
  /// its 88 SFPU call sites, and the ttkernel dialect documents the same default
  /// inline on `exp_tile`, which is one of the few operations able to carry the
  /// attribute at all. So 8 is what a tt-lang kernel runs unless an operation
  /// overrides it, and reading that override per operation is the remaining
  /// work; a kernel-wide default is right for every operation that cannot
  /// express one.
  unsigned iterations = 8;

  /// Whether the unpacker writes straight to DST, skipping SrcA. Recoverable
  /// from `ttl.unpack_to_dest_fp32` together with the CB's format and
  /// `destAcc`, but not read yet; see the keying TODO in
  /// cost_estimator_issue.md, and note that populating it alone would make an
  /// fp32 kernel reject the FPU rows that carry `unpack_to_dest=False` for no
  /// reason, since the mode does not apply to operands read through SrcA/B.
  bool unpackToDest = false;
};

/// Whether we can answer for every benchmark knob an entry was measured under.
///
/// An entry's `variant` names the knobs its benchmark swept. Some are kernel
/// configuration we can supply; the rest -- `iterations`, `fast_mode`,
/// `stable_sort`, `throttle_level` and the matmul dimensions -- describe the
/// benchmark's own setup and have no counterpart in our IR.
///
/// An entry naming one of those cannot be matched, and that is the point rather
/// than a limitation. It is what stops a `pack_tile` measured inside an SFPU
/// benchmark from answering for a `pack_tile` in an FPU kernel: the two measure
/// the same call in zones of different shape and differ by 13%, and nothing in
/// our IR says which one applies. Requiring every named knob to be one we can
/// supply picks the entry whose benchmark we can actually claim to resemble.
///
/// `iterations` is worth singling out, because it is the one knob that looks
/// answerable and is not useful. It is the trip count of the SFPU kernel's
/// `for (d = 0; d < ITERATIONS; d++)` loop, and every compute-API wrapper bakes
/// in 8 (53 occurrences across api/compute/eltwise_unary, no other value), while
/// the sweep measured 32. So answering it correctly rejects 1120 of the 1122
/// rows that name it: they measure a loop four times longer than any kernel
/// runs. That coverage needs the benchmark rerun at 8, not a change here.
bool variantMatches(llvm::StringRef variant, const KernelConfig &config) {
  while (!variant.empty()) {
    auto [entry, rest] = variant.split(';');
    variant = rest;
    auto [knob, value] = entry.split('=');
    if (knob == "unpack_to_dest") {
      if (value != (config.unpackToDest ? "True" : "False")) {
        return false;
      }
      continue;
    }
    // Kernel-wide, from the same descriptor header as MATH_FIDELITY. ttl_api
    // never sets it, so it is ttnn's default; see KernelConfig::approxMode.
    if (knob == "approx_mode") {
      if (value != (config.approxMode ? "Yes" : "No")) {
        return false;
      }
      continue;
    }
    // See KernelConfig::iterations. The sweeps now measure the value tt-metal
    // compiles, so this agrees rather than rejecting the way it did against the
    // old iterations=32 data.
    if (knob == "iterations") {
      unsigned measured = 0;
      if (value.getAsInteger(10, measured) || measured != config.iterations) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

/// Recover the configuration from one kernel function.
///
/// The output format is the kernel's, not the operation's: a compute operation
/// like `add_tiles` has no output circular buffer among its operands, while the
/// measurement was keyed on the format the packer wrote. Taken from the pack
/// operations in the function, and left empty when they disagree, so a kernel
/// packing two formats misses rather than picking one of them.
KernelConfig getKernelConfig(func::FuncOp funcOp) {
  KernelConfig config;
  if (auto destAcc =
          funcOp->getAttrOfType<BoolAttr>(ttl::kFp32DestAccEnAttrName)) {
    config.destAcc = destAcc.getValue();
  }
  if (auto fullSync =
          funcOp->getAttrOfType<BoolAttr>(ttl::kDstFullSyncEnAttrName)) {
    config.dstSync = fullSync.getValue() ? "Full" : "Half";
  }

  bool conflicting = false;
  funcOp.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().stripDialect();
    if (name != "pack_tile" && name != "pack_tile_block") {
      return;
    }
    for (Value operand : op->getOperands()) {
      llvm::StringRef format = getCbFormatName(operand);
      if (format.empty()) {
        continue;
      }
      if (config.outFormat.empty()) {
        config.outFormat = format;
      } else if (config.outFormat != format) {
        conflicting = true;
      }
    }
  });
  if (conflicting) {
    config.outFormat = {};
  }
  return config;
}

/// Whether a measured entry's key matches what we recovered.
///
/// Every field has to agree, `variant` included by way of variantMatches. A
/// caller still receives every matching entry rather than the first, so that a
/// key which cannot separate two different numbers shows up as a disagreement
/// instead of being resolved by row order.
bool keyMatches(const MeasuredCost &row, llvm::StringRef inFormat,
                const KernelConfig &config) {
  return row.inFormat == inFormat && row.outFormat == config.outFormat &&
         row.destAcc == config.destAcc && row.faces == config.faces &&
         // The eltwise CSVs carry no dest_sync column, so those rows have an
         // empty one and match any kernel's mode. Only a row that names a mode
         // has to agree with ours.
         (row.dstSync.empty() || row.dstSync == config.dstSync) &&
         // Likewise fidelity: it is empty for the SFPU benchmarks, whose cost
         // does not depend on it, and for the ops that pin it themselves.
         (row.fidelity.empty() || row.fidelity == config.fidelity) &&
         variantMatches(row.variant, config);
}

/// What one operation costs on one lane for one kernel's configuration.
struct LaneLookup {
  /// False when the operation does not run on this lane at all, in which case
  /// the other fields are meaningless and nothing is placed.
  bool onLane = false;
  uint64_t cost = 0;
  /// True when `cost` came from a measured row matching this kernel exactly,
  /// false when it fell back to the slot's placeholder.
  bool measured = false;
};

/// Resolve one (operation, lane) against the table.
///
/// A measured row wins when one matches; otherwise the slot's placeholder
/// answers, which it always can, since every lane an operation runs on carries
/// one. There is no third outcome: an operation either does not run here, or has
/// a cost.
///
/// Matching rows that disagree by more than a hair are treated as no match. Two
/// rows matching one key means the key is missing a field the measurement
/// depended on, which is exactly the mistake that would put an unverifiable
/// number into a report.
LaneLookup lookupLaneCost(const OpCost &entry, Lane lane,
                          llvm::StringRef inFormat, const KernelConfig &config) {
  const std::optional<LaneCost> &slot = laneSlot(entry, lane);
  if (!slot) {
    return {};
  }

  std::optional<double> found;
  for (const MeasuredCost &row : measuredRows(*slot)) {
    if (!keyMatches(row, inFormat, config)) {
      continue;
    }
    // Rows fitted against a block dimension carry an intercept the estimator
    // has nowhere to put yet, since it costs an operation without reading its
    // tile count. Refuse them rather than drop the constant silently.
    if (row.fixed != 0.0) {
      found.reset();
      break;
    }
    if (found && std::abs(*found - row.cost) > 0.01 * *found) {
      found.reset();
      break;
    }
    found = row.cost;
  }

  // Rounded to the nearest whole cost because the schedule counts in integers;
  // the fraction dropped is under half a cost per placement, which no ranking
  // turns on.
  if (found) {
    return {/*onLane=*/true, static_cast<uint64_t>(std::llround(*found)),
            /*measured=*/true};
  }
  return {/*onLane=*/true, slot->placeholder, /*measured=*/false};
}

} // namespace

llvm::ArrayRef<CostEstimator::Lane> CostEstimator::getAllLanes() {
  static constexpr Lane lanes[kNumLanes] = {Lane::Ncrisc, Lane::Trisc0Unpack,
                                            Lane::Trisc1Math, Lane::Trisc2Pack,
                                            Lane::Brisc};
  return lanes;
}

llvm::StringRef CostEstimator::getLaneName(Lane lane) {
  switch (lane) {
  case Lane::Ncrisc:
    return "NCRISC reader";
  case Lane::Trisc0Unpack:
    return "TRISC0 unpack";
  case Lane::Trisc1Math:
    return "TRISC1 math";
  case Lane::Trisc2Pack:
    return "TRISC2 pack";
  case Lane::Brisc:
    return "BRISC writer";
  }
  return "unknown lane";
}

std::string CostEstimator::Report::render() const {
  std::string text;
  llvm::raw_string_ostream out(text);

  uint64_t placements = measuredPlacements + unmeasuredPlacements;
  out << "cost estimate: " << measuredPlacements << " of " << placements
      << " placements measured";
  if (placements > 0) {
    out << llvm::format(" (%.0f%%)", 100.0 * measuredPlacements / placements);
  }
  out << ", the rest PLACEHOLDER\n";
  out << "  measured on " << kPerfTableArch << " from "
      << std::size(kMeasured) << " rows over " << std::size(kCostTable)
      << " operations; per-operation provenance in the detail view\n";

  out << "  kernels:";
  for (llvm::StringRef kernel : kernels) {
    out << " " << kernel;
  }
  if (kernels.empty()) {
    out << " (none)";
  }
  out << "\n";

  Lane busiestLane = Lane::Ncrisc;
  uint64_t busiestWork = 0;
  for (Lane lane : getAllLanes()) {
    const LaneReport &laneReport = lanes[getLaneIndex(lane)];
    // Occupancy, not the sum of the table's costs. The two differ for an
    // operation the table costs at zero, which still occupies its lane for the
    // one the scheduler charges as a floor, so that no interval is empty.
    uint64_t work = 0;
    for (const PlacedOp &op : laneReport.ops) {
      work += op.finish - op.start;
    }
    if (work > busiestWork) {
      busiestWork = work;
      busiestLane = lane;
    }
    // Split the non-busy time. "idle" alone conflates two situations that call
    // for opposite responses: a lane blocked on another lane is a dependency
    // signal, while a lane that has run out of work is a balance observation.
    //
    // The three partition totalCost exactly, because a lane's own span is
    // busy + stalled: each operation contributes its stall gap then its
    // occupancy, telescoping to the last finish.
    uint64_t stalled = 0;
    for (const PlacedOp &op : laneReport.ops) {
      stalled += op.stall;
    }
    uint64_t lastFinish =
        laneReport.ops.empty() ? 0 : laneReport.ops.back().finish;
    uint64_t drained = totalCost > lastFinish ? totalCost - lastFinish : 0;
    assert((laneReport.ops.empty() || work + stalled + drained == totalCost) &&
           "lane time must account for the whole run");

    out << "  " << getLaneName(lane) << ": " << laneReport.ops.size()
        << " ops, " << work << " busy, " << stalled << " stalled, " << drained
        << " drained";
    if (totalCost > 0) {
      out << llvm::format(", %.0f%% utilized", 100.0 * work / totalCost);
    }
    out << "\n";
  }

  out << "  latency: " << totalCost << "\n";
  out << "  busiest lane: " << getLaneName(busiestLane) << " (" << busiestWork
      << " busy)\n";
  return text;
}

std::string CostEstimator::Report::renderDetail() const {
  std::string text;
  llvm::raw_string_ostream out(text);

  // Widest op name across all lanes, so the source column lines up.
  size_t nameWidth = 0;
  for (const LaneReport &lane : lanes) {
    for (const PlacedOp &op : lane.ops) {
      nameWidth = std::max(nameWidth, op.name.size());
    }
  }

  for (Lane lane : getAllLanes()) {
    const LaneReport &laneReport = lanes[getLaneIndex(lane)];
    out << "\n  " << getLaneName(lane) << "\n";
    if (laneReport.ops.empty()) {
      out << "    (idle)\n";
      continue;
    }
    // The `src` column says where each cost came from, so a mixed report cannot
    // be misread as measured throughout.
    out << "    " << llvm::left_justify("op", nameWidth)
        << "  start    end   cost   wait  src\n";
    size_t listed =
        std::min<size_t>(laneReport.ops.size(), kMaxListedOpsPerLane);
    for (const PlacedOp &op :
         llvm::ArrayRef<PlacedOp>(laneReport.ops).take_front(listed)) {
      out << "    " << llvm::left_justify(op.name, nameWidth)
          << llvm::format("%7" PRIu64 "%7" PRIu64 "%7" PRIu64 "%7" PRIu64,
                          op.start, op.finish, op.cost, op.stall);
      out << (op.measured ? "  meas" : "  plac");
      std::string where = formatLoc(op.loc);
      if (!where.empty()) {
        out << "  [" << where << "]";
      }
      out << "\n";
    }
    if (listed < laneReport.ops.size()) {
      out << "    ... " << (laneReport.ops.size() - listed)
          << " more ops on this lane omitted\n";
    }
  }
  return text;
}

std::string CostEstimator::Report::renderTimeline() const {
  if (totalCost == 0) {
    return "";
  }

  // Row boundaries: every start, every finish, and the point a lane began
  // waiting, so a wait gets its own row instead of being folded into the
  // preceding op.
  llvm::SmallVector<uint64_t> times{0};
  for (const LaneReport &lane : lanes) {
    for (const PlacedOp &op : lane.ops) {
      times.push_back(op.start);
      times.push_back(op.finish);
      if (op.stall > 0) {
        times.push_back(op.start - op.stall);
      }
    }
  }
  llvm::sort(times);
  times.erase(std::unique(times.begin(), times.end()), times.end());

  size_t width = 8;
  for (const LaneReport &lane : lanes) {
    for (const PlacedOp &op : lane.ops) {
      width = std::max(width, getShortName(op.name).size() + 1);
    }
  }

  std::string text;
  llvm::raw_string_ostream out(text);
  out << "timeline: one row per event, '|' running, 'w' waiting, '^' ended, "
         "'.' idle\n\n";
  out << llvm::right_justify("cost", 8) << llvm::right_justify("gap", 7)
      << "  ";
  for (Lane lane : getAllLanes()) {
    out << llvm::left_justify(getLaneName(lane).split(' ').first.str(), width)
        << "|";
  }
  out << "\n" << std::string(17, ' ');
  for (unsigned i = 0; i < kNumLanes; ++i) {
    out << std::string(width, '-') << "|";
  }
  out << "\n";

  // A lane's operations are non-overlapping and sorted by start, so a cursor
  // that only moves forward keeps this linear in the number of rows. Scanning
  // each lane's whole list per row is quadratic, which is minutes of hang on a
  // kernel whose loops unroll to tens of thousands of operations.
  std::array<size_t, kNumLanes> cursor = {};

  size_t rows = std::min<size_t>(times.size(), kMaxTimelineRows);
  for (size_t row = 0; row < rows; ++row) {
    uint64_t now = times[row];
    bool last = row + 1 == times.size();
    out << llvm::format("%8" PRIu64, now);
    if (last) {
      out << llvm::right_justify("end", 7);
    } else {
      out << llvm::format("%7" PRIu64, times[row + 1] - now);
    }
    out << "  ";

    for (Lane lane : getAllLanes()) {
      llvm::ArrayRef<PlacedOp> ops = lanes[getLaneIndex(lane)].ops;
      size_t &at = cursor[getLaneIndex(lane)];
      while (at < ops.size() && ops[at].finish < now) {
        ++at;
      }
      // At most two operations can matter at one instant: the one ending here
      // and the one starting here, and they are adjacent.
      llvm::StringRef cell = ".";
      for (size_t k = at; k < ops.size() && k <= at + 1; ++k) {
        const PlacedOp &op = ops[k];
        if (op.start == now) {
          cell = getShortName(op.name);
          break;
        }
        if (now > op.start && now < op.finish) {
          cell = "|";
          break;
        }
        // A wait shown in preference to the previous op's end: the two share a
        // boundary and the wait is what explains the gap.
        if (op.stall > 0 && now >= op.start - op.stall && now < op.start) {
          cell = "w";
          break;
        }
        if (op.finish == now) {
          cell = "^"; // keep looking, in case the next op starts here too
        }
      }
      out << llvm::left_justify(cell.str(), width) << "|";
    }
    out << "\n";
  }
  if (rows < times.size()) {
    out << "  ... " << (times.size() - rows)
        << " later event rows omitted; use timeline-step=N for a sampled "
           "view\n";
  }
  return text;
}

std::string CostEstimator::Report::renderTimelineFixed(uint64_t step) const {
  if (totalCost == 0 || step == 0) {
    return "";
  }

  size_t width = 8;
  for (const LaneReport &lane : lanes) {
    for (const PlacedOp &op : lane.ops) {
      width = std::max(width, getShortName(op.name).size() + 1);
    }
  }

  std::string text;
  llvm::raw_string_ostream out(text);
  out << "timeline sampled every " << step
      << " of cost, '|' running, 'w' waiting, '.' idle";
  out << " (anything cheaper than " << step << " may be hidden)\n\n";
  out << llvm::right_justify("cost", 8) << "  ";
  for (Lane lane : getAllLanes()) {
    out << llvm::left_justify(getLaneName(lane).split(' ').first.str(), width)
        << "|";
  }
  out << "\n" << std::string(10, ' ');
  for (unsigned i = 0; i < kNumLanes; ++i) {
    out << std::string(width, '-') << "|";
  }
  out << "\n";

  // Forward-only cursor per lane, for the same reason as renderTimeline.
  std::array<size_t, kNumLanes> cursor = {};

  uint64_t emitted = 0;
  for (uint64_t now = 0; now <= totalCost; now += step) {
    if (++emitted > kMaxTimelineRows) {
      out << "  ... truncated at " << kMaxTimelineRows
          << " rows; raise timeline-step to cover the whole schedule\n";
      break;
    }
    uint64_t rowEnd = now + step;
    out << llvm::format("%8" PRIu64, now) << "  ";

    for (Lane lane : getAllLanes()) {
      llvm::ArrayRef<PlacedOp> ops = lanes[getLaneIndex(lane)].ops;
      size_t &at = cursor[getLaneIndex(lane)];
      while (at < ops.size() && ops[at].finish <= now) {
        ++at;
      }
      llvm::StringRef cell = ".";
      for (size_t k = at; k < ops.size() && k <= at + 1; ++k) {
        const PlacedOp &op = ops[k];
        // A start inside this row wins: it is the most informative thing that
        // happened. If two operations start in one row only the first is named,
        // which is the aliasing this view trades for proportional height.
        if (op.start >= now && op.start < rowEnd) {
          cell = getShortName(op.name);
          break;
        }
        if (op.start < now && op.finish > now) {
          cell = "|";
          break;
        }
        if (op.stall > 0 && op.start - op.stall < rowEnd && op.start > now) {
          cell = "w";
          break;
        }
      }
      out << llvm::left_justify(cell.str(), width) << "|";
    }
    out << "\n";
  }
  return text;
}

namespace {

/// Tiles of one circular buffer. `capacity` comes from the CB type, so the
/// number of blocks it holds is `capacity / tiles-per-op` and is never stored.
///
/// One counter, because the hardware has one quantity. A CB is a lock-free SPSC
/// ring holding two monotonic counters -- `tiles_received`, written only by the
/// producer, and `tiles_acked`, written only by the consumer -- so that neither
/// RISC read-modify-writes the other's word, and occupancy is their difference
/// (llk_io_pack.h:38-41). Tracking a separate reservation would add a state the
/// hardware cannot be in: `cb_reserve_back` only waits, and the write pointer
/// advances in `cb_push_back`, so reserving twice returns the same address
/// rather than two slots. Keeping the producer to one reservation at a time is
/// the SPSC discipline that ttl-insert-cb-sync and ttl-verify-dfb-spsc own.
struct CbState {
  uint64_t capacity = 0;
  uint64_t occupancy = 0; ///< pushed, not yet popped: received - acked

  uint64_t freeTiles() const { return capacity - occupancy; }
};

/// DST halves handed to PACK. `SyncHalf` gives two, so MATH can fill one while
/// PACK drains the other; `dst_full_sync_en` gives one and the two serialize.
///
/// One counter for the same reason, and here the hardware really is one: the
/// MATH_PACK semaphore. MATH posts on commit and the packer's
/// dest_section_done takes it back, while both waits only gate -- MATH stalls
/// on max, PACK stalls on zero. The half MATH writes advances at commit
/// (`dest_section_flip`), not at acquire, so acquiring twice keeps MATH on the
/// same half instead of claiming two.
struct DstState {
  unsigned halves = 2; ///< the semaphore's max count
  unsigned handed = 0; ///< committed to PACK, not yet returned == MATH_PACK
};

/// SrcA/SrcB banks between the unpacker and the Matrix Unit.
///
/// The unpacker sets dvalid on the bank it filled and the math MOP's end op
/// clears it, so this is a credit counter like the others and the ping-pong
/// between banks is emergent.
///
/// Simplification: SrcA and SrcB are counted as one credit rather than two
/// independent register files. That is exact for the AB-style ops that read
/// both and conservative for single-operand ops, which leave the other file
/// idle.
struct SrcState {
  unsigned freeBanks = 2; ///< banks the unpacker may fill
  unsigned valid = 0;     ///< filled and dvalid, not yet consumed by MATH
};

/// One lane's position in its own program.
struct LaneSim {
  size_t pc = 0;
  bool inFlight = false;
  uint64_t busyUntil = 0;
  uint64_t idleSince = 0;    ///< when the lane last became free, for stalls
  llvm::StringRef blockedOn; ///< set while a lane cannot start its next op
};

} // namespace

class CostEstimator::Impl {
public:
  Impl(ModuleOp module, Options options) : module(module), options(options) {}

  FailureOr<Report> estimate() {
    Report report;
    uint64_t ttkernelOps = 0;

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      auto thread =
          funcOp->getAttrOfType<ttkernel::ThreadTypeAttr>(kThreadAttrName);
      if (!thread) {
        continue;
      }
      report.kernels.push_back(funcOp.getSymName().str());

      switch (thread.getValue()) {
      case ttkernel::ThreadType::Noc:
        ttkernelOps += placeFunc(funcOp, getDataMovementLane(funcOp), report);
        break;
      case ttkernel::ThreadType::Compute:
        // DstSync::SyncHalf is the default, so an absent attribute means two
        // halves, not "unknown".
        if (auto fullSync =
                funcOp->getAttrOfType<BoolAttr>(ttl::kDstFullSyncEnAttrName)) {
          dstHalves = fullSync.getValue() ? 1 : 2;
        }
        ttkernelOps += placeFunc(funcOp, std::nullopt, report);
        break;
      default:
        // Structural: an unhandled thread type means the whole function's work
        // lands nowhere.
        failToModel(
            "thread-type", funcOp,
            "cost estimator does not model the kernel thread type on '" +
                funcOp.getSymName() + "'");
        break;
      }
    }

    // Reject IR from after convert-ttkernel-to-emitc: circular-buffer calls
    // have become opaque verbatim strings by then, so an empty walk here means
    // the module is past the stage this estimator reads, not that it is
    // trivial.
    if (ttkernelOps == 0 && !report.kernels.empty()) {
      module.emitWarning() << "cost estimator found no ttkernel operations in "
                           << report.kernels.size()
                           << " kernel function(s); the module is probably "
                              "already lowered to EmitC";
      return failure();
    }

    if (cannotModel) {
      // The diagnostic was already emitted at the operation responsible.
      return failure();
    }

    if (failed(schedule(report))) {
      return failure();
    }
    return report;
  }

private:
  /// Walk the five lane programs forward in time, blocking on resources.
  ///
  /// Each lane is strictly in order and blocking: these are bare RISCs spinning
  /// on a semaphore, with no reordering and nothing else to run. That makes the
  /// loop much simpler than a general resource scheduler, and it makes a stuck
  /// lane genuinely stuck.
  ///
  /// Acquires are checked and taken when an operation starts; releases apply
  /// when it retires. A push therefore becomes visible to the consumer at the
  /// end of the push, not the start.
  LogicalResult schedule(Report &report) {
    llvm::DenseMap<unsigned, CbState> cbs;
    for (auto &[index, capacity] : cbCapacity) {
      cbs[index].capacity = capacity;
    }

    DstState dst;
    dst.halves = dstHalves;

    SrcState src;
    src.freeBanks = options.srcBanks;

    std::array<LaneSim, kNumLanes> sim = {};

    auto canStart = [&](const PlacedOp &op) -> llvm::StringRef {
      const ResourceEffect &effect = op.effect;
      switch (effect.kind) {
      case ResourceEffect::Kind::CbReserve:
        return cbs[effect.cb].freeTiles() >= effect.tiles ? llvm::StringRef()
                                                          : "cb free tiles";
      case ResourceEffect::Kind::CbWait:
        return cbs[effect.cb].occupancy >= effect.tiles ? llvm::StringRef()
                                                        : "cb published tiles";
      case ResourceEffect::Kind::DstAcquire:
        // Stalls on max, like the SEMWAIT it stands for. It claims nothing, so
        // a second acquire with no commit between them passes, as on hardware.
        return dst.handed < dst.halves ? llvm::StringRef() : "dst half";
      case ResourceEffect::Kind::DstWait:
        return dst.handed > 0 ? llvm::StringRef() : "dst commit";
      case ResourceEffect::Kind::DstSyncInit:
        // Nothing handed out is the semaphore reading zero. A half MATH holds
        // but has not committed does not hold the reset up, matching the
        // hardware, which waits only on what the packer still owes.
        return dst.handed == 0 ? llvm::StringRef() : "dst drain";
      case ResourceEffect::Kind::SrcProduce:
        return src.freeBanks > 0 ? llvm::StringRef() : "srcA/B bank";
      case ResourceEffect::Kind::SrcConsume:
        return src.valid > 0 ? llvm::StringRef() : "srcA/B dvalid";
      default:
        return llvm::StringRef();
      }
    };

    // Only the Src handshake takes anything when an operation starts. The CB
    // and DST waits are gates: they read their counter and never move it, so
    // nothing here corresponds to `cb_reserve_back` or `tile_regs_acquire`.
    auto takeAtStart = [&](const PlacedOp &op) {
      const ResourceEffect &effect = op.effect;
      if (effect.kind == ResourceEffect::Kind::SrcProduce) {
        --src.freeBanks;
      } else if (effect.kind == ResourceEffect::Kind::SrcConsume) {
        --src.valid;
      }
    };

    auto releaseAtFinish = [&](const PlacedOp &op) {
      const ResourceEffect &effect = op.effect;
      CbState &cb = cbs[effect.cb];
      switch (effect.kind) {
      case ResourceEffect::Kind::CbPush:
        // The producer's `tiles_received += n`, which is also what advances the
        // write pointer, so the tiles become visible to the consumer here.
        cb.occupancy += effect.tiles;
        break;
      case ResourceEffect::Kind::CbPop:
        // The consumer's `tiles_acked += n`.
        cb.occupancy -= std::min(cb.occupancy, effect.tiles);
        break;
      case ResourceEffect::Kind::DstCommit:
        ++dst.handed; ///< t6_semaphore_post(MATH_PACK)
        break;
      case ResourceEffect::Kind::DstRelease:
        // The packer's dest_section_done is the semaphore_get matching MATH's
        // post at commit, and it runs after the pack itself has finished, so
        // the half comes back here rather than when PACK started draining it.
        dst.handed -= std::min<unsigned>(dst.handed, 1);
        break;
      case ResourceEffect::Kind::SrcProduce:
        // dvalid is set when the unpack retires, so MATH cannot start on this
        // tile before then.
        ++src.valid;
        break;
      case ResourceEffect::Kind::SrcConsume:
        ++src.freeBanks;
        break;
      default:
        break;
      }
    };

    uint64_t now = 0;
    while (true) {
      // Retire whatever finishes at `now` before anything starts, so a push and
      // the wait it satisfies cannot both resolve in the same instant.
      for (Lane lane : getAllLanes()) {
        LaneSim &laneSim = sim[getLaneIndex(lane)];
        if (!laneSim.inFlight || laneSim.busyUntil != now) {
          continue;
        }
        releaseAtFinish(report.lanes[getLaneIndex(lane)].ops[laneSim.pc]);
        laneSim.inFlight = false;
        laneSim.idleSince = now;
        ++laneSim.pc;
      }

      bool anyRemaining = false;
      bool anyInFlight = false;
      for (Lane lane : getAllLanes()) {
        LaneSim &laneSim = sim[getLaneIndex(lane)];
        LaneReport &laneReport = report.lanes[getLaneIndex(lane)];
        if (laneSim.inFlight) {
          anyInFlight = anyRemaining = true;
          continue;
        }
        if (laneSim.pc >= laneReport.ops.size()) {
          continue;
        }
        anyRemaining = true;

        PlacedOp &op = laneReport.ops[laneSim.pc];
        laneSim.blockedOn = canStart(op);
        if (!laneSim.blockedOn.empty()) {
          continue;
        }
        takeAtStart(op);
        op.start = now;
        op.stall = now - laneSim.idleSince;
        op.finish = now + std::max<uint64_t>(op.cost, 1);
        laneSim.busyUntil = op.finish;
        laneSim.inFlight = true;
        anyInFlight = true;
      }

      if (!anyRemaining) {
        break;
      }

      if (!anyInFlight) {
        // Every lane with work left is blocked and nothing is running, so no
        // resource can ever be released. Report it rather than returning a
        // plausible number for a program that cannot run.
        std::string detail;
        llvm::raw_string_ostream os(detail);
        for (Lane lane : getAllLanes()) {
          const LaneSim &laneSim = sim[getLaneIndex(lane)];
          if (laneSim.pc < report.lanes[getLaneIndex(lane)].ops.size()) {
            os << "\n    " << getLaneName(lane) << " blocked on "
               << laneSim.blockedOn;
          }
        }
        module.emitWarning()
            << "cost estimator deadlocked at cost " << now << detail;
        return failure();
      }

      uint64_t next = std::numeric_limits<uint64_t>::max();
      for (Lane lane : getAllLanes()) {
        const LaneSim &laneSim = sim[getLaneIndex(lane)];
        if (laneSim.inFlight) {
          next = std::min(next, laneSim.busyUntil);
        }
      }
      now = next;
      report.totalCost = std::max(report.totalCost, now);
    }

    return success();
  }

  /// Place every TTKernel operation in one kernel function.
  ///
  /// `dmLane` is set for a data-movement kernel, which compiles for a single
  /// RISC so all of its work lands on that one lane. For a compute kernel it is
  /// nullopt and each operation fans out onto the TRISCs whose macro keeps it.
  /// Returns the number of TTKernel operations seen, placed or not.
  /// Shared state for one function's traversal.
  struct PlaceContext {
    std::optional<Lane> dmLane;
    Report *report;
    uint64_t seen = 0;

    /// The measured table's key fields that come from the enclosing function
    /// rather than from each operation.
    KernelConfig config;

    /// Resolved costs, keyed by everything a lookup depends on.
    ///
    /// `config` is fixed for the function and `inFormat` comes from the
    /// operation, so the answer repeats on every placement of the same
    /// operation. Without this the scan runs once per placement, and a kernel
    /// whose loops unroll places hundreds of thousands against slices as long as
    /// 460 rows, each row costing a `variantMatches` parse. With it the scan
    /// runs once per distinct key -- a few dozen times per function.
    llvm::DenseMap<std::tuple<const OpCost *, Lane, const char *>, LaneLookup>
        resolved;

    LaneLookup resolve(const OpCost &entry, Lane lane,
                       llvm::StringRef inFormat) {
      auto key = std::make_tuple(&entry, lane, inFormat.data());
      auto [it, inserted] = resolved.try_emplace(key);
      if (inserted) {
        it->second = lookupLaneCost(entry, lane, inFormat, config);
      }
      return it->second;
    }
  };

  /// Record a gap the estimate cannot survive: diagnose it once per distinct
  /// `key` and mark the estimate unusable.
  ///
  /// Every gap found while walking the module reports through here, so one run
  /// names all of them. `key` decides how much counts as one gap, and the right
  /// granularity differs by kind: an operation the work table does not cover is
  /// keyed by name, because an unrolled loop body would otherwise repeat the
  /// same complaint once per iteration, while a loop the estimator cannot
  /// enumerate is keyed by what went wrong with it. Dedup spans the module
  /// rather than one kernel, because a three-thread module hits the same gap
  /// three times and the reader learns nothing from the repeats.
  ///
  /// A warning rather than an error. The gap is in the estimator, and the
  /// estimate is a read-only side output, so a module the estimator cannot
  /// account for still compiles; `estimate()` returning failure is what tells a
  /// caller not to trust a number.
  void failToModel(llvm::StringRef key, Operation *op,
                   const llvm::Twine &message) {
    if (reportedGaps.insert(key).second) {
      op->emitWarning() << message;
    }
    cannotModel = true;
  }

  uint64_t placeFunc(func::FuncOp funcOp, std::optional<Lane> dmLane,
                     Report &report) {
    PlaceContext ctx;
    ctx.dmLane = dmLane;
    ctx.report = &report;
    ctx.config = getKernelConfig(funcOp);

    // Walk regions structurally rather than with Operation::walk, because a
    // loop body has to be repeated in place: the correct lane order for a body
    // of A,B,C over two iterations is A,B,C,A,B,C, not A,A,B,B,C,C.
    LoopInductionBindings bindings;
    EnumerationBudget budget(kUnrollBudget);
    placeBlock(funcOp.getBody().front(), ctx, bindings, budget);
    return ctx.seen;
  }

  void placeBlock(Block &block, PlaceContext &ctx,
                  LoopInductionBindings &bindings, EnumerationBudget &budget) {
    for (Operation &op : block) {
      placeOperation(&op, ctx, bindings, budget);
    }
  }

  void placeOperation(Operation *op, PlaceContext &ctx,
                      LoopInductionBindings &bindings,
                      EnumerationBudget &budget) {
    if (op->getName().getDialectNamespace() == kTTKernelDialect) {
      placeTTKernelOp(op, ctx);
      return;
    }
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
      placeLoop(loop, ctx, bindings, budget);
      return;
    }
    // Any other region-carrying operation is control flow this does not model.
    // Walking into it would count a guarded body as taken and both arms of an
    // if/else as executed.
    //
    // TODO(ttl): resolve the condition rather than failing. Induction variables
    // are already bound during unrolling, so
    // evaluateIndexExpression(ifOp.getCondition(), bindings) folds the bounds
    // guards that dominate real kernels, and the live branch can be placed on
    // its own. Launch-coordinate predicates (ttl.is_src, ttl.is_dst) need the
    // same facts as LaunchNodeDomainAnalysis, or `specialize_cores` to have
    // folded them first. Only a genuinely dynamic condition need fail, because
    // two branches that mutate resource counters differently leave no sound
    // single state to continue from.
    if (op->getNumRegions() > 0) {
      failToModel(op->getName().getStringRef(), op,
                  "cost estimator currently does not model '" +
                      op->getName().getStringRef() +
                      "': a guarded body would be counted as taken and both "
                      "arms of a branch as executed");
    }
  }

  /// Repeat a loop body once per iteration, with the induction variables bound
  /// so that expressions inside the body see the current iteration.
  void placeLoop(LoopLikeOpInterface loop, PlaceContext &ctx,
                 LoopInductionBindings &bindings, EnumerationBudget &budget) {
    std::optional<uint64_t> trip = getLoopTripCount(loop, bindings);
    if (!trip) {
      // TODO(ttl): a dynamic trip count cannot be unrolled at all, so this
      // needs steady-state extrapolation rather than a larger budget: unroll a
      // bounded prefix, detect when the resource state and per-lane program
      // counters repeat, take that cost delta as the initiation interval, and
      // report prologue + II * (trip - warmup) + epilogue with the trip count
      // left symbolic. Until then, failing is the honest answer -- placing
      // nothing for the loop would schedule a different program.
      failToModel("loop-trip-count", loop,
                  "cost estimator cannot determine this loop's trip count "
                  "statically, and steady-state extrapolation is not "
                  "implemented yet, so no estimate is produced");
      return;
    }
    // Skipping the body would leave the lanes holding a different program than
    // the one being estimated, and scheduling that still yields a
    // confident-looking latency which is neither an upper nor a lower bound on
    // the real one.
    if (!budget.canConsume(*trip)) {
      // The budget is shared across the enclosing nest, so this loop is usually
      // an inner one that would have fit on its own and the iterations were
      // spent by the loops around it. Saying so matters: the trip count named
      // here is not what the limit was compared against.
      failToModel(
          "loop-unroll-budget", loop,
          "cost estimator cannot unroll this loop's " + std::to_string(*trip) +
              " iterations: the per-kernel budget of " +
              std::to_string(kUnrollBudget) +
              " iterations is exhausted, spent on this loop and the ones "
              "enclosing it. Steady-state extrapolation is not "
              "implemented yet, so no estimate is produced.");
      return;
    }

    LogicalResult enumerated =
        enumerateLoopNest({loop}, bindings, budget,
                          [&](const LoopInductionBindings &) -> LogicalResult {
                            for (Region &region : loop->getRegions()) {
                              for (Block &block : region) {
                                placeBlock(block, ctx, bindings, budget);
                              }
                            }
                            return success();
                          });
    // Worse than the case above: enumeration stopped partway, so some
    // iterations are already placed and the lanes hold a truncated program.
    if (failed(enumerated)) {
      failToModel("loop-enumeration", loop,
                  "cost estimator exhausted its unroll budget of " +
                      std::to_string(kUnrollBudget) +
                      " iterations partway through this loop, or hit an "
                      "iteration range it cannot enumerate");
    }
  }

  void placeTTKernelOp(Operation *op, PlaceContext &ctx) {
    Report &report = *ctx.report;
    const std::optional<Lane> dmLane = ctx.dmLane;
    const llvm::StringMap<const OpCost *> &table = getCostTable();
    {
      llvm::StringRef name = op->getName().getStringRef();
      ++ctx.seen;

      // Messages keep the qualified name so they can be grepped against the IR.
      auto found = table.find(op->getName().stripDialect());
      if (found == table.end()) {
        // Unreachable while the table covers the dialect, which generation
        // enforces by reading the op list out of TTKernelOps.td. Kept because a
        // hand-edited table would otherwise place the operation on no lane at
        // all, losing both its time and its resource effects.
        failToModel(name, op,
                    "no cost table entry for '" + name +
                        "': the operation would be left out of every lane");
        return;
      }
      const OpCost &entry = *found->second;

      // Input format comes from the operation's own circular buffer, output
      // format and the rest from the enclosing kernel; see KernelConfig.
      //
      // The first CB operand rather than operand zero: `pack_tile` leads with a
      // DST index, so position is not a reliable way to find the buffer.
      llvm::StringRef inFormat;
      for (Value operand : op->getOperands()) {
        inFormat = getCbFormatName(operand);
        if (!inFormat.empty()) {
          break;
        }
      }

      auto place = [&](Lane lane) {
        LaneLookup cost = ctx.resolve(entry, lane, inFormat);
        if (!cost.onLane) {
          return false;
        }
        if (cost.measured) {
          ++report.measuredPlacements;
        } else {
          ++report.unmeasuredPlacements;
        }

        ResourceEffect effect = getResourceEffect(op, lane);

        PlacedOp placed{name.str(), op->getLoc(), cost.cost, cost.measured,
                        effect};
        if (placed.effect.kind != ResourceEffect::Kind::None &&
            placed.effect.tiles > 0) {
          if (std::optional<uint64_t> capacity = getCbCapacity(op)) {
            uint64_t &known = cbCapacity[placed.effect.cb];
            known = std::max(known, *capacity);
          }
        }
        report.lanes[getLaneIndex(lane)].ops.push_back(std::move(placed));
        return true;
      };

      bool placed = false;
      if (dmLane) {
        // A data-movement kernel compiles for one RISC, so all of its work
        // lands on that lane; both read the entry's single `dm` slot.
        placed = place(*dmLane);
      } else {
        placed |= place(Lane::Trisc0Unpack);
        placed |= place(Lane::Trisc1Math);
        placed |= place(Lane::Trisc2Pack);
      }

      // The table knows this operation, but not for the kind of kernel it
      // appeared in. Placing nothing would silently report it as free.
      if (!placed && !runsNowhere(entry)) {
        failToModel(name, op,
                    "'" + name + "' runs on no lane of a " +
                        (dmLane ? "data-movement" : "compute") +
                        " kernel, so it would be left out of every lane");
      }
    }
  }

  /// Capacity in tiles per circular buffer, keyed by compile-time arg index and
  /// gathered while placing operations.
  llvm::DenseMap<unsigned, uint64_t> cbCapacity;

  /// DST halves available to MATH. `dst_full_sync_en` collapses them to one.
  unsigned dstHalves = 2;

  /// Gap keys already diagnosed, so each is reported once per module. See
  /// failToModel.
  llvm::StringSet<> reportedGaps;

  /// Set when the module contains anything the estimate cannot account for: an
  /// operation the work table does not cover, control flow whose outcome is not
  /// resolved, or a loop that cannot be unrolled. The lanes then describe
  /// either a different program than the real one or the same program with work
  /// missing from it, and a latency computed from either is neither an upper
  /// nor a lower bound on the real one, so estimate() fails rather than
  /// reporting it.
  bool cannotModel = false;

  ModuleOp module;
  Options options;
};

CostEstimator::CostEstimator(ModuleOp module)
    : CostEstimator(module, Options{}) {}

CostEstimator::CostEstimator(ModuleOp module, Options options)
    : impl(std::make_unique<Impl>(module, options)) {}

CostEstimator::~CostEstimator() = default;
CostEstimator::CostEstimator(CostEstimator &&) noexcept = default;
CostEstimator &CostEstimator::operator=(CostEstimator &&) noexcept = default;

FailureOr<CostEstimator::Report> CostEstimator::estimate() {
  return impl->estimate();
}

} // namespace mlir::tt
