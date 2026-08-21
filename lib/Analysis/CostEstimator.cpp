// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/CostEstimator.h"
#include "ttlang/Analysis/OpCost.h"

#include "ttlang/Analysis/LoopIterationUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelTraits.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
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
using opcost::Engine;
using opcost::KernelConfig;

/// The engine a lane's work is charged to.
///
/// Five in-order instruction streams over four engines: NCRISC and BRISC both
/// read `Dm`, because an operation does not choose which data-movement core
/// runs it -- `ttl.noc_index` on the enclosing function does.
Engine engineFor(Lane lane) {
  switch (lane) {
  case Lane::Ncrisc:
  case Lane::Brisc:
    return Engine::Dm;
  case Lane::Trisc0Unpack:
    return Engine::Unpack;
  case Lane::Trisc1Math:
    return Engine::Math;
  case Lane::Trisc2Pack:
    return Engine::Pack;
  }
  llvm_unreachable("unhandled lane");
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
bool lanesAllowSrcCoupling(llvm::StringRef name, opcost::Arch arch) {
  return opcost::runsOnEngine(name, Engine::Unpack, arch) &&
         opcost::runsOnEngine(name, Engine::Math, arch);
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
/// here once the cost table covers them.
const llvm::StringSet<> &getDstSyncInitOps() {
  static const llvm::StringSet<> ops = {"binary_op_init_common",
                                        "unary_op_init_common",
                                        "init_sfpu",
                                        "compute_kernel_hw_startup",
                                        "mm_init",
                                        "mm_block_init"};
  return ops;
}

ResourceEffect getResourceEffect(Operation *op, Lane lane, opcost::Arch arch) {
  llvm::StringRef name = op->getName().stripDialect();

  // The Src handshake is the one effect that depends on which lane the
  // placement is for: the UNPACK half fills a bank and the MATH half drains
  // one. Every other effect below lands on exactly one lane, so they ignore
  // `lane`.
  bool coupled = op->hasTrait<ttkernel::TTKernelFPUOpTrait>() ||
                 getNonFpuSrcCouplingOps().contains(name);
  assert((!coupled || !opcost::isKnownOp(name, arch) ||
          lanesAllowSrcCoupling(name, arch)) &&
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

/// The cost library's name for one ttcore architecture, or nothing when it has
/// no notion of it.
///
/// Costs do not transfer across architectures, so an architecture the library
/// cannot name is one it cannot answer for -- which is a refusal, not a reason
/// to fall back on the architecture that happens to have a table.
std::optional<opcost::Arch> getOpCostArch(ttcore::Arch arch) {
  switch (arch) {
  case ttcore::Arch::Blackhole:
    return opcost::Arch::Blackhole;
  case ttcore::Arch::WormholeB0:
    return opcost::Arch::Wormhole;
  case ttcore::Arch::Quasar:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Name of an architecture for the report.
llvm::StringRef getArchName(opcost::Arch arch) {
  switch (arch) {
  case opcost::Arch::Blackhole:
    return "blackhole";
  case opcost::Arch::Wormhole:
    return "wormhole";
  }
  return "unknown";
}

/// Data format as the perf CSVs spell it.
///
/// The benchmarks name formats the way the LLK does, so the table's keys are
/// those spellings and a lookup has to translate. An unmapped type returns
/// empty, which makes the lookup miss rather than match some other format's
/// cost.
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

/// Format of the tiles a circular buffer holds, or empty when the operand is
/// not a circular buffer of tiles.
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

/// Whether an operation is one of the packer's writes.
bool isPackOp(llvm::StringRef name) {
  return name == "pack_tile" || name == "pack_tile_block";
}

/// The formats one kernel reads and writes.
///
/// Both are the kernel's rather than any one operation's, because a measurement
/// is keyed on the pair while an operation rarely names either: a compute
/// operation like `add_tiles` has no output buffer among its operands, and an
/// SFPU operation like `exp_tile` has neither -- it takes a DST index, and the
/// tile it works on reached DST from a buffer several operations earlier.
///
/// A buffer the packer writes is an output and every other one is an input,
/// which is what lets a kernel reading bf16 and packing f32 -- a pair the table
/// measures -- answer for both sides instead of calling its two formats a
/// conflict. Either side is empty when the kernel gives more than one answer
/// for it, so the lookup misses rather than picking one of them.
struct KernelFormats {
  llvm::StringRef in;
  llvm::StringRef out;
};

KernelFormats getKernelFormats(func::FuncOp funcOp) {
  /// Every circular buffer the kernel names, with the compile-time argument
  /// index that identifies it. A buffer that does not come from an argument
  /// cannot be matched against the packed set, so it counts as an input; every
  /// buffer in lowered IR does come from one.
  llvm::SmallVector<std::pair<std::optional<int32_t>, llvm::StringRef>> buffers;
  llvm::SmallDenseSet<int32_t, 4> packed;

  funcOp.walk([&](Operation *op) {
    bool packs = isPackOp(op->getName().stripDialect());
    for (Value operand : op->getOperands()) {
      llvm::StringRef format = getCbFormatName(operand);
      if (format.empty()) {
        continue;
      }
      std::optional<int32_t> index;
      if (auto argVal = operand.getDefiningOp<ttkernel::GetCompileArgValOp>()) {
        index = argVal.getArgIndex();
      }
      if (packs && index) {
        packed.insert(*index);
      }
      buffers.emplace_back(index, format);
    }
  });

  // Classified after the walk rather than during it: a buffer read early in the
  // function is not known to be an output until the pack that writes it is
  // seen.
  KernelFormats formats;
  bool inConflict = false;
  bool outConflict = false;
  for (auto [index, format] : buffers) {
    bool isOutput = index && packed.contains(*index);
    llvm::StringRef &side = isOutput ? formats.out : formats.in;
    bool &conflict = isOutput ? outConflict : inConflict;
    if (side.empty()) {
      side = format;
    } else if (side != format) {
      conflict = true;
    }
  }
  if (inConflict) {
    formats.in = {};
  }
  if (outConflict) {
    formats.out = {};
  }
  return formats;
}

/// Recover the kernel-wide half of a lookup key from one kernel function.
///
/// The other half is per placement: the format the operation reads and the
/// knobs it can answer for, assembled into an OpKey by resolve().
KernelConfig getKernelConfig(func::FuncOp funcOp,
                             const KernelFormats &formats) {
  KernelConfig config;
  if (auto destAcc =
          funcOp->getAttrOfType<BoolAttr>(ttl::kFp32DestAccEnAttrName)) {
    config.destAcc = destAcc.getValue();
  }
  if (auto fullSync =
          funcOp->getAttrOfType<BoolAttr>(ttl::kDstFullSyncEnAttrName)) {
    config.dstSync = fullSync.getValue() ? "Full" : "Half";
  }
  config.outFormat = formats.out;
  return config;
}

/// Circular buffers the unpacker writes straight to DST for, as CB indices.
///
/// A per-buffer decision rather than a kernel-wide one, and not a small effect:
/// the same `copy_tile` reads 120.78 cycles/tile on unpack for a listed buffer
/// against 42.20 for an unlisted one. The index is the compile-time argument
/// index a `get_compile_time_arg_val` carries, because that is what the ttl
/// `cb_index` lowers to, so the attribute's numbers and the operation's operand
/// are the same namespace.
llvm::SmallVector<int32_t, 4> getUnpackToDestCbs(func::FuncOp funcOp) {
  llvm::SmallVector<int32_t, 4> cbs;
  if (auto listed = funcOp->getAttrOfType<DenseI32ArrayAttr>(
          ttl::kUnpackToDestFp32AttrName)) {
    llvm::append_range(cbs, listed.asArrayRef());
  }
  return cbs;
}

/// CB index of an operation's first circular-buffer operand, or nothing when it
/// has none.
///
/// The first rather than operand zero, for the same reason the input format is
/// found that way: `pack_tile` leads with a DST index, so position is not a
/// reliable way to find the buffer.
std::optional<int32_t> getFirstCbIndex(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!mlir::isa<ttkernel::CBType>(operand.getType())) {
      continue;
    }
    if (auto argVal = operand.getDefiningOp<ttkernel::GetCompileArgValOp>()) {
      return argVal.getArgIndex();
    }
  }
  return std::nullopt;
}

/// The reduced dimension as the table spells it.
///
/// The table calls it `mathop`, which is the sweep's name for whatever the
/// benchmark varied, and for the reduce benchmarks that is the dimension. The
/// spellings are the LLK enum names, matching what
/// convert-ttkernel-to-emitc emits for the same attribute
/// (`ReduceDim::REDUCE_ROW` and friends). An unmapped case returns empty, which
/// leaves the knob unanswered and makes the row miss rather than match another
/// dimension's cost.
llvm::StringRef getReduceDimName(ttkernel::ReduceDim dim) {
  switch (dim) {
  case ttkernel::ReduceDim::Row:
    return "ReduceRow";
  case ttkernel::ReduceDim::Col:
    return "ReduceColumn";
  case ttkernel::ReduceDim::Scalar:
    return "ReduceScalar";
  }
  return {};
}

/// The pooling function as the table spells it. Avg returns empty: no sweep
/// measured it, so there is nothing for it to match.
llvm::StringRef getReducePoolTypeName(ttkernel::ReduceType type) {
  switch (type) {
  case ttkernel::ReduceType::Sum:
    return "Sum";
  case ttkernel::ReduceType::Max:
    return "Max";
  case ttkernel::ReduceType::Avg:
    return {};
  }
  return {};
}

/// Metal's default SFPU inner-loop trip count, and what tt-lang compiles at
/// every call site that does not carry the attribute. The table's SFPU rows are
/// all keyed on 8, so a kernel that asked for another count misses rather than
/// borrowing this one.
constexpr int64_t kSfpuIterationsDefault = 8;

/// The knobs one placement can answer for.
///
/// `OpKey::knobs` is deliberately untyped, so the three origins that
/// meet here are the caller's to keep straight: a kernel-wide value the IR does
/// not carry (`math_fidelity`), a per-buffer decision (`unpack_to_dest`), and
/// the attributes of the operation being asked about (`approx_mode`,
/// `iterations`, `input_clamping`, `mathop`, `reduce_pool_type`).
///
/// Supplying a knob no row names is harmless -- matching walks the row's knobs
/// and looks each up here, never the other way round -- so this answers
/// everything it can rather than trying to predict which rows the lookup will
/// reach.
///
/// Every value is a string literal but for `iterations`, whose text this owns
/// -- hence the one purpose-built setter rather than a general numeric one,
/// since a second number would move the first one's storage out from under it.
/// The knob list points into that buffer, so a set has to outlive the lookup it
/// was built for and must not be copied in between.
struct KnobSet {
  llvm::SmallVector<opcost::Knob, 4> knobs;
  llvm::SmallString<16> iterationsText;

  void add(llvm::StringRef name, llvm::StringRef value) {
    knobs.push_back({name, value});
  }

  void setIterations(int64_t value) {
    assert(iterationsText.empty() && "iterations can only be set once");
    llvm::raw_svector_ostream(iterationsText) << value;
    add("iterations", iterationsText);
  }
};

void gatherKnobs(Operation *op, llvm::StringRef mathFidelity,
                 llvm::ArrayRef<int32_t> unpackToDestCbs, KnobSet &set) {
  if (!mathFidelity.empty()) {
    set.add("math_fidelity", mathFidelity);
  }

  // The buffer the unpacker reads, which is the operation's first: the knob
  // describes an input route, and no operation keyed on it leads with an output
  // buffer.
  if (std::optional<int32_t> cb = getFirstCbIndex(op)) {
    set.add("unpack_to_dest",
            llvm::is_contained(unpackToDestCbs, *cb) ? "true" : "false");
  }

  // Read from the attributes rather than from the operation type, so the two
  // reduce operations and anything later given the same attributes are covered
  // by one rule. The attribute names are the ones TTKernelOps.td declares.
  if (auto dim = op->getAttrOfType<ttkernel::ReduceDimAttr>("reduce_dim")) {
    if (llvm::StringRef name = getReduceDimName(dim.getValue());
        !name.empty()) {
      set.add("mathop", name);
    }
  }
  if (auto type = op->getAttrOfType<ttkernel::ReduceTypeAttr>("reduce_type")) {
    if (llvm::StringRef name = getReducePoolTypeName(type.getValue());
        !name.empty()) {
      set.add("reduce_pool_type", name);
    }
  }

  // The SFPU exponential is the one operation whose own attributes reach the
  // table. Absent attributes are metal's defaults rather than unknowns: the
  // operation lowers to a bare `exp_tile(idst)` call, which is compiled with
  // them, and the Python wrapper drops every flag left at its default -- so the
  // defaults are the ordinary case rather than an edge one.
  BoolAttr approx;
  IntegerAttr iterations;
  ttkernel::InputClampingAttr clamping;
  if (auto exp = dyn_cast<ttkernel::ExpTileOp>(op)) {
    approx = exp.getApproxAttr();
    iterations = exp.getIterationsAttr();
    clamping = exp.getInputClampingAttr();
  } else if (auto expInit = dyn_cast<ttkernel::ExpTileInitOp>(op)) {
    // exp_tile_init carries no trip count of its own; it configures the SFPU
    // for the loop the following exp_tile runs.
    approx = expInit.getApproxAttr();
    clamping = expInit.getInputClampingAttr();
  } else {
    return;
  }
  set.add("approx_mode", approx && approx.getValue() ? "true" : "false");
  set.setIterations(iterations ? iterations.getInt() : kSfpuIterationsDefault);

  // The table spells the clamp as a boolean: on for ClampToNegative and off for
  // None, which skips the check. Absent means on, the template default -- and
  // that is not a detail worth glossing, because it is the difference between
  // 112 cycles/tile and 29 for the approximate exponential.
  bool clamped = !clamping || clamping.getValue() ==
                                  ttkernel::InputClamping::ClampToNegative;
  set.add("input_clamping", clamped ? "true" : "false");
}

/// What one placement is charged for a resolved cost.
///
/// The estimator charges a flat integer per placement, and one placement is one
/// call over one tile: a `PerTile` value is charged as it stands and a
/// `PerCall` value carries its intercept. A `PerTile` intercept is dropped,
/// because it is the intercept of a fit against a block dimension -- it belongs
/// to the loop rather than to any one tile of it, and once the loop is unrolled
/// there is no placement to attach it to. Every row in today's table has a zero
/// intercept, so nothing is dropped yet.
///
/// No cost means nothing measured, so nothing charged. Scheduling still gives
/// the placement a one-cost floor, so it holds its position on the lane and its
/// resource effect happens in order.
uint64_t getChargedCost(const std::optional<opcost::Cost> &cost) {
  if (!cost) {
    return 0;
  }
  double total = cost->unit == opcost::Unit::PerCall ? cost->value + cost->fixed
                                                     : cost->value;
  return static_cast<uint64_t>(std::llround(std::max(0.0, total)));
}

/// Why a placement carries no measured cost, for the detail view's `src`
/// column.
///
/// Recomputed from the table rather than stored per placement: it is a property
/// of the (operation, engine) slot, and a report holds hundreds of thousands of
/// placements.
llvm::StringRef getCostSource(CostEstimator::PlacedOp::Provenance provenance) {
  switch (provenance) {
  case CostEstimator::PlacedOp::Provenance::Measured:
    return "meas";
  // Rows exist that this kernel cannot key, which supplying the missing field
  // closes; untimed means no rows at all, which waits on a sweep.
  case CostEstimator::PlacedOp::Provenance::NoMatchingKey:
    return "nokey";
  case CostEstimator::PlacedOp::Provenance::Untimed:
    return "untimed";
  }
  return "?";
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

  uint64_t placements =
      measuredPlacements + unmatchedPlacements + untimedPlacements;
  out << "cost estimate: " << measuredPlacements << " of " << placements
      << " placements measured";
  if (placements > 0) {
    out << llvm::format(" (%.0f%%)", 100.0 * measuredPlacements / placements);
  }
  out << "\n";
  // The rest is charged nothing, and the two reasons call for opposite
  // responses: an unmatched placement is a key this kernel cannot supply, which
  // a caller can close, while an untimed one waits on a sweep.
  out << "  " << unmatchedPlacements
      << " unmatched (measured in no configuration this kernel can supply), "
      << untimedPlacements << " untimed (nothing measured them); both charged 0"
      << "\n";
  out << "  measured on " << arch << " from " << tableRows << " rows over "
      << tableOperations
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
      // Fixed width, so the source position that follows lines up.
      out << "  " << llvm::left_justify(getCostSource(op.provenance).str(), 7);
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
    out << "  ... " << (times.size() - rows) << " later event rows omitted\n";
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

    // Which architecture the module targets, before any cost is asked for. A
    // cost is only valid for the architecture it was measured on, and this is
    // the only place that knows which one the module means -- so an answer that
    // cannot be had is refused here rather than silently taken from whichever
    // architecture has a table.
    std::string reason;
    FailureOr<std::optional<ttcore::Arch>> targetArch =
        resolveTargetArch(module, reason);
    if (failed(targetArch)) {
      module.emitWarning() << "cost estimator cannot resolve the target "
                              "architecture: "
                           << reason;
      return failure();
    }
    if (!*targetArch) {
      module.emitWarning()
          << "cost estimator does not know which architecture this module "
             "targets: no '"
          << kTargetArchAttrName
          << "' and no device to read it from. Costs do not transfer between "
             "architectures, so none is assumed";
      return failure();
    }
    std::optional<opcost::Arch> resolved = getOpCostArch(**targetArch);
    if (!resolved || !opcost::hasTable(*resolved)) {
      module.emitWarning()
          << "cost estimator has no cost table for "
          << ttcore::stringifyArch(**targetArch)
          << "; costs are not transferable from another architecture";
      return failure();
    }
    arch = *resolved;
    report.arch = getArchName(arch);
    opcost::TableStats stats = opcost::getTableStats(arch);
    report.tableRows = stats.measuredRows;
    report.tableOperations = stats.operations;

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

    /// Kernel-wide, and not in the IR; see Options::mathFidelity.
    llvm::StringRef mathFidelity;

    /// The module's architecture, resolved once by estimate().
    const opcost::Arch *arch = nullptr;

    /// Input format for the operations that name no buffer; see KernelFormats.
    llvm::StringRef kernelFormat;

    /// See getUnpackToDestCbs.
    llvm::SmallVector<int32_t, 4> unpackToDestCbs;

    /// Resolved costs, keyed by everything a lookup depends on.
    ///
    /// `config` is fixed for the function while `inFormat` and the knobs come
    /// from the operation, so the answer repeats on every placement of the same
    /// operation in the same configuration. Without this the scan runs once per
    /// placement, and a kernel whose loops unroll places hundreds of thousands
    /// against slices as long as 460 rows, each row costing a `variantMatches`
    /// parse. With it the scan runs once per distinct key -- a few dozen times
    /// per function.
    ///
    /// Keyed on text rather than on pointers because the knobs are part of the
    /// key and a tuple of pointers cannot spell them. Hashing a few dozen
    /// characters is nothing against the scan it saves.
    llvm::StringMap<std::optional<opcost::Cost>> resolved;

    std::optional<opcost::Cost> resolve(Operation *op, llvm::StringRef name,
                                        Lane lane, llvm::StringRef inFormat) {
      // Outlives the lookup below and is not copied, which is what KnobSet
      // requires of whoever holds one.
      KnobSet knobs;
      gatherKnobs(op, mathFidelity, unpackToDestCbs, knobs);

      llvm::SmallString<128> key;
      {
        // Separated by a byte that cannot appear in a name or a value, so no
        // two distinct keys can spell the same string.
        llvm::raw_svector_ostream keyOut(key);
        keyOut << name << '\0' << static_cast<unsigned>(lane) << '\0'
               << inFormat;
        for (const opcost::Knob &knob : knobs.knobs) {
          keyOut << '\0' << knob.name << '=' << knob.value;
        }
      }

      auto [it, inserted] = resolved.try_emplace(key);
      if (inserted) {
        opcost::OpKey opKey{inFormat, knobs.knobs};
        it->second =
            opcost::lookup(name, engineFor(lane), opKey, config, *arch);
      }
      return it->second;
    }
  };

  /// Record a gap the estimate cannot survive: diagnose it once per distinct
  /// `key` and mark the estimate unusable.
  ///
  /// Every gap found while walking the module reports through here, so one run
  /// names all of them. `key` decides how much counts as one gap, and the right
  /// granularity differs by kind: an operation the cost table does not know is
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
    KernelFormats formats = getKernelFormats(funcOp);
    ctx.config = getKernelConfig(funcOp, formats);
    ctx.mathFidelity = options.mathFidelity;
    ctx.arch = &arch;
    ctx.kernelFormat = formats.in;
    ctx.unpackToDestCbs = getUnpackToDestCbs(funcOp);

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
    {
      llvm::StringRef name = op->getName().getStringRef();
      ++ctx.seen;

      // Messages keep the qualified name so they can be grepped against the IR.
      llvm::StringRef bare = op->getName().stripDialect();
      if (!opcost::isKnownOp(bare, arch)) {
        // Unreachable while the table covers the dialect, which generation
        // enforces by reading the op list out of TTKernelOps.td. Kept because a
        // hand-edited table would otherwise place the operation on no lane at
        // all, losing both its time and its resource effects.
        failToModel(name, op,
                    "no cost table entry for '" + name +
                        "': the operation would be left out of every lane");
        return;
      }
      // Input format comes from the operation's own circular buffer, output
      // format and the rest from the enclosing kernel; see OpKey and
      // KernelConfig.
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
      // An operation with no buffer of its own -- every SFPU operation, which
      // reads DST -- takes the kernel's.
      if (inFormat.empty()) {
        inFormat = ctx.kernelFormat;
      }

      auto place = [&](Lane lane) {
        // The table decides where an operation runs, and a missing measurement
        // does not move it: placing on a lane it does not occupy would invent
        // work, and skipping one it does occupy because nothing timed it would
        // drop its ordering and its resource effect along with its time.
        Engine engine = engineFor(lane);
        if (!opcost::runsOnEngine(bare, engine, arch)) {
          return false;
        }

        std::optional<opcost::Cost> cost =
            ctx.resolve(op, bare, lane, inFormat);
        PlacedOp::Provenance provenance;
        if (cost) {
          provenance = PlacedOp::Provenance::Measured;
          ++report.measuredPlacements;
        } else if (opcost::getMeasurementCount(bare, engine, arch) > 0) {
          provenance = PlacedOp::Provenance::NoMatchingKey;
          ++report.unmatchedPlacements;
        } else {
          provenance = PlacedOp::Provenance::Untimed;
          ++report.untimedPlacements;
        }

        ResourceEffect effect = getResourceEffect(op, lane, arch);

        PlacedOp placed{name.str(), op->getLoc(), getChargedCost(cost),
                        provenance, effect};
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
      if (!placed && !opcost::runsNowhere(bare, arch)) {
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

  /// The module's architecture, resolved by estimate() before any placement.
  opcost::Arch arch = opcost::Arch::Blackhole;

  /// DST halves available to MATH. `dst_full_sync_en` collapses them to one.
  unsigned dstHalves = 2;

  /// Gap keys already diagnosed, so each is reported once per module. See
  /// failToModel.
  llvm::StringSet<> reportedGaps;

  /// Set when the module contains anything the estimate cannot account for: an
  /// operation the cost table does not know, control flow whose outcome is not
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
