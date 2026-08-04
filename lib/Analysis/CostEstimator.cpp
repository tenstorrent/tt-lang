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

/// What one TTKernel operation costs on each lane it runs on.
///
/// An empty slot means the operation does no work on that lane: it is not
/// placed there, has no resource effect there and takes no time there. A value
/// means it is what the operation costs there.
///
/// A lane with no measurement behind it carries 1, not 0. One is what the
/// scheduler charges as its floor, so the cost a lane declares and the span it
/// occupies agree, and it never makes the claim a 0 would: that the lane does
/// no work and an operation placed there is free.
///
/// `dm` is one slot rather than one per data-movement core because the
/// operation does not choose which core it runs on: `ttl.noc_index` on the
/// enclosing function does. The other three are the per-TRISC halves inside a
/// compute kernel.
///
/// Lanes and costs are one table because a lane is exactly what a cost is
/// keyed on. The `llk_*` calls an operation expands to are not recorded: cost
/// is measured per operation per lane (see scripts/gen_cost_table.py), and a
/// resource effect is a property of the operation rather than of any one call
/// it makes.
///
/// The `= std::nullopt` defaults let an entry name only the lanes it uses;
/// without them -Wmissing-field-initializers rejects the omitted trailing
/// members.
struct ThreadWork {
  std::optional<uint64_t> dm = std::nullopt;
  std::optional<uint64_t> unpack = std::nullopt;
  std::optional<uint64_t> math = std::nullopt;
  std::optional<uint64_t> pack = std::nullopt;

  /// True when the operation runs on no lane at all. Distinct from being absent
  /// from the table, which means unknown, and from a zero cost, which still
  /// places the operation so that its resource effect applies.
  bool isFree() const { return !dm && !unpack && !math && !pack; }
};

/// Per-lane work for TTKernel operations, keyed by operation name.
///
/// >>> THE CYCLE COUNTS ARE PLACEHOLDERS, NOT MEASUREMENTS. <<<
///
/// They are ordered sensibly relative to each other -- a semaphore poll is
/// cheaper than a tile of unpack, which is cheaper than a tile of FPU work --
/// and nothing else, so the shape of a report is meaningful while its
/// magnitudes are not. Replace with the measured per-operation, per-lane
/// numbers from scripts/gen_cost_table.py before any decision depends on them.
///
/// Known inaccuracy: a cost here is flat per operation, but some operations
/// scale with a tile count taken from an operand. pack_tile_block in particular
/// packs `ntiles` tiles in a loop, so a flat cost is wrong for any block bigger
/// than one tile. The measured data is keyed on the same operation and lane but
/// carries a per-tile term, so operand-dependent scaling arrives with it.
///
/// Which lanes an operation runs on is a fact about the compiled program rather
/// than a measurement. A compute kernel is one source file compiled three
/// times, once per TRISC, with -DTRISC_UNPACK / -DTRISC_MATH / -DTRISC_PACK.
/// The UNPACK(), MATH() and PACK() macros in api/compute/common_globals.h keep
/// only the calls belonging to the thread being compiled and erase the rest, so
/// the wrapper around a call is what decides which TRISC runs it. An operation
/// is on a lane below when its wrapper keeps at least one call for that thread.
/// Read off the non-Quasar branch of the tt-metal headers:
///
///   circular_buffer.h:31-69 (COMPILE_FOR_TRISC path)
///     wait_front, pop_front    -> UNPACK
///     reserve_back, push_back  -> PACK
///
///   reg_api.h:45-89
///     tile_regs_acquire, tile_regs_commit -> MATH
///     tile_regs_wait, tile_regs_release   -> PACK
///
///   eltwise_binary.h:31-55   binary_op_init_common -> UNPACK, MATH, PACK
///   eltwise_binary.h:72-83, 128-132  add_tiles_init -> UNPACK, MATH
///     (UNPACK only because binary_tiles_init passes full_init = true, which
///     keeps the call inside its `if constexpr (full_init)` guard)
///   eltwise_binary.h:206-214 add_tiles -> UNPACK, MATH
///   pack.h:128-135           pack_tile_block -> PACK
///
/// `state_configure()` and `LLK_SAN_FUNCTION()` appear in several of these but
/// are sentinel/sanitizer hooks that compile to nothing in a normal build, so
/// an operation whose wrapper keeps only those counts as free rather than as
/// work.
const llvm::StringMap<ThreadWork> &getThreadWorkTable() {
  static const llvm::StringMap<ThreadWork> table = [] {
    // Keys omit the `ttkernel.` prefix; the lookup strips it. That avoids a
    // string concatenation and allocation per entry at first use.
    llvm::StringMap<ThreadWork> t;

    // Slot order is dm, unpack, math, pack; trailing slots left off are empty,
    // and `{}` is an empty slot the entry has to name to reach a later one.
    // Written with /*name=*/ comments because designated initializers are C++20
    // and this builds as C++17.

    // -- Circular buffers, api/dataflow/circular_buffer.h:31-69 -------------
    // Under COMPILE_FOR_TRISC the four methods are wrapped PACK/PACK/UNPACK/
    // UNPACK; otherwise they call the plain dataflow functions.
    //
    // These and the DST lifecycle below are unmeasured: no benchmark in the LLK
    // perf suite isolates a handshake, and none of them touches a circular
    // buffer. What matters about them is not the call anyway, it is the credit
    // they move -- the waiting is derived by the scheduler -- so they carry the
    // unmeasured value of one and let their resource effect do the work.
    t["cb_wait_front"] = {/*dm=*/1, /*unpack=*/1};
    t["cb_pop_front"] = {/*dm=*/1, /*unpack=*/1};
    t["cb_reserve_back"] = {/*dm=*/1, /*unpack=*/{}, /*math=*/{}, /*pack=*/1};
    t["cb_push_back"] = {/*dm=*/1, /*unpack=*/{}, /*math=*/{}, /*pack=*/1};

    // -- DST lifecycle, api/compute/reg_api.h:45-89 ------------------------
    t["tile_regs_acquire"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/1};
    t["tile_regs_commit"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/1};
    t["tile_regs_wait"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{}, /*pack=*/1};
    t["tile_regs_release"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                              /*pack=*/1};

    // -- Eltwise binary, api/compute/eltwise_binary.h ----------------------
    t["binary_op_init_common"] = {/*dm=*/{}, /*unpack=*/100, /*math=*/100,
                                  /*pack=*/140};
    t["add_tiles_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};
    t["add_tiles"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/300};

    // -- Pack, api/compute/pack.h -----------------------------------------
    // pack_tile, lines 86-94: one tile. pack_tile_block, lines 128-135: the
    // same packer work hoisted out of a loop over ntiles. Both forms occur:
    // ttkernel-combine-pack-tiles fuses a run of pack_tile into one
    // pack_tile_block, but only when the CB indices step by one starting from
    // zero, so a subblocked compute keeps separate pack_tile ops for every
    // round after the first.
    t["pack_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{}, /*pack=*/200};
    t["pack_tile_block"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                            /*pack=*/260};

    // -- Data movement, api/dataflow/{noc,dataflow_api,circular_buffer}.h --
    // A barrier's cost is the transfer it waits on, not the call itself.
    t["noc_async_read_tile"] = {/*dm=*/60};
    t["noc_async_write_tile"] = {/*dm=*/60};
    t["noc_async_read_barrier"] = {/*dm=*/30};
    t["noc_async_write_barrier"] = {/*dm=*/30};
    t["get_common_arg_val"] = {/*dm=*/10};
    t["get_write_ptr"] = {/*dm=*/5};
    t["get_read_ptr"] = {/*dm=*/5};
    t["TensorAccessor"] = {/*dm=*/20};

    // -- Known to be free --------------------------------------------------
    // Present on no lane, which is how the table says "costs nothing", as
    // opposed to being absent, which means unknown. get_compile_time_arg_val
    // is a compile-time constant that only constructs a CircularBuffer handle
    // holding an id; TensorAccessorArgs is a template instantiation.
    t["get_compile_time_arg_val"] = {};
    t["TensorAccessorArgs"] = {};
    t["my_logical_x_"] = {};
    t["my_logical_y_"] = {};

    // -- SFPU binary ops that call through a macro -------------------------
    // eltwise_binary_sfpu.h:73 and :214 do not name an llk_ function directly:
    // they go through SFPU_BINARY_CALL / SFPU_BINARY_INIT_FN
    // (llk_math_eltwise_binary_sfpu_macros.h:49, :81). Both expand under
    // MATH(), so the lane is unambiguous either way.
    t["add_binary_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["add_binary_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    // -- Remaining compute-API ops -----------------------------------------
    // Extracted from the non-Quasar branches of api/compute/**.h by matching
    // each op mnemonic to its ALWI wrapper and recording which of UNPACK(),
    // MATH() and PACK() appear in it, resolving one level of delegation
    // (add_tiles_init -> binary_tiles_init).
    //
    // Ops whose wrapper contains mutually exclusive branches are still absent:
    // matmul_block, mm_block_init, mm_block_init_short,
    // pack_reconfig_data_format, tilize_init, transpose_wh_init,
    // transpose_wh_tile, unary_bcast, unary_bcast_init and untilize_uninit.
    // Lane membership is often the same in every arm, so these are now more
    // tractable than they were at call granularity -- but each still needs its
    // header read, and the arms differ in cost even where they agree on lanes,
    // so they report as unknown rather than guessed.

    // -- api/compute/add_int_sfpu.h ------------------------------------------
    t["add_int_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["add_int_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    // -- api/compute/binary_max_min.h ----------------------------------------
    t["binary_max_int32_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["binary_max_int32_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};
    t["binary_max_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["binary_max_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};
    t["binary_min_int32_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["binary_min_int32_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};
    t["binary_min_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["binary_min_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    // -- api/compute/binop_with_scalar.h -------------------------------------
    t["binop_with_scalar_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};
    t["mul_unary_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};

    // -- api/compute/compute_kernel_hw_startup.h -----------------------------
    t["compute_kernel_hw_startup"] = {/*dm=*/{}, /*unpack=*/60, /*math=*/100,
                                      /*pack=*/140};

    // -- api/compute/eltwise_binary.h ----------------------------------------
    t["binary_dest_reuse_tiles"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/300};
    t["binary_dest_reuse_tiles_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};
    t["mul_tiles"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/300};
    t["mul_tiles_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};
    t["sub_tiles"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/300};
    t["sub_tiles_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};

    // -- api/compute/eltwise_binary_sfpu.h -----------------------------------
    t["div_binary_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["div_binary_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};
    t["mul_binary_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["mul_binary_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    // -- api/compute/eltwise_unary.h -----------------------------------------
    t["init_sfpu"] = {/*dm=*/{}, /*unpack=*/100, /*math=*/140, /*pack=*/140};
    t["unary_op_init_common"] = {/*dm=*/{}, /*unpack=*/100, /*math=*/140,
                                 /*pack=*/140};

    // -- api/compute/matmul.h ------------------------------------------------
    t["matmul_tiles"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/400};
    t["mm_init"] = {/*dm=*/{}, /*unpack=*/100, /*math=*/140, /*pack=*/140};
    t["mm_init_short"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};

    // -- api/compute/mul_int_sfpu.h ------------------------------------------
    t["mul_int_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300};
    t["mul_int_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    // -- api/compute/pack.h --------------------------------------------------
    t["pack_reconfig_l1_acc"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                                 /*pack=*/200};

    // -- api/compute/pack_untilize.h -----------------------------------------
    t["pack_untilize_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};
    t["pack_untilize_uninit"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                                 /*pack=*/480};

    // -- api/compute/reduce.h ------------------------------------------------
    t["reduce_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40, /*pack=*/200};
    t["reduce_tile"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/300};
    t["reduce_uninit"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/300, /*pack=*/200};

    // -- api/compute/tile_move_copy.h ----------------------------------------
    t["copy_block_matmul_partials"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/150};
    t["copy_tile"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/150};
    t["copy_tile_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};

    // -- api/compute/tilize.h ------------------------------------------------
    // tilize_block and untilize_block run the DST handshake themselves: their
    // MATH halves call llk_math_wait_for_dest_available and
    // llk_math_dest_section_done, and their PACK halves the packer's matching
    // pair. getResourceEffect keys the DST effects on the tile_regs_* ops only,
    // so those internal acquires are not modelled -- a pre-existing gap that
    // the per-call lists used to make visible and this comment now carries.
    t["tilize_block"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/190, /*pack=*/240};
    t["tilize_uninit"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/{}, /*pack=*/40};

    // -- api/compute/untilize.h ----------------------------------------------
    t["untilize_block"] = {/*dm=*/{}, /*unpack=*/200, /*math=*/40,
                           /*pack=*/240};
    t["untilize_init"] = {/*dm=*/{}, /*unpack=*/40, /*math=*/40};

    // -- api/compute/where.h -------------------------------------------------
    t["where_tile"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/350};
    t["where_tile_init"] = {/*dm=*/{}, /*unpack=*/{}, /*math=*/40};

    return t;
  }();
  return table;
}

/// Cost of one operation on one lane.
///
/// >>> PLACEHOLDER VALUES, NOT MEASUREMENTS. <<<
///
/// Each number is the sum of the placeholder per-call costs this table held
/// when it was keyed on `llk_*` names, so a report keeps the shape it had at
/// call granularity while its magnitudes stay invented. They are ordered
/// sensibly relative to each other -- a semaphore poll is cheaper than a tile
/// of unpack, which is cheaper than a tile of FPU work -- and nothing else.
/// Replace with the measured per-operation, per-lane numbers from
/// scripts/gen_cost_table.py before any decision depends on them.
///
/// Known inaccuracy: a cost here is flat per operation, but some operations
/// scale with a tile count taken from an operand. pack_tile_block in particular
/// packs `ntiles` tiles in a loop, so a flat cost is wrong for any block bigger
/// than one tile. The measured table is keyed on the same operation and lane
/// but carries a per-tile term, so operand-dependent scaling arrives with it.
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
bool lanesAllowSrcCoupling(const ThreadWork &work) {
  return work.unpack && work.math;
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
  const llvm::StringMap<ThreadWork> &table = getThreadWorkTable();
  auto entry = table.find(name);
  assert((!coupled || entry == table.end() ||
          lanesAllowSrcCoupling(entry->second)) &&
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

  out << "cost estimate: scheduled with PLACEHOLDER per-lane costs; every "
         "figure below is cost, not measured cycles\n";

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
    out << "    " << llvm::left_justify("op", nameWidth)
        << "  start    end   cost   wait\n";
    size_t listed =
        std::min<size_t>(laneReport.ops.size(), kMaxListedOpsPerLane);
    for (const PlacedOp &op :
         llvm::ArrayRef<PlacedOp>(laneReport.ops).take_front(listed)) {
      out << "    " << llvm::left_justify(op.name, nameWidth)
          << llvm::format("%7" PRIu64 "%7" PRIu64 "%7" PRIu64 "%7" PRIu64,
                          op.start, op.finish, op.cost, op.stall);
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
    const llvm::StringMap<ThreadWork> &table = getThreadWorkTable();
    {
      llvm::StringRef name = op->getName().getStringRef();
      ++ctx.seen;

      // Messages keep the qualified name so they can be grepped against the IR.
      auto found = table.find(op->getName().stripDialect());
      if (found == table.end()) {
        // With no entry the operation is placed on no lane at all, so its time
        // and its resource effects are both missing.
        failToModel(name, op,
                    "no per lane work for '" + name +
                        "': the operation would be left out of every lane");
        return;
      }
      const ThreadWork &work = found->second;

      auto place = [&](Lane lane, std::optional<uint64_t> cost) {
        if (!cost) {
          return false;
        }
        ResourceEffect effect = getResourceEffect(op, lane);

        PlacedOp placed{name.str(), op->getLoc(), *cost, effect};
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
        placed = place(*dmLane, work.dm);
      } else {
        placed |= place(Lane::Trisc0Unpack, work.unpack);
        placed |= place(Lane::Trisc1Math, work.math);
        placed |= place(Lane::Trisc2Pack, work.pack);
      }

      // The table knows this operation, but not for the kind of kernel it
      // appeared in. Placing nothing would silently report it as free.
      if (!placed && !work.isFree()) {
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
