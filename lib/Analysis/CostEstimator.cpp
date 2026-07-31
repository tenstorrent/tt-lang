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
constexpr llvm::StringLiteral kTTLDialect("ttl");

/// Loop iterations a single kernel may unroll. Exceeding it fails the estimate
/// rather than truncating the program; see placeLoop.
///
/// Counted in iterations, which is not the quantity that matters -- a body of N
/// operations costs trip * N placements, so this bounds the work only loosely.
/// Bounding placements directly is the right fix and is not done yet.
constexpr std::uint64_t kUnrollBudget = 8192;

/// Output caps. A kernel whose loops unroll to tens of thousands of operations
/// would otherwise emit a report longer than anyone will read. Truncation is
/// always announced, never silent.
constexpr size_t kMaxTimelineRows = 200;
constexpr size_t kMaxListedOpsPerLane = 30;

using Lane = CostEstimator::Lane;

/// Functions one TTKernel operation expands to on each RISC.
///
/// `dm` is the expansion when the operation appears in a data-movement kernel.
/// It is a single list rather than one per DM core because the operation does
/// not choose which core it runs on: `ttl.noc_index` on the enclosing function
/// does. The other three are the per-TRISC expansions inside a compute kernel.
/// The `= {}` defaults let an entry name only the lanes it uses; without them
/// -Wmissing-field-initializers rejects the omitted trailing members.
struct ThreadWork {
  llvm::SmallVector<llvm::StringRef, 3> dm = {};
  llvm::SmallVector<llvm::StringRef, 3> unpack = {};
  llvm::SmallVector<llvm::StringRef, 3> math = {};
  llvm::SmallVector<llvm::StringRef, 3> pack = {};

  /// True when the operation is known to cost nothing anywhere. Distinct from
  /// being absent from the table, which means unknown.
  bool isFree() const {
    return dm.empty() && unpack.empty() && math.empty() && pack.empty();
  }
};

/// Thread affinity for TTKernel operations, keyed by operation name.
///
/// A compute kernel is one source file compiled three times, once per TRISC,
/// with -DTRISC_UNPACK / -DTRISC_MATH / -DTRISC_PACK. The UNPACK(), MATH() and
/// PACK() macros in api/compute/common_globals.h keep only the calls belonging
/// to the thread being compiled and erase the rest, so each compute-API call
/// expands to a different number of LLK calls per thread. The counts below are
/// read off the non-Quasar branch of the tt-metal headers:
///
///   circular_buffer.h:31-69 (COMPILE_FOR_TRISC path)
///     wait_front   -> UNPACK llk_wait_tiles
///     pop_front    -> UNPACK llk_pop_tiles
///     reserve_back -> PACK   llk_wait_for_free_tiles
///     push_back    -> PACK   llk_push_tiles
///
///   reg_api.h:45-89
///     tile_regs_acquire -> MATH llk_math_wait_for_dest_available
///     tile_regs_commit  -> MATH llk_math_dest_section_done
///     tile_regs_wait    -> PACK llk_packer_wait_for_math_done
///     tile_regs_release -> PACK llk_pack_dest_section_done
///
///   eltwise_binary.h:31-55  binary_op_init_common
///     UNPACK llk_unpack_hw_configure, llk_unpack_AB_init
///     MATH   llk_math_pack_sync_init, llk_math_hw_configure
///     PACK   llk_pack_hw_configure, llk_pack_init, llk_pack_dest_init
///
///   eltwise_binary.h:72-83, 128-132  add_tiles_init
///     via binary_tiles_init<full_init = true, ELWADD>
///     MATH   llk_math_eltwise_binary_init
///     UNPACK llk_unpack_AB_init   (guarded by `if constexpr (full_init)`)
///
///   eltwise_binary.h:206-214  add_tiles
///     UNPACK llk_unpack_AB
///     MATH   llk_math_eltwise_binary
///
///   pack.h:128-135  pack_tile_block
///     PACK   llk_matmul_pack
///
/// `state_configure()` and `LLK_SAN_FUNCTION()` appear in several of these but
/// are sentinel/sanitizer hooks that compile to nothing in a normal build, so
/// they are not counted.
const llvm::StringMap<ThreadWork> &getThreadWorkTable() {
  static const llvm::StringMap<ThreadWork> table = [] {
    // Keys omit the `ttkernel.` prefix; the lookup strips it. That avoids a
    // string concatenation and allocation per entry at first use.
    llvm::StringMap<ThreadWork> t;

    // Member order is dm, unpack, math, pack; trailing members left off are
    // empty. Written with /*name=*/ comments because designated initializers
    // are C++20 and this builds as C++17.

    // -- Circular buffers, api/dataflow/circular_buffer.h:31-69 -------------
    // Under COMPILE_FOR_TRISC the four methods are wrapped PACK/PACK/UNPACK/
    // UNPACK; otherwise they call the plain dataflow functions.
    t["cb_wait_front"] =
        ThreadWork{/*dm=*/{"cb_wait_front"}, /*unpack=*/{"llk_wait_tiles"}};
    t["cb_pop_front"] =
        ThreadWork{/*dm=*/{"cb_pop_front"}, /*unpack=*/{"llk_pop_tiles"}};
    t["cb_reserve_back"] =
        ThreadWork{/*dm=*/{"cb_reserve_back"}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_wait_for_free_tiles"}};
    t["cb_push_back"] =
        ThreadWork{/*dm=*/{"cb_push_back"}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_push_tiles"}};

    // -- DST lifecycle, api/compute/reg_api.h:45-89 ------------------------
    t["tile_regs_acquire"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{}, /*math=*/{"llk_math_wait_for_dest_available"}};
    t["tile_regs_commit"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{}, /*math=*/{"llk_math_dest_section_done"}};
    t["tile_regs_wait"] =
        ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_packer_wait_for_math_done"}};
    t["tile_regs_release"] =
        ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                   /*pack=*/{"llk_pack_dest_section_done"}};

    // -- Eltwise binary, api/compute/eltwise_binary.h ----------------------
    // binary_op_init_common, lines 31-55.
    t["binary_op_init_common"] = ThreadWork{
        /*dm=*/{},
        /*unpack=*/{"llk_unpack_hw_configure", "llk_unpack_AB_init"},
        /*math=*/{"llk_math_pack_sync_init", "llk_math_hw_configure"},
        /*pack=*/{"llk_pack_hw_configure", "llk_pack_init",
                  "llk_pack_dest_init"}};
    // add_tiles_init, lines 128-132, via binary_tiles_init lines 72-83. It
    // passes full_init = true, so the UNPACK call inside the
    // `if constexpr (full_init)` guard is kept.
    t["add_tiles_init"] =
        ThreadWork{/*dm=*/{}, /*unpack=*/{"llk_unpack_AB_init"},
                   /*math=*/{"llk_math_eltwise_binary_init"}};
    // add_tiles, lines 206-214.
    t["add_tiles"] =
        ThreadWork{/*dm=*/{}, /*unpack=*/{"llk_unpack_AB"},
                   /*math=*/{"llk_math_eltwise_binary"}};

    // -- Pack, api/compute/pack.h -----------------------------------------
    // pack_tile, lines 86-94: one tile.
    t["pack_tile"] = ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                                    /*pack=*/{"llk_pack"}};
    // pack_tile_block, lines 128-135: llk_matmul_pack is llk_pack hoisted out of
    // a loop over ntiles; the name is vestigial and pack_tile_block is its only
    // caller. Both forms occur: ttkernel-combine-pack-tiles fuses a run of
    // pack_tile into one pack_tile_block, but only when the CB indices step by
    // one starting from zero, so a subblocked compute keeps separate pack_tile
    // ops for every round after the first.
    t["pack_tile_block"] = ThreadWork{/*dm=*/{}, /*unpack=*/{}, /*math=*/{},
                                          /*pack=*/{"llk_matmul_pack"}};

    // -- Data movement, api/dataflow/{noc,dataflow_api,circular_buffer}.h --
    // A barrier's cost is the transfer it waits on, not the call itself.
    t["noc_async_read_tile"] = ThreadWork{/*dm=*/{"Noc::async_read"}};
    t["noc_async_write_tile"] = ThreadWork{/*dm=*/{"Noc::async_write"}};
    t["noc_async_read_barrier"] =
        ThreadWork{/*dm=*/{"Noc::async_read_barrier"}};
    t["noc_async_write_barrier"] =
        ThreadWork{/*dm=*/{"Noc::async_write_barrier"}};
    t["get_common_arg_val"] = ThreadWork{/*dm=*/{"get_common_arg_val"}};
    t["get_write_ptr"] =
        ThreadWork{/*dm=*/{"CircularBuffer::get_write_ptr"}};
    t["get_read_ptr"] = ThreadWork{/*dm=*/{"CircularBuffer::get_read_ptr"}};
    t["TensorAccessor"] =
        ThreadWork{/*dm=*/{"TensorAccessor::TensorAccessor"}};

    // -- Known to be free --------------------------------------------------
    // Present with no calls, which is how the table says "costs nothing", as
    // opposed to being absent, which means unknown. get_compile_time_arg_val
    // is a compile-time constant that only constructs a CircularBuffer handle
    // holding an id; TensorAccessorArgs is a template instantiation.
    t["get_compile_time_arg_val"] = {};
    t["TensorAccessorArgs"] = {};


    // -- SFPU binary ops that call through a macro -------------------------
    // eltwise_binary_sfpu.h:73 and :214 do not call an llk_ function directly:
    // they go through SFPU_BINARY_CALL / SFPU_BINARY_INIT_FN, which expand to
    // _llk_math_eltwise_binary_sfpu_params_ and
    // llk_math_eltwise_binary_sfpu_init respectively
    // (llk_math_eltwise_binary_sfpu_macros.h:49, :81). The `_sfpu_binary_check_`
    // that SFPU_BINARY_CALL also emits is a validation helper and is omitted for
    // the same reason as LLK_SAN_FUNCTION.
    t["add_binary_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"_llk_math_eltwise_binary_sfpu_params_"}, /*pack=*/{}};
    t["add_binary_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_init"}, /*pack=*/{}};

    // -- Remaining compute-API ops -----------------------------------------
    // Extracted from the non-Quasar branches of api/compute/**.h by matching
    // each op mnemonic to its ALWI wrapper and collecting the UNPACK()/MATH()/
    // PACK() calls in order, resolving one level of delegation (add_tiles_init
    // -> binary_tiles_init).
    //
    // Ops whose wrapper contains mutually exclusive branches are deliberately
    // absent: collecting every macro in such a body double-counts arms that
    // never both run. matmul_block, mm_block_init, mm_block_init_short,
    // pack_reconfig_data_format, tilize_init, transpose_wh_init,
    // transpose_wh_tile, unary_bcast, unary_bcast_init and untilize_uninit need
    // reading by hand, and until then report as unknown rather than inflated.
    // -- api/compute/add_int_sfpu.h ------------------------------------------
    t["add_int_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_add_int"}, /*pack=*/{}};
    t["add_int_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_add_int_init"}, /*pack=*/{}};

    // -- api/compute/binary_max_min.h ----------------------------------------
    t["binary_max_int32_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max_int32"}, /*pack=*/{}};
    t["binary_max_int32_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max_min_int32_init"}, /*pack=*/{}};
    t["binary_max_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max"}, /*pack=*/{}};
    t["binary_max_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max_min_init"}, /*pack=*/{}};
    t["binary_min_int32_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_min_int32"}, /*pack=*/{}};
    t["binary_min_int32_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max_min_int32_init"}, /*pack=*/{}};
    t["binary_min_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_min"}, /*pack=*/{}};
    t["binary_min_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binary_max_min_init"}, /*pack=*/{}};

    // -- api/compute/binop_with_scalar.h -------------------------------------
    t["binop_with_scalar_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_unary_sfpu_binop_with_scalar_init"}, /*pack=*/{}};
    t["mul_unary_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_unary_sfpu_binop_with_scalar"}, /*pack=*/{}};

    // -- api/compute/compute_kernel_hw_startup.h -----------------------------
    t["compute_kernel_hw_startup"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_hw_configure"},
        /*math=*/{"llk_math_pack_sync_init", "llk_math_hw_configure"}, /*pack=*/{"llk_pack_hw_configure", "llk_pack_init", "llk_pack_dest_init"}};

    // -- api/compute/eltwise_binary.h ----------------------------------------
    t["binary_dest_reuse_tiles"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A"},
        /*math=*/{"llk_math_eltwise_binary"}, /*pack=*/{}};
    t["binary_dest_reuse_tiles_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A_init"},
        /*math=*/{"llk_math_eltwise_binary_init"}, /*pack=*/{}};
    t["mul_tiles"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB"},
        /*math=*/{"llk_math_eltwise_binary"}, /*pack=*/{}};
    t["mul_tiles_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_init"},
        /*math=*/{"llk_math_eltwise_binary_init"}, /*pack=*/{}};
    t["sub_tiles"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB"},
        /*math=*/{"llk_math_eltwise_binary"}, /*pack=*/{}};
    t["sub_tiles_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_init"},
        /*math=*/{"llk_math_eltwise_binary_init"}, /*pack=*/{}};

    // -- api/compute/eltwise_binary_sfpu.h -----------------------------------
    t["div_binary_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binop_div"}, /*pack=*/{}};
    t["div_binary_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binop_init"}, /*pack=*/{}};
    t["mul_binary_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binop_mul"}, /*pack=*/{}};
    t["mul_binary_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_binop_init"}, /*pack=*/{}};

    // -- api/compute/eltwise_unary.h -----------------------------------------
    t["init_sfpu"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_hw_configure", "llk_unpack_A_init"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_init", "llk_math_pack_sync_init", "llk_math_hw_configure"}, /*pack=*/{"llk_pack_hw_configure", "llk_pack_init", "llk_pack_dest_init"}};
    t["unary_op_init_common"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_hw_configure", "llk_unpack_A_init"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_init", "llk_math_pack_sync_init", "llk_math_hw_configure"}, /*pack=*/{"llk_pack_hw_configure", "llk_pack_init", "llk_pack_dest_init"}};

    // -- api/compute/matmul.h ------------------------------------------------
    t["matmul_tiles"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_matmul"},
        /*math=*/{"llk_math_matmul"}, /*pack=*/{}};
    t["mm_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_hw_configure", "llk_unpack_AB_matmul_init"},
        /*math=*/{"llk_math_hw_configure", "llk_math_pack_sync_init", "llk_math_matmul_init"}, /*pack=*/{"llk_pack_hw_configure", "llk_pack_dest_init", "llk_pack_init"}};
    t["mm_init_short"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_matmul_init"},
        /*math=*/{"llk_math_matmul_init"}, /*pack=*/{}};

    // -- api/compute/mul_int_sfpu.h ------------------------------------------
    t["mul_int_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_mul_int"}, /*pack=*/{}};
    t["mul_int_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_binary_sfpu_mul_int_init"}, /*pack=*/{}};

    // -- api/compute/pack.h --------------------------------------------------
    t["pack_reconfig_l1_acc"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{}, /*pack=*/{"llk_pack_reconfig_l1_acc"}};

    // -- api/compute/pack_untilize.h -----------------------------------------
    t["pack_untilize_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A_init"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_init"}, /*pack=*/{}};
    t["pack_untilize_uninit"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{}, /*pack=*/{"llk_pack_untilize_uninit", "llk_init_packer_dest_offset_registers", "llk_pack_reconfig_data_format", "llk_pack_init"}};

    // -- api/compute/reduce.h ------------------------------------------------
    t["reduce_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_reduce_init"},
        /*math=*/{"llk_math_reduce_init"}, /*pack=*/{"llk_pack_reduce_mask_config"}};
    t["reduce_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_AB_reduce"},
        /*math=*/{"llk_math_reduce"}, /*pack=*/{}};
    t["reduce_uninit"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_reduce_uninit"}, /*pack=*/{"llk_pack_reduce_mask_clear"}};

    // -- api/compute/tile_move_copy.h ----------------------------------------
    t["copy_block_matmul_partials"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A_block"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_block"}, /*pack=*/{}};
    t["copy_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A"},
        /*math=*/{"llk_math_eltwise_unary_datacopy"}, /*pack=*/{}};
    t["copy_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_A_init"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_init"}, /*pack=*/{}};

    // -- api/compute/tilize.h ------------------------------------------------
    t["tilize_block"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_tilize_block"},
        /*math=*/{"llk_math_wait_for_dest_available", "llk_math_eltwise_unary_datacopy", "llk_math_dest_section_done"}, /*pack=*/{"llk_packer_wait_for_math_done", "llk_pack", "llk_pack_dest_section_done"}};
    t["tilize_uninit"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_tilize_uninit"},
        /*math=*/{}, /*pack=*/{"llk_pack_init"}};

    // -- api/compute/untilize.h ----------------------------------------------
    t["untilize_block"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_untilize"},
        /*math=*/{"llk_math_wait_for_dest_available", "llk_math_dest_section_done"}, /*pack=*/{"llk_packer_wait_for_math_done", "llk_pack", "llk_pack_dest_section_done"}};
    t["untilize_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{"llk_unpack_untilize_init"},
        /*math=*/{"llk_math_eltwise_unary_datacopy_init"}, /*pack=*/{}};

    // -- api/compute/where.h -------------------------------------------------
    t["where_tile"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_ternary_sfpu_where"}, /*pack=*/{}};
    t["where_tile_init"] = ThreadWork{
        /*dm=*/{}, /*unpack=*/{},
        /*math=*/{"llk_math_eltwise_ternary_sfpu_where_init"}, /*pack=*/{}};

    return t;
  }();
  return table;
}

/// Cycle cost of one emitted call.
///
/// >>> PLACEHOLDER VALUES, NOT MEASUREMENTS. <<<
///
/// They are ordered sensibly relative to each other -- a semaphore poll is
/// cheaper than a tile of unpack, which is cheaper than a tile of FPU work --
/// so the shape of a report is meaningful while its magnitudes are not. Replace
/// with per-call measurements before any decision depends on the numbers.
///
/// Known inaccuracy: costs are flat per call, but some calls scale with a tile
/// count taken from an operand. llk_matmul_pack in particular is llk_pack in a
/// loop over `ntiles`, so a flat cost is wrong for any block bigger than one
/// tile. Operand-dependent scaling is the next refinement.
const llvm::StringMap<uint64_t> &getCallCostTable() {
  static const llvm::StringMap<uint64_t> table = [] {
    llvm::StringMap<uint64_t> t;

    // Circular-buffer semaphore ops, uncontended. The cost of waiting is the
    // simulator's business; this is just the call.
    t["llk_wait_tiles"] = 20;
    t["llk_pop_tiles"] = 20;
    t["llk_wait_for_free_tiles"] = 20;
    t["llk_push_tiles"] = 20;
    t["cb_wait_front"] = 30;
    t["cb_pop_front"] = 30;
    t["cb_reserve_back"] = 30;
    t["cb_push_back"] = 30;

    // DST handshake.
    t["llk_math_wait_for_dest_available"] = 20;
    t["llk_math_dest_section_done"] = 20;
    t["llk_packer_wait_for_math_done"] = 20;
    t["llk_pack_dest_section_done"] = 20;

    // Config writes, paid per block because the inits sit inside the loop.
    t["llk_unpack_hw_configure"] = 60;
    t["llk_unpack_AB_init"] = 40;
    t["llk_math_pack_sync_init"] = 40;
    t["llk_math_hw_configure"] = 60;
    t["llk_math_eltwise_binary_init"] = 40;
    t["llk_pack_hw_configure"] = 60;
    t["llk_pack_init"] = 40;
    t["llk_pack_dest_init"] = 40;

    // Real per-tile work.
    t["llk_unpack_AB"] = 200;
    t["llk_math_eltwise_binary"] = 300;
    t["llk_pack"] = 200;
    t["llk_matmul_pack"] = 260;

    // Data movement. A barrier's real cost is the transfer it waits on, which
    // the simulator has to model from bytes and bandwidth; this is the call.
    t["Noc::async_read"] = 60;
    t["Noc::async_write"] = 60;
    t["Noc::async_read_barrier"] = 30;
    t["Noc::async_write_barrier"] = 30;
    t["get_common_arg_val"] = 10;
    t["CircularBuffer::get_write_ptr"] = 5;
    t["CircularBuffer::get_read_ptr"] = 5;
    t["TensorAccessor::TensorAccessor"] = 20;


    // Class-level placeholders for the extracted compute ops. Grouped by class
    // because the value is invented either way, and the class is the unit that
    // real per-op-per-RISC measurements arrive in.
    // datacopy
    t["llk_math_eltwise_unary_datacopy"] = 150;
    t["llk_math_eltwise_unary_datacopy_block"] = 150;
    // init
    t["llk_init_packer_dest_offset_registers"] = 40;
    t["llk_math_eltwise_binary_sfpu_add_int_init"] = 40;
    t["llk_math_eltwise_binary_sfpu_binary_max_min_init"] = 40;
    t["llk_math_eltwise_binary_sfpu_binary_max_min_int32_init"] = 40;
    t["llk_math_eltwise_binary_sfpu_binop_init"] = 40;
    t["llk_math_eltwise_binary_sfpu_mul_int_init"] = 40;
    t["llk_math_eltwise_ternary_sfpu_where_init"] = 40;
    t["llk_math_eltwise_unary_datacopy_init"] = 40;
    t["llk_math_eltwise_unary_sfpu_binop_with_scalar_init"] = 40;
    t["llk_math_matmul_init"] = 40;
    t["llk_math_reduce_init"] = 40;
    t["llk_unpack_AB_matmul_init"] = 40;
    t["llk_unpack_AB_reduce_init"] = 40;
    t["llk_unpack_A_init"] = 40;
    t["llk_unpack_untilize_init"] = 40;
    // matmul math
    t["llk_math_matmul"] = 400;
    // pack
    t["llk_pack_reconfig_data_format"] = 200;
    t["llk_pack_reconfig_l1_acc"] = 200;
    t["llk_pack_reduce_mask_clear"] = 200;
    t["llk_pack_reduce_mask_config"] = 200;
    t["llk_pack_untilize_uninit"] = 200;
    // reduce math
    t["llk_math_reduce"] = 300;
    t["llk_math_reduce_uninit"] = 300;
    t["_llk_math_eltwise_binary_sfpu_params_"] = 300;
    t["llk_math_eltwise_binary_sfpu_init"] = 40;

    // sfpu binary
    t["llk_math_eltwise_binary_sfpu_add_int"] = 300;
    t["llk_math_eltwise_binary_sfpu_binary_max"] = 300;
    t["llk_math_eltwise_binary_sfpu_binary_max_int32"] = 300;
    t["llk_math_eltwise_binary_sfpu_binary_min"] = 300;
    t["llk_math_eltwise_binary_sfpu_binary_min_int32"] = 300;
    t["llk_math_eltwise_binary_sfpu_binop_div"] = 300;
    t["llk_math_eltwise_binary_sfpu_binop_mul"] = 300;
    t["llk_math_eltwise_binary_sfpu_mul_int"] = 300;
    // sfpu ternary
    t["llk_math_eltwise_ternary_sfpu_where"] = 350;
    // sfpu unary
    t["llk_math_eltwise_unary_sfpu_binop_with_scalar"] = 300;
    // unpack
    t["llk_unpack_A"] = 200;
    t["llk_unpack_AB_matmul"] = 200;
    t["llk_unpack_AB_reduce"] = 200;
    t["llk_unpack_A_block"] = 200;
    t["llk_unpack_tilize_block"] = 200;
    t["llk_unpack_tilize_uninit"] = 200;
    t["llk_unpack_untilize"] = 200;

    return t;
  }();
  return table;
}

using ResourceEffect = CostEstimator::ResourceEffect;

/// Resource effect of one TTKernel operation, read from its operands.
///
/// Nothing here is hand-maintained data: the buffer identity, the tile count and
/// the capacity all come from the module. Only the op-name-to-kind mapping is
/// fixed, and that follows from the operation's own semantics.
/// Operations that unpack into SrcA/SrcB without carrying
/// `TTKernelFPUOpTrait`. The trait covers the six FPU ops; these are the
/// datacopy and (un)tilize paths, which feed Src exactly the same way but are
/// not FPU operations. Missing one means MATH is free to run before the data
/// exists, so the assert below cross-checks this list against each operation's
/// own expansion.
const llvm::StringSet<> &getNonFpuSrcCouplingOps() {
  static const llvm::StringSet<> ops = {
      "copy_tile", "copy_block_matmul_partials", "tilize_block",
      "untilize_block"};
  return ops;
}

/// Whether the operation's own expansion implies a Src handshake: its UNPACK
/// half moves data (as opposed to only configuring the unpacker with `*_init` or
/// `*_hw_configure`) and MATH consumes it. Used only to validate the list above.
bool expansionImpliesSrcCoupling(const ThreadWork &work) {
  if (work.math.empty()) {
    return false;
  }
  for (llvm::StringRef call : work.unpack) {
    if (!call.ends_with("_init") && !call.contains("hw_configure")) {
      return true;
    }
  }
  return false;
}

ResourceEffect getResourceEffect(Operation *op, Lane lane) {
  llvm::StringRef name = op->getName().stripDialect();

  // The Src handshake is the one effect that depends on which lane the placement
  // is for: the UNPACK half fills a bank and the MATH half drains one. Every
  // other effect below lands on exactly one lane, so they ignore `lane`.
  bool coupled = op->hasTrait<ttkernel::TTKernelFPUOpTrait>() ||
                 getNonFpuSrcCouplingOps().contains(name);
  const llvm::StringMap<ThreadWork> &table = getThreadWorkTable();
  auto entry = table.find(name);
  assert((entry == table.end() ||
          coupled == expansionImpliesSrcCoupling(entry->second)) &&
         "Src coupling disagrees with the operation's expansion: an op that "
         "unpacks into Src needs TTKernelFPUOpTrait or an entry in "
         "getNonFpuSrcCouplingOps");
  if (coupled) {
    if (lane == Lane::Trisc0Unpack) {
      return {ResourceEffect::Kind::SrcProduce, 0, 0};
    }
    if (lane == Lane::Trisc1Math) {
      return {ResourceEffect::Kind::SrcConsume, 0, 0};
    }
    return {};
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
  auto argVal =
      op->getOperand(0).getDefiningOp<ttkernel::GetCompileArgValOp>();
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
/// name without its dialect prefix, so an unlisted operation renders wide rather
/// than wrong. Mechanical truncation is not an option because `tile_regs_*`
/// would collide.
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
  static constexpr Lane lanes[kNumLanes] = {
      Lane::Ncrisc, Lane::Trisc0Unpack, Lane::Trisc1Math, Lane::Trisc2Pack,
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

  out << "cost estimate: scheduled with PLACEHOLDER call costs\n";

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
    uint64_t work = 0;
    for (const PlacedOp &op : laneReport.ops) {
      work += op.cycles;
    }
    if (work > busiestWork) {
      busiestWork = work;
      busiestLane = lane;
    }
    // Split the non-busy time. "idle" alone conflates two situations that call
    // for opposite responses: a lane blocked on another lane is a dependency
    // signal, while a lane that has run out of work is a balance observation.
    //
    // The three partition totalCycles exactly, because a lane's own span is
    // busy + stalled: each operation contributes its stall gap then its cost,
    // telescoping to the last finish.
    uint64_t stalled = 0;
    for (const PlacedOp &op : laneReport.ops) {
      stalled += op.stall;
    }
    uint64_t lastFinish =
        laneReport.ops.empty() ? 0 : laneReport.ops.back().finish;
    uint64_t drained = totalCycles > lastFinish ? totalCycles - lastFinish : 0;
    assert((laneReport.ops.empty() ||
            work + stalled + drained == totalCycles) &&
           "lane time must account for the whole run");

    out << "  " << getLaneName(lane) << ": " << laneReport.ops.size()
        << " ops, " << laneReport.llkCalls() << " llk calls, " << work
        << " busy, " << stalled << " stalled, " << drained << " drained";
    if (totalCycles > 0) {
      out << llvm::format(", %.0f%% utilized", 100.0 * work / totalCycles);
    }
    out << "\n";
  }

  out << "  latency: " << totalCycles << " cycles\n";
  out << "  busiest lane: " << getLaneName(busiestLane) << " (" << busiestWork
      << " cycles busy)\n";

  if (isComplete()) {
    out << "  complete: every operation was placed\n";
    return text;
  }

  out << "  incomplete: " << unknowns.size() << " unaccounted\n";
  for (const Unknown &unknown : unknowns) {
    out << "    " << unknown.message << "\n";
  }
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
        << "  start    end   cost   wait  calls\n";
    size_t listed =
        std::min<size_t>(laneReport.ops.size(), kMaxListedOpsPerLane);
    for (const PlacedOp &op :
         llvm::ArrayRef<PlacedOp>(laneReport.ops).take_front(listed)) {
      out << "    " << llvm::left_justify(op.name, nameWidth)
          << llvm::format("%7" PRIu64 "%7" PRIu64 "%7" PRIu64 "%7" PRIu64 "  ",
                          op.start, op.finish, op.cycles, op.stall);
      llvm::interleave(op.calls, out, ", ");
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
  if (totalCycles == 0) {
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
  out << llvm::right_justify("cycle", 8) << llvm::right_justify("gap", 7)
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
        << " later event rows omitted; use timeline-step=N for a sampled view\n";
  }
  return text;
}

std::string CostEstimator::Report::renderTimelineFixed(uint64_t step) const {
  if (totalCycles == 0 || step == 0) {
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
      << " cycles, '|' running, 'w' waiting, '.' idle";
  out << " (anything shorter than " << step << " cycles may be hidden)\n\n";
  out << llvm::right_justify("cycle", 8) << "  ";
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
  for (uint64_t now = 0; now <= totalCycles; now += step) {
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
struct CbState {
  uint64_t capacity = 0;
  uint64_t available = 0; ///< published by the producer, not yet popped
  uint64_t reserved = 0;  ///< claimed by the producer, not yet pushed

  uint64_t freeTiles() const { return capacity - available - reserved; }
};

/// DST halves. `SyncHalf` gives two, so MATH can fill one while PACK drains the
/// other; `dst_full_sync_en` gives one and the two serialize.
struct DstState {
  unsigned freeHalves = 2;
  unsigned committed = 0;
};

/// SrcA/SrcB banks between the unpacker and the Matrix Unit.
///
/// The unpacker sets dvalid on the bank it filled and the math MOP's end op
/// clears it, so this is a credit counter like the others and the ping-pong
/// between banks is emergent.
///
/// Simplification: SrcA and SrcB are counted as one credit rather than two
/// independent register files. That is exact for the AB-style ops that read both
/// and conservative for single-operand ops, which leave the other file idle.
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
    // Reject IR from before convert-ttl-to-ttkernel: the per-thread operation
    // sequence does not exist yet, so any estimate would be of the wrong
    // program.
    WalkResult preLowering = module.walk([](Operation *op) {
      if (op->getName().getDialectNamespace() == kTTLDialect &&
          op->getName().stripDialect() == "compute") {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (preLowering.wasInterrupted()) {
      return module.emitError()
             << "cost estimator needs TTKernel IR, but the module still "
                "contains ttl.compute; run it after convert-ttl-to-ttkernel";
    }

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
        if (auto fullSync = funcOp->getAttrOfType<BoolAttr>(
                ttl::kDstFullSyncEnAttrName)) {
          dstHalves = fullSync.getValue() ? 1 : 2;
        }
        ttkernelOps += placeFunc(funcOp, std::nullopt, report);
        break;
      default:
        // Structural: an unhandled thread type means the whole function's work
        // lands nowhere.
        cannotModel = true;
        funcOp->emitError() << "cost estimator does not model the kernel thread "
                               "type on '"
                            << funcOp.getSymName() << "'";
        break;
      }
    }

    // Reject IR from after convert-ttkernel-to-emitc: circular-buffer calls
    // have become opaque verbatim strings by then, so an empty walk here means
    // the module is past the stage this estimator reads, not that it is
    // trivial.
    if (ttkernelOps == 0 && !report.kernels.empty()) {
      return module.emitError()
             << "cost estimator found no ttkernel operations in "
             << report.kernels.size()
             << " kernel function(s); the module is probably already lowered "
                "to EmitC";
    }

    if (cannotModel) {
      // The diagnostic was already emitted at the offending loop.
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
    dst.freeHalves = dstHalves;

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
        return cbs[effect.cb].available >= effect.tiles ? llvm::StringRef()
                                                        : "cb published tiles";
      case ResourceEffect::Kind::DstAcquire:
        return dst.freeHalves > 0 ? llvm::StringRef() : "dst half";
      case ResourceEffect::Kind::DstWait:
        return dst.committed > 0 ? llvm::StringRef() : "dst commit";
      case ResourceEffect::Kind::SrcProduce:
        return src.freeBanks > 0 ? llvm::StringRef() : "srcA/B bank";
      case ResourceEffect::Kind::SrcConsume:
        return src.valid > 0 ? llvm::StringRef() : "srcA/B dvalid";
      default:
        return llvm::StringRef();
      }
    };

    auto takeAtStart = [&](const PlacedOp &op) {
      const ResourceEffect &effect = op.effect;
      if (effect.kind == ResourceEffect::Kind::CbReserve) {
        cbs[effect.cb].reserved += effect.tiles;
      } else if (effect.kind == ResourceEffect::Kind::DstAcquire) {
        --dst.freeHalves;
      } else if (effect.kind == ResourceEffect::Kind::SrcProduce) {
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
        cb.reserved -= std::min(cb.reserved, effect.tiles);
        cb.available += effect.tiles;
        break;
      case ResourceEffect::Kind::CbPop:
        cb.available -= std::min(cb.available, effect.tiles);
        break;
      case ResourceEffect::Kind::DstCommit:
        ++dst.committed;
        break;
      case ResourceEffect::Kind::DstRelease:
        ++dst.freeHalves;
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
        op.finish = now + std::max<uint64_t>(op.cycles, 1);
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
        return module.emitError()
               << "cost estimator deadlocked at cycle " << now << detail;
      }

      uint64_t next = std::numeric_limits<uint64_t>::max();
      for (Lane lane : getAllLanes()) {
        const LaneSim &laneSim = sim[getLaneIndex(lane)];
        if (laneSim.inFlight) {
          next = std::min(next, laneSim.busyUntil);
        }
      }
      now = next;
      report.totalCycles = std::max(report.totalCycles, now);
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
    llvm::StringSet<> reportedNames;
    uint64_t seen = 0;
  };

  /// One unknown per distinct key, so a repeated loop body does not repeat the
  /// same complaint once per iteration.
  void reportOnce(PlaceContext &ctx, llvm::StringRef key, std::string message,
                  Location loc) {
    if (ctx.reportedNames.insert(key).second) {
      ctx.report->unknowns.push_back({std::move(message), loc});
    }
  }

  /// Record a structural gap: diagnose it once per distinct `key` so one run
  /// names everything that is missing, and mark the estimate unusable.
  void failToModel(PlaceContext &ctx, llvm::StringRef key, Operation *op,
                   const llvm::Twine &message) {
    if (ctx.reportedNames.insert(key).second) {
      op->emitError() << message;
    }
    cannotModel = true;
  }

  uint64_t placeFunc(func::FuncOp funcOp, std::optional<Lane> dmLane,
                     Report &report) {
    PlaceContext ctx;
    ctx.dmLane = dmLane;
    ctx.report = &report;

    // Walk regions structurally rather than with Operation::walk, because a loop
    // body has to be repeated in place: the correct lane order for a body of
    // A,B,C over two iterations is A,B,C,A,B,C, not A,A,B,B,C,C.
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
      failToModel(ctx, op->getName().getStringRef(), op,
                  "cost estimator does not model '" +
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
      // TODO(ttl): a dynamic trip count cannot be unrolled at all, so this needs
      // steady-state extrapolation rather than a larger budget: unroll a bounded
      // prefix, detect when the resource state and per-lane program counters
      // repeat, take that cycle delta as the initiation interval, and report
      // prologue + II * (trip - warmup) + epilogue with the trip count left
      // symbolic. Until then, failing is the honest answer -- placing nothing
      // for the loop would schedule a different program.
      if (!cannotModel) {
        loop->emitError()
            << "cost estimator cannot determine this loop's trip count "
               "statically, and steady-state extrapolation is not implemented "
               "yet, so no estimate is produced";
      }
      cannotModel = true;
      return;
    }
    // Over budget is a hard failure, not an unknown. Skipping the body would
    // leave the lanes holding a different program than the one being estimated,
    // and scheduling that still yields a confident-looking latency which is
    // neither an upper nor a lower bound on the real one. An unknown is the
    // right signal for a missing cost; it is too weak for a missing program.
    if (!budget.canConsume(*trip)) {
      if (!cannotModel) {
        loop->emitError() << "cost estimator cannot unroll a loop of " << *trip
                          << " iterations: the per-kernel unroll budget is "
                          << kUnrollBudget
                          << " iterations. Steady-state extrapolation is not "
                             "implemented yet, so no estimate is produced.";
      }
      cannotModel = true;
      return;
    }

    LogicalResult enumerated = enumerateLoopNest(
        {loop}, bindings, budget,
        [&](const LoopInductionBindings &) -> LogicalResult {
          for (Region &region : loop->getRegions()) {
            for (Block &block : region) {
              placeBlock(block, ctx, bindings, budget);
            }
          }
          return success();
        });
    if (failed(enumerated)) {
      // Also a hard failure, and worse than the case above: enumeration stopped
      // partway, so some iterations are already placed and the lanes hold a
      // truncated program.
      if (!cannotModel) {
        loop->emitError() << "cost estimator exhausted its unroll budget of "
                          << kUnrollBudget
                          << " iterations partway through this loop, or hit an "
                             "iteration range it cannot enumerate";
      }
      cannotModel = true;
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
        // One unknown per distinct name keeps the report readable while still
        // naming everything the affinity table is missing.
        // Structural: with no entry the operation is placed on no lane at all,
        // so its time and its resource effects are both missing.
        failToModel(ctx, name, op,
                    "no thread affinity for '" + name +
                        "': the operation would be left out of every lane");
        return;
      }
      const ThreadWork &work = found->second;

      const llvm::StringMap<uint64_t> &costs = getCallCostTable();
      auto place = [&](Lane lane, llvm::ArrayRef<llvm::StringRef> calls) {
        if (calls.empty()) {
          return false;
        }
        uint64_t cycles = 0;
        for (llvm::StringRef call : calls) {
          auto cost = costs.find(call);
          if (cost == costs.end()) {
            reportOnce(ctx, call, ("no cost for call '" + call + "'").str(),
                       op->getLoc());
            continue;
          }
          cycles += cost->second;
        }
        ResourceEffect effect = getResourceEffect(op, lane);

        llvm::SmallVector<llvm::StringRef> callList(calls.begin(), calls.end());
        PlacedOp placed{name.str(), op->getLoc(), std::move(callList), cycles,
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
        placed = place(*dmLane, work.dm);
      } else {
        placed |= place(Lane::Trisc0Unpack, work.unpack);
        placed |= place(Lane::Trisc1Math, work.math);
        placed |= place(Lane::Trisc2Pack, work.pack);
      }

      // The table knows this operation, but not for the kind of kernel it
      // appeared in. Placing nothing would silently report it as free.
      if (!placed && !work.isFree()) {
        failToModel(ctx, name, op,
                    "'" + name + "' has no expansion for a " +
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

  /// Set when the module contains something whose *structure* cannot be
  /// reproduced: an operation that will not be placed, control flow whose
  /// outcome is unknown, or two kernels sharing a lane. The placed lanes then
  /// describe a different program than the real one, so estimate() fails rather
  /// than reporting a latency for it.
  ///
  /// Distinct from Report::unknowns, which covers *cost* gaps: there the program
  /// is right and only a magnitude is missing, so a partial answer is still
  /// worth having.
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
