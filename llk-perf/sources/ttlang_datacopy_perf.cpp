// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Isolates the datacopy that tt-metal's `copy_tile` performs: unpack A into
// SrcA (or straight into Dest) followed by a datacopy on MATH.
//
// Also covers `unary_bcast` and `transpose_wh_tile`, because on Blackhole all
// three are the same pair of LLK calls with one knob moved:
//
//   copy_tile          BroadcastType::NONE, A2D, no transpose
//   unary_bcast        BroadcastType::Col/Row, B2D  (bcast.h:94-125)
//   transpose_wh_tile  the identical calls, with the unpacker configured to
//                      transpose in the *init* (transpose.h:107-125 -- its
//                      non-fp32 path issues exactly copy_tile's two calls)
//
// So one source measures three operations, and the init split below measures
// their three inits, rather than adopting two more upstream sources that would
// each need re-deriving on every uplift.
//
// This exists because no other perf source measures that pair on its own. The
// SFPU sources run the same datacopy, but only ahead of an SFPU operation in
// the same MATH loop, so their MATH column covers two ttkernel ops and cannot
// be attributed to either. Worse, they *elide* the datacopy entirely under
// `unpack_to_dest`:
//
//     if constexpr (!unpack_to_dest) { _llk_math_eltwise_unary_datacopy_(...); }
//
// which is not what `copy_tile` does. `copy_tile` always issues both calls and
// passes the mode down to each (api/compute/tile_move_copy.h):
//
//     UNPACK((llk_unpack_A            <..., UnpackToDestEn>(in_cb_id, in_tile_index)));
//     MATH  ((llk_math_eltwise_unary_datacopy<..., UnpackToDestEn>(dst_tile_index, in_cb_id)));
//
// so under `unpack_to_dest` the MATH half becomes synchronization only rather
// than disappearing.
//
// This source therefore calls the datacopy in both modes wherever it can. The
// one place it cannot is MATH_ISOLATE under `unpack_to_dest`: there the call
// enters a two-semaphore handshake with a unpack thread that has already
// returned, and the harness has no fake handshake for those semaphores the way
// it does for the Src banks. See the note at that branch. So the
// `unpack_to_dest` math cost is reachable only through L1_TO_L1, where both
// threads run, and the isolate columns cover `unpack_to_dest = false` -- which
// is the mode tt-lang generates for everything except fp32 with dest_acc.
//
// Operands come from the stimuli buffers (`params.buffer_A`, `params.buffer_Res`)
// rather than the fixed PERF_INPUT_A / PERF_OUTPUT regions. PERF_ADDRESS
// hardcodes a 4096-byte tile stride (helpers/include/perf.h:36), but a tile is
// 4096 bytes only for Float32 -- 2048 for Float16_b, 1088 for Bfp8_b. The stride
// is harmless for correctness in a scratch region, but it makes this kernel walk
// L1 differently from every other perf source, and the unpack lane notices:
// against the SFPU sources at the same N, copy_tile/unpack disagreed by 14-22%.
// The stimuli buffers are laid out at the format's real tile size, which is what
// those sources read.
//
// Structure follows eltwise_binary_fpu_perf.cpp: one kernel per TRISC behind
// the LLK_TRISC_* defines, an INIT zone and a TILE_LOOP zone per thread, and
// DEST blocked at MAX_TILES_DEST with the MATH/PACK handshake outside the
// isolate runs.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT     = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& buffer_A            = params.buffer_A;

    // Runtime rather than template parameters, matching unpack_transpose_perf.cpp.
    // They reach only the init: transpose is a configuration of the unpacker, and
    // the tile loop issues the same call either way -- which is why
    // transpose_wh_tile and copy_tile share a tile-loop body.
    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;
#endif
    // `_llk_unpack_A_*` takes <BType, acc_to_dest, binary_reuse_dest,
    // unpack_to_dest>, all defaulted. Only the last is interesting here, and
    // C++ has no named template arguments, so the first three are spelled out
    // to reach it. They match what copy_tile passes (tile_move_copy.h:112):
    // no broadcast, no accumulation into Dest, no dest reuse.
    //
    // Note `acc_to_dest` is not `is_fp32_dest_acc_en` -- accumulate-into-Dest
    // versus 32-bit Dest. copy_tile passes false; passing the dest-accumulation
    // flag here would silently enable accumulation on every dest_acc variant.
    constexpr bool acc_to_dest                       = false;
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    // The init block splits into a kernel-wide half and an op-specific half.
    // `_llk_unpack_A_init_` is what copy_tile_init issues; the hw_configure is
    // setup every op on this thread pays once. Every existing perf source
    // brackets both as one number, which is why no op-specific init has its own
    // cost anywhere in the data.
    //
    // MEASURE_OP_INIT selects which half sits inside the bracket, so each is
    // measured directly rather than one being inferred by subtracting the
    // other. Both halves run in both variants and hw_configure always runs
    // first, so the hardware sees identical work either way.
    auto hw_configure = [&]()
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    };
    auto op_init = [&]()
    {
        // The (face_r_dim, num_faces) pair became a single TensorShape at the
        // ea042c4ad uplift; make_tensor_shape_from_legacy is what the upstream
        // sources use to bridge it.
        _llk_unpack_A_init_<BROADCAST_TYPE, acc_to_dest, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
            formats.unpack_A_src,
            formats.unpack_A_dst);
    };

    if constexpr (MEASURE_OP_INIT)
    {
        hw_configure();
        {
            START_PERF_MEASURE("INIT")
            op_init();
            PROFILER_SYNC();
        }
    }
    else
    {
        {
            START_PERF_MEASURE("INIT")
            hw_configure();
            PROFILER_SYNC();
        }
        op_init();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Fake the hardware handshake so MATH never waits on us. With
            // unpack_to_dest there are no Src banks to validate: the data is
            // taken to be in Dest already. SrcB is validated only under
            // dest_acc, matching eltwise_unary_sfpu_perf.cpp:73-76.
            if constexpr (!unpack_to_dest)
            {
                // A broadcast leaves the tile in SrcB rather than SrcA, which is
                // why those shapes pair with B2D (bcast.h:32-35). MATH then waits
                // on the SrcB bank, so the fake handshake has to validate it --
                // without this the broadcast shapes hang the device, since
                // MATH_ISOLATE has already returned from the unpack thread.
                constexpr bool kBroadcast = BROADCAST_TYPE != BroadcastType::NONE;
                _perf_unpack_loop_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en || kBroadcast>(
                    TILE_CNT * TILE_NUM_FACES * LOOP_FACTOR);
            }
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, acc_to_dest, reuse_dest_type, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT     = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
#endif
    // DATA_COPY_TYPE comes from the generated build.h; see the test module.
    constexpr DataCopyType data_copy_type = DATA_COPY_TYPE;

    // Same split as the unpack thread. `_llk_math_eltwise_unary_datacopy_init_`
    // is copy_tile_init's math half; pack_sync_init and hw_configure are the
    // kernel-wide setup. MEASURE_OP_INIT picks which sits inside the bracket.
    //
    // Ordering is preserved in both variants: the common setup always runs
    // before the op init, which is the order a real kernel issues them in.
    auto hw_configure = [&]()
    {
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    };
    auto op_init = [&]()
    {
        // BROADCAST_TYPE has to reach the init as well as the execute call. It is
        // the third template parameter and defaults to NONE, so omitting it
        // configures the math engine for a plain datacopy while the tile loop
        // runs a broadcast -- the engine then waits on a SrcB bank it was never
        // told to expect. unary_bcast passes it to both halves for this reason
        // (bcast.h:56-61).
        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE>(
            TILE_NUM_FACES, formats.math);
    };

    if constexpr (MEASURE_OP_INIT)
    {
        hw_configure();
        {
            START_PERF_MEASURE("INIT")
            op_init();
            PROFILER_SYNC();
        }
    }
    else
    {
        {
            START_PERF_MEASURE("INIT")
            hw_configure();
            PROFILER_SYNC();
        }
        op_init();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // MATH performs only the synchronization UNPACK needs to make
            // progress. With unpack_to_dest that acknowledgement is the
            // datacopy itself, which in that mode does no data movement; see
            // the note at the top of this file. Structure and template
            // arguments follow eltwise_unary_sfpu_perf.cpp:138-152.
            for (std::uint32_t tile = 0; tile < TILE_CNT * LOOP_FACTOR; tile++)
            {
                if constexpr (unpack_to_dest)
                {
                    _llk_math_eltwise_unary_datacopy_<data_copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                        tile % MAX_TILES_DEST, formats.math, formats.math);
                }
                else
                {
                    _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(TILE_NUM_FACES);
                }
            }
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    // Not measurable in isolation under unpack_to_dest, so the
                    // call is elided rather than deadlocking. In that mode the
                    // datacopy enters a two-semaphore handshake with the
                    // unpacker -- math_unpack_to_dest_math_ready() spins on
                    // MATH_DONE and math_unpack_to_dest_tile_ready() waits on
                    // UNPACK_TO_DEST (cmath_common.h:208-222) -- and in
                    // MATH_ISOLATE the unpack thread has already returned, so
                    // nothing ever posts them.
                    //
                    // The harness fakes the Src-bank handshake for exactly this
                    // reason (_perf_unpack_loop_set_valid) but has no
                    // equivalent for the unpack-to-dest semaphores, which is
                    // why eltwise_unary_sfpu_perf.cpp:205 elides it here too.
                    // Adding that helper would make this measurable; until then
                    // the unpack_to_dest math cost is only reachable through
                    // L1_TO_L1, where both threads run.
                    if constexpr (!unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_math_eltwise_unary_datacopy_<data_copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                        block_tile, formats.math, formats.math);
                }
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT     = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& buffer_Res          = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_packer_wait_for_math_done_();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
