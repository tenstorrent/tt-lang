// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const auto& buffer_A         = params.buffer_A;
    const auto& buffer_B         = params.buffer_B;
#endif

    // Split, but cumulatively rather than disjointly like the math thread -- see
    // the else branch below. The zone is two calls: hw_configure, and the
    // operation's own `llk_unpack_AB_init`, which is all `binary_tiles_init`
    // issues on this thread (eltwise_binary.h:43-45).
    auto hw_configure = [&]()
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            /* num_faces */ 4,
            /* num_faces */ 4);
    };
    auto op_init = [&]()
    {
        _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE);
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
        // Cumulative, not disjoint: `binary_op_init_common` issues both calls on
        // this thread (eltwise_binary.h:484-485), so the whole zone is its unpack
        // half and stays measured. `binary_tiles_init` issues only the second
        // (eltwise_binary.h:43-45), which the branch above brackets alone. Both
        // owners come out of the same pair of runs with nothing left over -- the
        // math zone cannot do this, because there the op-specific call is not
        // part of `binary_op_init_common` at all.
        {
            START_PERF_MEASURE("INIT")
            hw_configure();
            op_init();
            PROFILER_SYNC();
        }
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT * TILE_NUM_FACES);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    // Operands from the StimuliConfig layout, matching
                    // eltwise_unary_sfpu_perf.cpp. PERF_ADDRESS hardcodes a
                    // 4096-byte tile stride (perf.h:36) while a tile is that size
                    // only for Float32, so the two layouts walk L1 differently:
                    // this source read 28.5 cycles/tile on pack against 32.4 from
                    // the three sources already migrated.
                    _llk_unpack_AB_<>(L1_ADDRESS(buffer_A[tile]), L1_ADDRESS(buffer_B[tile]));
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif

    // MEASURE_OP_INIT selects which half of the init the measured zone brackets.
    // Both halves run in both variants and in the same order, so the hardware
    // sees identical work either way and only the window moves. Unsplit, the
    // zone covers the common init and the operation's own together and is
    // attributable to neither -- which is why every `<op>_init` on math is
    // currently a dropped lump.
    auto common_init = [&]()
    {
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    };
    auto op_init = [&]()
    {
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(DEFAULT_TENSOR_SHAPE, 0 /* acc_to_dest */);
    };

    if constexpr (MEASURE_OP_INIT)
    {
        common_init();
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
            common_init();
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
            _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT * TILE_NUM_FACES);
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                            DEFAULT_TENSOR_SHAPE, block_tile, false);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                            DEFAULT_TENSOR_SHAPE, block_tile, false);
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const auto& buffer_Res       = params.buffer_Res;
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
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
        }
        PROFILER_SYNC();
    }
}

#endif
