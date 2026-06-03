// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Single-core compute kernel for the flash-shard cycle A/B: the metal baseline.
// Loops over K chunks calling compute_sdpa_chunk -- the DST-resident online
// softmax (QK^T -> running max/sum rescale -> exp -> PV) from tt-metal's public
// sdpa.h. m/l/o accumulators stay in the dest registers across chunks (the
// *_dst_offset args carve up DST); only the final O and the running max are
// packed out. V is the leading num_tiles_v tiles of each K chunk (MLA coupling).
//
// sdpa.h is referenced in-place via the KernelDescriptor compiler_include_paths.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "compute_kernel_api/sdpa.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_v = get_compile_time_arg_val(7);
    constexpr uint32_t num_tiles_stats = get_compile_time_arg_val(8);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(9);
    static_assert(num_tiles_stats == 1, "num_tiles_stats must be 1");
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;
    constexpr bool exp_approx_mode = false;

    // DST layout: O accumulator at 0, then max/sum (sum shares max's tile, col 2),
    // then the correction-exp scratch, then the QK scores. 8 rows/face * 2 faces.
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * num_tiles_v;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;

    PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
    PACK(SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, true, scale_fp32, true));
    sdpa_custom_mm_block_init<transpose_k>(cb_q, cb_k, cb_out, chunk_size);

    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(SFPU_FPU, 0, 1));

    cb_wait_front(cb_q, num_tiles_k);
    cb_reserve_back(cb_out, num_tiles_v);
    cb_reserve_back(cb_stats, num_tiles_stats);
    tile_regs_acquire();
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        compute_sdpa_chunk<
            chunk_size,
            num_tiles_k,
            num_tiles_v,
            scale_fp32,
            scale_bf16,
            transpose_k,
            transpose_v,
            packed_tile_size,
            exp_approx_mode>(
            cb_q,
            cb_k,
            0,  // cb_mask (unused)
            cb_out,
            mm1_dst_offset,
            mm2_dst_offset,
            max_dst_offset,
            sum_dst_offset,
            corr_exp_dst_offset,
            chunk == 0,
            chunk == num_chunks - 1,
            false /* mask_chunk */);
    }

    // Pack the O accumulator (sem incremented once per 2 tiles -- it caps at 15).
    for (uint32_t i = 0; i < num_tiles_v; i += 2) {
        PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
        pack_tile(mm2_dst_tile_offset + i, cb_out);
        pack_tile(mm2_dst_tile_offset + i + 1, cb_out);
        PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
    }
    // Stall for the reduce-sum to finish, then pack the running max as stats.
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile(max_dst_tile_offset, cb_stats);

    cb_push_back(cb_out, num_tiles_v);
    cb_push_back(cb_stats, num_tiles_stats);
    cb_pop_front(cb_q, num_tiles_k);
    tile_regs_commit();
    tile_regs_release();
    sdpa_custom_mm_block_uninit();
}
