// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
    const uint32_t num_output_tiles = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);

    // Instead of processing all tiles, we process only the assigned amount of tiles.
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        // Same inner loop as in the single core example, only the outer loop is adjusted
        // to produce the assigned number of tiles.
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
