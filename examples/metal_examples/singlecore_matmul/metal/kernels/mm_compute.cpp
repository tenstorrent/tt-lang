// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"
#include <cstdint>

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
  const uint32_t Mt = get_compile_time_arg_val(0);
  const uint32_t Kt = get_compile_time_arg_val(1);
  const uint32_t Nt = get_compile_time_arg_val(2);
  constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
  constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
  constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

  // Setup the FPU (matrix engine) for the matmul operation. And specify the
  // input and output circular buffers.
  mm_init(cb_in0, cb_in1, cb_out);

  // the simplest possible version of outer product blocked matmul
  // the reader is expected to read the A's and B's tile rows and tile columns
  // for each output tile
  for (uint32_t mt = 0; mt < Mt; ++mt) {
    for (uint32_t nt = 0; nt < Nt; ++nt) {
      // Make sure registers can be used for the output tile. This also sets the
      // registers to zero.
      tile_regs_acquire();
      for (uint32_t kt = 0; kt < Kt; kt++) {
        // Wait for the input tiles to be available in the input circular
        // buffers.
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Perform the matrix multiplication for the current tile.
        // NOTE: This function also accumulates the result into the destination
        // tile.
        matmul_tiles(/*in0_cb_id=*/cb_in0, /*in1_cb_id=*/cb_in1,
                     /*in0_tile_index=*/0, /*in1_tile_index=*/0,
                     /*idst=*/0, /*transpose=*/false);

        // Mark the input tiles as used by popping them from the front of the
        // circular buffers.
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
      }

      // Commit and wait for the registers are populated with the results from
      // the FPU
      tile_regs_commit();
      tile_regs_wait();

      // Ensure the output circular buffer has space for the result tile.
      cb_reserve_back(cb_out, 1);
      // Pack the result tile into the output circular buffer.
      pack_tile(0, cb_out);
      // Mark the output tile as ready so the writer can read it.
      cb_push_back(cb_out, 1);

      // We don't need the registers anymore, so we can release them and prepare
      // for the next output tile.
      tile_regs_release();
    }
  }
}
} // namespace NAMESPACE
