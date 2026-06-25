// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone FPU block-transpose: Y = transpose(X). X is (R, C) tiles row-major,
// Y is (C, R) tiles. Each input tile (i, j) is transpose_wh'd (the 32x32 content
// is transposed on the FPU) and packed to the transposed grid position (j, i).
//
// Subblocked by a configurable (sub_h x sub_w) chunk = the DST tiles held per
// tile_regs_acquire. The transpose has strictly 1:1 tile mapping (zero operand
// reuse), so the subblock is *purely* the per-acquire DST chunk -- the same lever
// MB1's pack/unpack probe isolates, but with real FPU math in the loop.
//
// The measured zone repeats the whole transpose `iters` times over a resident
// input (the CB reserve/wait/push/pop hop is outside the zone), so trisc_max
// reflects steady-state transpose + per-acquire overhead, not one-shot dispatch.
//
// Requires sub_h | R and sub_w | C and sub_h*sub_w <= DST capacity.
// Compile-time args: 0 = R, 1 = C, 2 = iters, 3 = sub_h, 4 = sub_w.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t R = get_compile_time_arg_val(0);
  const uint32_t C = get_compile_time_arg_val(1);
  const uint32_t iters = get_compile_time_arg_val(2);
  const uint32_t sh = get_compile_time_arg_val(3);
  const uint32_t sw = get_compile_time_arg_val(4);
  const uint32_t tiles = R * C;

  // Seed the input once and hold it resident; reserve the output once. The CB
  // hop stays out of the measured zone so we time only the transpose work.
  cb_wait_front(cb_in, tiles);
  cb_reserve_back(cb_out, tiles);
  transpose_wh_init(cb_in, cb_out);

  {
    DeviceZoneScopedN("transpose_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      for (uint32_t r = 0; r < R; r += sh) {
        for (uint32_t c = 0; c < C; c += sw) {
          tile_regs_acquire();
          // input tile (r+i, c+j) at (r+i)*C + (c+j) -> DST slot i*sw + j
          for (uint32_t i = 0; i < sh; ++i)
            for (uint32_t j = 0; j < sw; ++j)
              transpose_wh_tile(cb_in, (r + i) * C + (c + j), i * sw + j);
          tile_regs_commit();
          tile_regs_wait();
          // pack to the transposed grid position (c+j, r+i) at (c+j)*R + (r+i)
          for (uint32_t i = 0; i < sh; ++i)
            for (uint32_t j = 0; j < sw; ++j)
              pack_tile<true>(i * sw + j, cb_out, (c + j) * R + (r + i));
          tile_regs_release();
        }
      }
    }
  }

  cb_push_back(cb_out, tiles);
  cb_pop_front(cb_in, tiles);
}
