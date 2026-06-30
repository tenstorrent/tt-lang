// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FPU-op subblock probe (MB5.B). Applies one FPU op per tile over an (R x C)
// block, subblocked by a configurable (sub_h x sub_w) chunk = the DST tiles held
// per tile_regs_acquire, repeated `iters` times over a resident input (the CB
// reserve/wait/push/pop hop is outside the measured zone). Every op reads its
// operand(s) straight from the CB into srcA/srcB and writes DST -- no copy_tile,
// which the FPU ops do not need. Strictly 1:1 (or transposed) tile mapping and
// zero operand reuse, so the subblock is purely the per-acquire chunk.
//
//   op 0  transpose : transpose_wh per tile, packed to the transposed grid
//                     position (j,i) -- its access pattern depends on the
//                     subblock shape.
//   op 1  add       : add_tiles(in, in) = 2*x   (binary FPU, same position)
//   op 2  mul       : mul_tiles(in, in) = x*x   (binary FPU, same position)
//
// The binary ops feed cb_in as *both* operands, so there is no second operand
// buffer -- the footprint stays 2*R*C tiles for every op, so all reach the same
// max block size.
//
// Compile-time args: 0 = R, 1 = C, 2 = iters, 3 = sub_h, 4 = sub_w, 5 = op.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
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
  const uint32_t op = get_compile_time_arg_val(5);
  const uint32_t tiles = R * C;

  cb_wait_front(cb_in, tiles);
  cb_reserve_back(cb_out, tiles);
  if (op == 0) {
    transpose_wh_init(cb_in, cb_out);
  } else {
    binary_op_init_common(cb_in, cb_in, cb_out);
    if (op == 1) {
      add_tiles_init(cb_in, cb_in);
    } else {
      mul_tiles_init(cb_in, cb_in);
    }
  }

  {
    DeviceZoneScopedN("fpu_op_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      for (uint32_t r = 0; r < R; r += sh) {
        for (uint32_t c = 0; c < C; c += sw) {
          tile_regs_acquire();
          for (uint32_t i = 0; i < sh; ++i) {
            for (uint32_t j = 0; j < sw; ++j) {
              const uint32_t idx = (r + i) * C + (c + j);
              const uint32_t dst = i * sw + j;
              if (op == 0) {
                transpose_wh_tile(cb_in, idx, dst);
              } else if (op == 1) {
                add_tiles(cb_in, cb_in, idx, idx, dst);
              } else {
                mul_tiles(cb_in, cb_in, idx, idx, dst);
              }
            }
          }
          tile_regs_commit();
          tile_regs_wait();
          for (uint32_t i = 0; i < sh; ++i) {
            for (uint32_t j = 0; j < sw; ++j) {
              const uint32_t dst = i * sw + j;
              // transpose -> transposed grid position (j,i); binary -> same.
              const uint32_t out = (op == 0) ? ((c + j) * R + (r + i))
                                             : ((r + i) * C + (c + j));
              pack_tile<true>(dst, cb_out, out);
            }
          }
          tile_regs_release();
        }
      }
    }
  }

  cb_push_back(cb_out, tiles);
  cb_pop_front(cb_in, tiles);
}
