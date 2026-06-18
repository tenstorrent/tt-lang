// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB2, L1-pack accumulation: out = initial + sum(expr(delta_i)). The
// accumulator lives in L1 and is re-packed every step. The initial value is
// packed once with pack_reconfig_l1_acc(0) (overwrite); each contribution
// expression is computed in DST and packed with pack_reconfig_l1_acc(1), so the
// accumulator round-trips through L1 every step.
//
// Compile-time args: 0 = accumulator tiles, 1 = contributions, 2 = DST
// capacity, 3 = reuse (1 = contributions L1-resident: re-read one block every
// iteration; 0 = streamed: consume a fresh contribution block each iteration),
// 4 = expr. expr ids: 0 add, 1 mul, 2 gelu.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_init = 0;
constexpr uint32_t dfb_delta = 1;
constexpr uint32_t dfb_out = 16;
constexpr uint32_t expr_add = 0;
constexpr uint32_t expr_mul = 1;
constexpr uint32_t expr_gelu = 2;

// Copy acc_tiles tiles from src_cb into DST in DST-capacity sub-blocks and pack
// them to dfb_out (packer L1-accumulation governed by the caller's reconfig).
inline void copy_and_pack(uint32_t src_cb, uint32_t acc_tiles, uint32_t cap) {
  for (uint32_t base = 0; base < acc_tiles; base += cap) {
    uint32_t chunk = (acc_tiles - base < cap) ? (acc_tiles - base) : cap;
    tile_regs_acquire();
    for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
      copy_tile(src_cb, base + tile_offset, tile_offset);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
      pack_tile<true>(tile_offset, dfb_out, base + tile_offset);
    }
    tile_regs_release();
  }
}

inline void expr_and_pack(uint32_t src_cb, uint32_t acc_tiles, uint32_t cap,
                          uint32_t expr) {
  for (uint32_t base = 0; base < acc_tiles; base += cap) {
    uint32_t chunk = (acc_tiles - base < cap) ? (acc_tiles - base) : cap;
    tile_regs_acquire();
    if (expr == expr_add) {
      for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
        copy_tile(src_cb, base + tile_offset, tile_offset);
      }
    } else if (expr == expr_mul) {
      for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
        mul_tiles(src_cb, src_cb, base + tile_offset, base + tile_offset,
                  tile_offset);
      }
    } else if (expr == expr_gelu) {
      for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
        copy_tile(src_cb, base + tile_offset, tile_offset);
      }
      for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
        gelu_tile(tile_offset);
      }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile_offset = 0; tile_offset < chunk; ++tile_offset) {
      pack_tile<true>(tile_offset, dfb_out, base + tile_offset);
    }
    tile_regs_release();
  }
}

void kernel_main() {
  const uint32_t acc_tiles = get_compile_time_arg_val(0);
  const uint32_t iters = get_compile_time_arg_val(1);
  const uint32_t cap = get_compile_time_arg_val(2);
  const uint32_t reuse = get_compile_time_arg_val(3);
  const uint32_t expr = get_compile_time_arg_val(4);

  init_sfpu(dfb_init, dfb_out);
  copy_tile_init(dfb_init);

  cb_reserve_back(dfb_out, acc_tiles);
  pack_reconfig_l1_acc(0); // initial value overwrites the L1 accumulator
  {
    DeviceZoneScopedN("acc_loop");
    cb_wait_front(dfb_init, acc_tiles);
    copy_and_pack(dfb_init, acc_tiles, cap);
    cb_pop_front(dfb_init, acc_tiles);
    if (expr == expr_add) {
      copy_tile_init(dfb_delta);
    } else if (expr == expr_mul) {
      binary_op_init_common(dfb_delta, dfb_delta, dfb_out);
      mul_tiles_init(dfb_delta, dfb_delta, 0, __builtin_LINE());
    } else if (expr == expr_gelu) {
      init_sfpu(dfb_delta, dfb_out);
      copy_tile_init(dfb_delta);
      gelu_tile_init();
    }
    pack_reconfig_l1_acc(1); // contributions accumulate in L1
    for (uint32_t iter = 0; iter < iters; ++iter) {
      cb_wait_front(dfb_delta, acc_tiles);
      expr_and_pack(dfb_delta, acc_tiles, cap, expr);
      if (!reuse) {
        cb_pop_front(dfb_delta, acc_tiles);
      }
    }
    if (reuse) {
      cb_pop_front(dfb_delta, acc_tiles);
    }
  }
  pack_reconfig_l1_acc(0);
  cb_push_back(dfb_out, acc_tiles);
}
