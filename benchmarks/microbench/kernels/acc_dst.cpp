// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB2, DST-resident accumulation: out = initial + sum(expr(delta_i)).
// The accumulator lives in DST across the loop (one acquire). It is seeded from
// the initial DFB with copy_tile, then each contribution is accumulated in
// place. expr=add uses binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>. expr=gelu
// copies each delta to a temporary DST slot, applies fast GELU, then adds the
// temporary tile into the accumulator via add_binary_tile. The result is packed
// once; no per-iteration pack to L1.
//
// expr=mul has no DST-resident form: mul_tiles writes its product with
// EltwiseBinaryReuseDestType::NONE (overwrites the dest tile, no accumulation)
// and no FPU op adds two DST tiles, so a product cannot accumulate in DST in
// place. It is handled only by the L1-pack strategy (packer accumulation). See
// https://github.com/tenstorrent/tt-metal/blob/ba9340e3a45ac5ba51c752a49341f2def28d0514/tt_metal/hw/inc/api/compute/eltwise_binary.h#L168-L185
//
// Compile-time args: 0 = accumulator tiles, 1 = contributions, 2 = reuse,
// 3 = expr. reuse: 1 re-reads one resident contribution block; 0 consumes one
// streamed contribution block per iteration. expr ids: 0 add, 2 gelu.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
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
constexpr uint32_t expr_gelu = 2;

void kernel_main() {
  const uint32_t acc_tiles = get_compile_time_arg_val(0);
  const uint32_t iters = get_compile_time_arg_val(1);
  const uint32_t reuse = get_compile_time_arg_val(2);
  const uint32_t expr = get_compile_time_arg_val(3);

  if (expr == expr_gelu) {
    init_sfpu(dfb_delta, dfb_out);
  } else if (expr == expr_add) {
    // binary_op_init_common reinitializes the math/pack sync and dest config,
    // resetting DST, so it must run before the copy_tile seed below, not after.
    binary_op_init_common(dfb_delta, dfb_delta, dfb_out);
  }
  copy_tile_init(dfb_init);

  tile_regs_acquire();
  cb_reserve_back(dfb_out, acc_tiles);
  {
    DeviceZoneScopedN("acc_loop");
    cb_wait_front(dfb_init, acc_tiles);
    for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
      copy_tile(dfb_init, tile_index, tile_index);
    }
    cb_pop_front(dfb_init, acc_tiles);

    if (expr == expr_add) {
      binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD,
                                   EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
          dfb_delta);
    }

    for (uint32_t iter = 0; iter < iters; ++iter) {
      cb_wait_front(dfb_delta, acc_tiles);
      if (expr == expr_add) {
        for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
          binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD,
                                  EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
              dfb_delta, tile_index, tile_index);
        }
      } else if (expr == expr_gelu) {
        copy_tile_init(dfb_delta);
        for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
          copy_tile(dfb_delta, tile_index, acc_tiles + tile_index);
        }
        gelu_tile_init();
        for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
          gelu_tile(acc_tiles + tile_index);
        }
        add_binary_tile_init();
        for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
          add_binary_tile(tile_index, acc_tiles + tile_index, tile_index);
        }
      }
      if (!reuse) {
        cb_pop_front(dfb_delta, acc_tiles);
      }
    }
    if (reuse) {
      cb_pop_front(dfb_delta, acc_tiles);
    }
    // Single pack is inside the timed zone, matching L1-pack's per-step packs.
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile_index = 0; tile_index < acc_tiles; ++tile_index) {
      pack_tile<true>(tile_index, dfb_out, tile_index);
    }
  }
  cb_push_back(dfb_out, acc_tiles);
  tile_regs_release();
}
