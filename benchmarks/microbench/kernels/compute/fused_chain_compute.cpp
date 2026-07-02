// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused SFPU chain compute probe, extracted from the tt-lang-generated kernel
// for  v = abs(neg(relu(sigmoid(c) + tanh(b)))).  Applies the chain to a flat
// array of `tiles` output tiles (two inputs b, c; one output), subblocked by
// `sub` = the number of OUTPUT tiles processed per tile_regs_acquire. Same flat
// style as compute_unary.cpp (no iters loop -- one pass over the tiles).
//
// DST layout matches the generated code: each output tile uses TWO dst slots
// because the fused add needs both operands live at once -- c/sigmoid -> even
// slot 2*i, b/tanh -> odd slot 2*i+1, then add_binary(2*i, 2*i+1 -> 2*i) and the
// relu/neg/abs tail run on the even slot, which is packed out. So the subblock
// costs 2*sub dst slots: sub must satisfy 2*sub <= DST capacity.
//
// The per-op inits stay INSIDE the acquire, interleaved with their applies
// (init_sfpu once at top; then per op: *_tile_init followed by *_tile over the
// chunk), matching the generated body -- each SFPU op reconfigures the unit, so
// its init must precede its applies and cannot be hoisted across ops or chunks.
// add_binary_tile is the DST->DST fused add; the output is per-tile pack_tile.
//
// Compile-time args: 0 = tiles (output tiles), 1 = sub (output tiles per acquire).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_b = 0;    // tanh operand
constexpr uint32_t cb_c = 1;    // sigmoid operand
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t tiles = get_compile_time_arg_val(0);
  const uint32_t sub = get_compile_time_arg_val(1);

  init_sfpu(cb_b, cb_out);

  cb_wait_front(cb_b, tiles);      // resident inputs (flat array of `tiles`)
  cb_wait_front(cb_c, tiles);
  cb_reserve_back(cb_out, tiles);
  {
    DeviceZoneScopedN("fused_chain_loop");
    for (uint32_t base = 0; base < tiles; base += sub) {
      uint32_t chunk = (tiles - base < sub) ? (tiles - base) : sub;
      tile_regs_acquire();

      // Load: c -> even slot 2*i, b -> odd slot 2*i+1.
      copy_tile_init(cb_c);
      for (uint32_t i = 0; i < chunk; ++i) {
        copy_tile(cb_c, base + i, 2 * i);
      }
      copy_tile_init(cb_b);
      for (uint32_t i = 0; i < chunk; ++i) {
        copy_tile(cb_b, base + i, 2 * i + 1);
      }

      // sigmoid(c) on even slots, tanh(b) on odd slots.
      sigmoid_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        sigmoid_tile(2 * i);
      }
      tanh_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        tanh_tile(2 * i + 1);
      }

      // sigmoid(c) + tanh(b) -> even slot (DST->DST add).
      add_binary_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        add_binary_tile(2 * i, 2 * i + 1, 2 * i);
      }

      // relu -> neg -> abs on the even slot (the accumulated result).
      relu_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        relu_tile(2 * i);
      }
      negative_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        negative_tile(2 * i);
      }
      abs_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        abs_tile(2 * i);
      }

      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t i = 0; i < chunk; ++i) {
        pack_tile<true>(2 * i, cb_out, base + i);
      }
      tile_regs_release();
    }
  }
  cb_pop_front(cb_b, tiles);
  cb_pop_front(cb_c, tiles);
  cb_push_back(cb_out, tiles);
}
