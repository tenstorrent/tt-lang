// SPDX-FileCopyrightText: (C) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Elementwise add with broadcast on the second input.

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

// Block configuration (passed via defines from Python).
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

namespace NAMESPACE {
void MAIN {
  const uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr auto cb_in0 = tt::CBIndex::c_0; // A
  constexpr auto cb_in1 = tt::CBIndex::c_1; // B (broadcast tile)
  constexpr auto cb_out = tt::CBIndex::c_16;

  constexpr uint32_t block_h = BLOCK_H;
  constexpr uint32_t block_w = BLOCK_W;
  constexpr uint32_t block_size = block_h * block_w;

  constexpr uint32_t num_dst_regs = 8;
  constexpr uint32_t tiles_per_dst_cycle = num_dst_regs;

#if defined(BCAST_ROW)
  // Initialize bcast add (rows). B is a single tile.
  init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(cb_in0, cb_in1,
                                                           cb_out);
#elif defined(BCAST_SCALAR)
  // Initialize bcast add (scalar). B is a single tile.
  init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(cb_in0, cb_in1,
                                                              cb_out);
#else
  // Initialize bcast add (columns). B is a single tile.
  init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(cb_in0, cb_in1,
                                                           cb_out);
#endif

  uint32_t num_blocks = n_tiles / block_size;
  uint32_t remaining_tiles = n_tiles % block_size;

  constexpr uint32_t full_cycles_per_block = block_size / tiles_per_dst_cycle;
  constexpr uint32_t block_remainder = block_size % tiles_per_dst_cycle;

  for (uint32_t block = 0; block < num_blocks; block++) {
    cb_wait_front(cb_in0, block_size);
    cb_wait_front(cb_in1, 1);

    DeviceZoneScopedN("COMPUTE-block");

    for (uint32_t dst_cycle = 0; dst_cycle < full_cycles_per_block;
         dst_cycle++) {
      const uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

      tile_regs_acquire();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
#if defined(BCAST_ROW)
        add_tiles_bcast_rows(cb_in0, cb_in1, base_t + i, 0, i);
#elif defined(BCAST_SCALAR)
        add_tiles_bcast_scalar(cb_in0, cb_in1, base_t + i, 0, i);
#else
        add_tiles_bcast_cols(cb_in0, cb_in1, base_t + i, 0, i);
#endif
      }

      tile_regs_commit();
      tile_regs_wait();

      cb_reserve_back(cb_out, tiles_per_dst_cycle);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        pack_tile(i, cb_out);
      }
      tile_regs_release();
      cb_push_back(cb_out, tiles_per_dst_cycle);
    }

    if constexpr (block_remainder > 0) {
      const uint32_t base_t = full_cycles_per_block * tiles_per_dst_cycle;

      tile_regs_acquire();
      for (uint32_t i = 0; i < block_remainder; i++) {
#if defined(BCAST_ROW)
        add_tiles_bcast_rows(cb_in0, cb_in1, base_t + i, 0, i);
#elif defined(BCAST_SCALAR)
        add_tiles_bcast_scalar(cb_in0, cb_in1, base_t + i, 0, i);
#else
        add_tiles_bcast_cols(cb_in0, cb_in1, base_t + i, 0, i);
#endif
      }

      tile_regs_commit();
      tile_regs_wait();

      cb_reserve_back(cb_out, block_remainder);
      for (uint32_t i = 0; i < block_remainder; i++) {
        pack_tile(i, cb_out);
      }
      tile_regs_release();
      cb_push_back(cb_out, block_remainder);
    }

    cb_pop_front(cb_in0, block_size);
  }

  if (remaining_tiles > 0) {
    cb_wait_front(cb_in0, remaining_tiles);
    cb_wait_front(cb_in1, 1);

    const uint32_t remaining_dst_cycles = remaining_tiles / tiles_per_dst_cycle;
    const uint32_t final_remainder = remaining_tiles % tiles_per_dst_cycle;

    for (uint32_t dst_cycle = 0; dst_cycle < remaining_dst_cycles;
         dst_cycle++) {
      const uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

      tile_regs_acquire();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
#if defined(BCAST_ROW)
        add_tiles_bcast_rows(cb_in0, cb_in1, base_t + i, 0, i);
#elif defined(BCAST_SCALAR)
        add_tiles_bcast_scalar(cb_in0, cb_in1, base_t + i, 0, i);
#else
        add_tiles_bcast_cols(cb_in0, cb_in1, base_t + i, 0, i);
#endif
      }

      tile_regs_commit();
      tile_regs_wait();

      cb_reserve_back(cb_out, tiles_per_dst_cycle);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        pack_tile(i, cb_out);
      }
      tile_regs_release();
      cb_push_back(cb_out, tiles_per_dst_cycle);
    }

    if (final_remainder > 0) {
      const uint32_t base_t = remaining_dst_cycles * tiles_per_dst_cycle;

      tile_regs_acquire();
      for (uint32_t i = 0; i < final_remainder; i++) {
#if defined(BCAST_ROW)
        add_tiles_bcast_rows(cb_in0, cb_in1, base_t + i, 0, i);
#elif defined(BCAST_SCALAR)
        add_tiles_bcast_scalar(cb_in0, cb_in1, base_t + i, 0, i);
#else
        add_tiles_bcast_cols(cb_in0, cb_in1, base_t + i, 0, i);
#endif
      }

      tile_regs_commit();
      tile_regs_wait();

      cb_reserve_back(cb_out, final_remainder);
      for (uint32_t i = 0; i < final_remainder; i++) {
        pack_tile(i, cb_out);
      }
      tile_regs_release();
      cb_push_back(cb_out, final_remainder);
    }

    cb_pop_front(cb_in0, remaining_tiles);
  }
}
} // namespace NAMESPACE
