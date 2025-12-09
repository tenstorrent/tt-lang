// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused kernel: performs f(A + B) + C where f is exp or relu (configurable via
// USE_RELU define) Processes tiles in 2D blocks (BLOCK_H x BLOCK_W tiles) Uses
// DST_TILES tiles per DST acquire cycle to maximize register usage With 8 DST
// registers and 2 needed per tile, we can process 4 tiles at once

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

// Unary operation selection: USE_RELU for cheap op, otherwise exp (expensive)
#ifdef USE_RELU
#define UNARY_INIT() relu_tile_init()
#define UNARY_TILE(i) relu_tile(i)
#else
#define UNARY_INIT() exp_tile_init()
#define UNARY_TILE(i) exp_tile(i)
#endif

// Block configuration (passed via defines from Python)
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr auto cb_in0 = tt::CBIndex::c_0; // A
  constexpr auto cb_in1 = tt::CBIndex::c_1; // B
  constexpr auto cb_in2 = tt::CBIndex::c_2; // C
  constexpr auto cb_out = tt::CBIndex::c_16;

  constexpr uint32_t block_h = BLOCK_H;
  constexpr uint32_t block_w = BLOCK_W;
  constexpr uint32_t block_size = block_h * block_w;

  // 8 DST registers available, 2 needed per tile for binary ops = 4 tiles max
  constexpr uint32_t num_dst_regs = 8;
  constexpr uint32_t dst_regs_per_tile = 2; // chain of binary & unary ops
  constexpr uint32_t tiles_per_dst_cycle = num_dst_regs / dst_regs_per_tile;

  // Initialize the SFPU path (unary operations + copy_tile)
  init_sfpu(cb_in0, cb_out);

  // Process tiles in blocks
  uint32_t num_blocks = n_tiles / block_size;
  uint32_t remaining_tiles = n_tiles % block_size;

  // Compute remainder for block processing (64 % 4 = 0, but keep for
  // generality)
  constexpr uint32_t full_cycles_per_block = block_size / tiles_per_dst_cycle;
  constexpr uint32_t block_remainder = block_size % tiles_per_dst_cycle;

  for (uint32_t block = 0; block < num_blocks; block++) {
    // Wait for entire block of inputs from reader kernel
    cb_wait_front(cb_in0, block_size);
    cb_wait_front(cb_in1, block_size);
    cb_wait_front(cb_in2, block_size);

    DeviceZoneScopedN("COMPUTE-block");

    // Process block in DST-sized chunks
    for (uint32_t dst_cycle = 0; dst_cycle < full_cycles_per_block;
         dst_cycle++) {
      uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

      // Acquire DST registers for this batch
      tile_regs_acquire();

      // Copy A tiles to DST registers 0-3
      copy_tile_init(cb_in0);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in0, base_t + i, i); // A[i] -> DST[i]
      }

      // Copy B tiles to DST registers 4-7
      copy_tile_init(cb_in1);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in1, base_t + i,
                  i + tiles_per_dst_cycle); // B[i] -> DST[i+4]
      }

      // First add: A + B for all tiles
      add_binary_tile_init();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle,
                        i); // DST[i] + DST[i+4] -> DST[i]
      }

      // Unary op: f(A + B) for all tiles (in-place on registers 0-3)
      UNARY_INIT();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        UNARY_TILE(i);
      }

      // Copy C tiles to DST registers 4-7
      copy_tile_init(cb_in2);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in2, base_t + i,
                  i + tiles_per_dst_cycle); // C[i] -> DST[i+4]
      }

      // Second add: f(A + B) + C for all tiles
      add_binary_tile_init();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle,
                        i); // DST[i] + DST[i+4] -> DST[i]
      }

      // Commit and wait for results
      tile_regs_commit();
      tile_regs_wait();

      // Pack all output tiles from DST registers 0-3
      cb_reserve_back(cb_out, tiles_per_dst_cycle);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        pack_tile(i, cb_out);
      }
      tile_regs_release();

      cb_push_back(cb_out, tiles_per_dst_cycle);
    }

    // Handle remainder tiles within block (if block_size not divisible by
    // tiles_per_dst_cycle)
    if constexpr (block_remainder > 0) {
      uint32_t base_t = full_cycles_per_block * tiles_per_dst_cycle;

      tile_regs_acquire();

      copy_tile_init(cb_in0);
      for (uint32_t i = 0; i < block_remainder; i++) {
        copy_tile(cb_in0, base_t + i, i);
      }
      copy_tile_init(cb_in1);
      for (uint32_t i = 0; i < block_remainder; i++) {
        copy_tile(cb_in1, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < block_remainder; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
      }

      UNARY_INIT();
      for (uint32_t i = 0; i < block_remainder; i++) {
        UNARY_TILE(i);
      }

      copy_tile_init(cb_in2);
      for (uint32_t i = 0; i < block_remainder; i++) {
        copy_tile(cb_in2, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < block_remainder; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
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

    // Pop entire block of inputs
    cb_pop_front(cb_in0, block_size);
    cb_pop_front(cb_in1, block_size);
    cb_pop_front(cb_in2, block_size);
  }

  // Handle remaining tiles (if n_tiles not divisible by block_size)
  if (remaining_tiles > 0) {
    cb_wait_front(cb_in0, remaining_tiles);
    cb_wait_front(cb_in1, remaining_tiles);
    cb_wait_front(cb_in2, remaining_tiles);

    // Process remaining tiles in DST-sized chunks
    uint32_t remaining_dst_cycles = remaining_tiles / tiles_per_dst_cycle;
    uint32_t final_remainder = remaining_tiles % tiles_per_dst_cycle;

    for (uint32_t dst_cycle = 0; dst_cycle < remaining_dst_cycles;
         dst_cycle++) {
      uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

      tile_regs_acquire();

      copy_tile_init(cb_in0);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in0, base_t + i, i);
      }
      copy_tile_init(cb_in1);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in1, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
      }

      UNARY_INIT();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        UNARY_TILE(i);
      }

      copy_tile_init(cb_in2);
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        copy_tile(cb_in2, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
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

    // Handle final remainder (< tiles_per_dst_cycle)
    if (final_remainder > 0) {
      uint32_t base_t = remaining_dst_cycles * tiles_per_dst_cycle;

      tile_regs_acquire();

      copy_tile_init(cb_in0);
      for (uint32_t i = 0; i < final_remainder; i++) {
        copy_tile(cb_in0, base_t + i, i);
      }
      copy_tile_init(cb_in1);
      for (uint32_t i = 0; i < final_remainder; i++) {
        copy_tile(cb_in1, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < final_remainder; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
      }

      UNARY_INIT();
      for (uint32_t i = 0; i < final_remainder; i++) {
        UNARY_TILE(i);
      }

      copy_tile_init(cb_in2);
      for (uint32_t i = 0; i < final_remainder; i++) {
        copy_tile(cb_in2, base_t + i, i + tiles_per_dst_cycle);
      }

      add_binary_tile_init();
      for (uint32_t i = 0; i < final_remainder; i++) {
        add_binary_tile(i, i + tiles_per_dst_cycle, i);
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
    cb_pop_front(cb_in1, remaining_tiles);
    cb_pop_front(cb_in2, remaining_tiles);
  }
}
} // namespace NAMESPACE
