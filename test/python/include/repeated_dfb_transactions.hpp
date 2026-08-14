// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

template <typename Source, typename Completion>
inline void consume_repeated_dfb_and_signal() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  for (uint32_t transaction = 0; transaction < 4; ++transaction) {
    cb_wait_front(Source::index, Source::pages_per_block);
    cb_pop_front(Source::index, Source::pages_per_block);
  }
  cb_reserve_back(Completion::index, Completion::pages_per_block);
  cb_push_back(Completion::index, Completion::pages_per_block);
#endif
}

template <typename Destination, typename SourceAccessor>
inline void read_high_water_dfb_logical_dm(SourceAccessor source) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
  static_assert(Destination::pages_per_block == 4);
  static_assert(Destination::block_count == 3);
  constexpr uint32_t transactionCount = 4;
  constexpr uint32_t transactionTiles = Destination::pages_per_block;
  constexpr uint32_t highWaterTiles = 2 * Destination::pages_per_block;

  CircularBuffer destination(get_compile_time_arg_val(Destination::index));
  Noc noc0(0);
  destination.reserve_back(highWaterTiles);
  for (uint32_t transaction = 0; transaction < transactionCount;
       ++transaction) {
    for (uint32_t tile = 0; tile < transactionTiles; ++tile) {
      uint32_t pageId = transaction * transactionTiles + tile;
      noc0.async_read(source, destination, Destination::page_size_bytes,
                      {.page_id = pageId},
                      {.offset_bytes = tile * Destination::page_size_bytes});
    }
    noc0.async_read_barrier();
    destination.push_back(transactionTiles);
    if (transaction + 1 < transactionCount) {
      destination.reserve_back(highWaterTiles);
    }
  }
#endif
}

template <typename Source, typename DestinationAccessor>
inline void write_high_water_dfb_logical_dm(DestinationAccessor destination) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
  static_assert(Source::pages_per_block == 4);
  static_assert(Source::block_count == 3);
  constexpr uint32_t transactionCount = 4;
  constexpr uint32_t transactionTiles = Source::pages_per_block;
  CircularBuffer source(get_compile_time_arg_val(Source::index));
  Noc noc0(0);
  for (uint32_t transaction = 0; transaction < transactionCount;
       ++transaction) {
    source.wait_front(transactionTiles);
    for (uint32_t tile = 0; tile < transactionTiles; ++tile) {
      uint32_t pageId = transaction * transactionTiles + tile;
      noc0.async_write(source, destination, Source::page_size_bytes,
                       {.offset_bytes = tile * Source::page_size_bytes},
                       {.page_id = pageId});
    }
    noc0.async_write_barrier();
    source.pop_front(transactionTiles);
  }
#endif
}

template <typename Source, typename Completion, typename DestinationAccessor>
inline void
write_high_water_dfb_and_signal_logical_dm(DestinationAccessor destination) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
  write_high_water_dfb_logical_dm<Source>(destination);
  CircularBuffer completion(get_compile_time_arg_val(Completion::index));
  completion.reserve_back(Completion::pages_per_block);
  completion.push_back(Completion::pages_per_block);
#endif
}

template <typename Source, typename Destination>
inline void copy_repeated_dfb() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  unary_op_init_common(Source::index, Destination::index);
  copy_tile_init(Source::index);
  for (uint32_t transaction = 0; transaction < 4; ++transaction) {
    cb_wait_front(Source::index, Source::pages_per_block);
    cb_reserve_back(Destination::index, Destination::pages_per_block);
    for (uint32_t tile = 0; tile < Source::pages_per_block; ++tile) {
      tile_regs_acquire();
      copy_tile(Source::index, tile, 0);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(0, Destination::index);
      tile_regs_release();
    }
    cb_pop_front(Source::index, Source::pages_per_block);
    cb_push_back(Destination::index, Destination::pages_per_block);
  }
#endif
}
