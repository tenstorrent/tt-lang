// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the metal flash-shard baseline. Q/out/stats stay single-core L1
// shards (their CBs alias the data), but K streams from a DRAM-interleaved
// tensor tile-by-tile into a double-buffered cb_k so the per-core K slice need
// not be L1-resident -- this is what lets the metal baseline reach 32k+ seq.
//
// The stream mirrors tt-lang's emitted ncrisc reader: a TensorAccessor over the
// interleaved K tensor, then per chunk reserve_back / noc_async_read_tile loop
// / read_barrier / push_back. K tiles are row-major, so chunk `c` occupies the
// contiguous tile range [c * tiles_per_chunk, (c+1) * tiles_per_chunk).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
  constexpr uint32_t cb_q = get_compile_time_arg_val(0);
  constexpr uint32_t cb_k = get_compile_time_arg_val(1);
  constexpr uint32_t chunk_size = get_compile_time_arg_val(2);
  constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
  constexpr uint32_t num_tiles_k = get_compile_time_arg_val(4);
  // K's TensorAccessorArgs follow the five scalar CTAs above.
  constexpr uint32_t K_CTA_BASE = 5;

  // Q is an L1 shard aliased by cb_q -- signal it once, no NoC.
  cb_reserve_back(cb_q, num_tiles_k);
  cb_push_back(cb_q, num_tiles_k);

  constexpr auto k_args = TensorAccessorArgs<K_CTA_BASE>();
  const uint32_t k_addr = get_common_arg_val<uint32_t>(0);
  const uint32_t tile_bytes = get_tile_size(cb_k);
  const auto k = TensorAccessor(k_args, k_addr, tile_bytes);

  const uint32_t tiles_per_chunk = num_tiles_k * chunk_size;
  for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
    cb_reserve_back(cb_k, tiles_per_chunk);
    uint32_t wptr = get_write_ptr(cb_k);
    const uint32_t base_tile = chunk * tiles_per_chunk;
    for (uint32_t t = 0; t < tiles_per_chunk; t++) {
      noc_async_read_tile(base_tile + t, k, wptr);
      wptr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, tiles_per_chunk);
  }
}
