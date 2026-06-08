// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Stream NITERS chunks of NT input tiles each from a DRAM-interleaved tensor
// into cb_in (double-buffered, handshaked). Chunk `it` = tiles [it*NT, ..).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t niters = get_compile_time_arg_val(2);
    constexpr uint32_t CTA_BASE = 3;

    constexpr auto in_args = TensorAccessorArgs<CTA_BASE>();
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);

    for (uint32_t it = 0; it < niters; it++) {
        cb_reserve_back(cb_in, num_tiles);
        uint32_t wptr = get_write_ptr(cb_in);
        const uint32_t base = it * num_tiles;
        for (uint32_t t = 0; t < num_tiles; t++) {
            noc_async_read_tile(base + t, in, wptr);
            wptr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, num_tiles);
    }
}
