// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Drain NITERS chunks of NT output tiles each from cb_out to a DRAM tensor.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t niters = get_compile_time_arg_val(2);
    constexpr uint32_t CTA_BASE = 3;

    constexpr auto out_args = TensorAccessorArgs<CTA_BASE>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t it = 0; it < niters; it++) {
        cb_wait_front(cb_out, num_tiles);
        uint32_t rptr = get_read_ptr(cb_out);
        const uint32_t base = it * num_tiles;
        for (uint32_t t = 0; t < num_tiles; t++) {
            noc_async_write_tile(base + t, out, rptr);
            rptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, num_tiles);
    }
}
