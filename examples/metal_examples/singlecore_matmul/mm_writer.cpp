// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments to write data back into the output buffer.
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = 16;

    // Create the address generator for the output buffer. Due to us sharing buffer and circular buffer
    // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
    // the same parameters from the circular buffer as we would from the DRAM buffer.
    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, get_tile_size(cb_id_out0));

    // write to asigned tiles
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Wait for the matrix multiplication kernel to produce an output
        cb_wait_front(cb_id_out0, 1);
        // Write the output tile to DRAM.
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        // Write to the correct offset based on start_tile
        noc_async_write_tile(i + start_tile, c, l1_read_addr);
        noc_async_write_barrier();  
        cb_pop_front(cb_id_out0, 1);
    }
}
