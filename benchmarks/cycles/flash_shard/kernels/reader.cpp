// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the metal flash-shard baseline. Q and K are single-core L1 shards,
// so the CBs already alias their data -- the reader just signals availability
// (reserve/push, no NoC). Keeping the whole K slice resident isolates the
// compute being measured (no DRAM streaming); the chunk loop walks the K shard.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(4);

    cb_reserve_back(cb_q, num_tiles_k);
    cb_push_back(cb_q, num_tiles_k);
    for (uint32_t i = 0; i < num_chunks; i++) {
        cb_reserve_back(cb_k, num_tiles_k * chunk_size);
        cb_push_back(cb_k, num_tiles_k * chunk_size);
    }
}
