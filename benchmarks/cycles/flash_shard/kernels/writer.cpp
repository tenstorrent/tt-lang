// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the metal flash-shard baseline. Output and stats are single-core
// L1 shards aliased by their CBs, so the writer just waits for compute to fill
// them (no NoC) -- the data is already in place once the front is signalled.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(1);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t stats_num_tiles = get_compile_time_arg_val(3);

    cb_wait_front(cb_out, out_num_tiles);
    cb_wait_front(cb_stats, stats_num_tiles);
}
