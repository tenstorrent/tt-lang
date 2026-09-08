// Equal-size chain: stable owner order uses three ranges; degree-aware ordering needs two.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})' | FileCheck %s --check-prefix=MULTI
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=multi-order-decreasing l1-budget-override=4160})' | FileCheck %s --check-prefix=MULTI
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing})' | FileCheck %s --check-prefix=STABLE
// MULTI: ttl.l1_arena_bytes = 4160 : i64
// MULTI-LABEL: func.func @schedule
// STABLE: ttl.l1_arena_bytes = 6208 : i64
// STABLE-LABEL: func.func @schedule

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @schedule() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %storage_0 = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %storage_1 = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %storage_2 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %storage_3 = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_0 = ttl.cb_reserve %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_1 = ttl.cb_reserve %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_2 = ttl.cb_wait %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_3 = ttl.cb_reserve %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_4 = ttl.cb_wait %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_5 = ttl.cb_reserve %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_6 = ttl.cb_wait %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view_7 = ttl.cb_wait %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
