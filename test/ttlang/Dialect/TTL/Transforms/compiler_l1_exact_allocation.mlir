// Verifies exact placement improves both decreasing-size heuristics and fits a lower SRAM budget.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing})' | FileCheck %s --check-prefix=GREEDY
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing})' | FileCheck %s --check-prefix=GREEDY
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact l1-budget-override=40000})' | FileCheck %s --check-prefix=EXACT

// Both heuristics use 22 tiles. Exact placement proves that 16 tiles suffice.
// GREEDY: module attributes {ttl.dfb_allocations = [
// GREEDY-SAME: ttl.l1_arena_bytes = 45120 : i64
// GREEDY-LABEL: func.func @fragmented_schedule
// EXACT: module attributes {ttl.dfb_allocations = [
// EXACT-SAME: dfb_index = 0
// EXACT-SAME: l1_payload_offset = 10304 : i64
// EXACT-SAME: dfb_index = 1
// EXACT-SAME: l1_payload_offset = 64 : i64
// EXACT-SAME: dfb_index = 2
// EXACT-SAME: l1_payload_offset = 18496 : i64
// EXACT-SAME: dfb_index = 3
// EXACT-SAME: l1_payload_offset = 64 : i64
// EXACT-SAME: dfb_index = 4
// EXACT-SAME: l1_payload_offset = 16448 : i64
// EXACT-SAME: dfb_index = 5
// EXACT-SAME: l1_payload_offset = 24640 : i64
// EXACT-SAME: ttl.l1_arena_bytes = 32832 : i64
// EXACT-LABEL: func.func @fragmented_schedule

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fragmented_schedule()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %storage_0 = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %storage_1 = ttl.bind_cb {cb_index = 1, block_count = 6} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %storage_2 = ttl.bind_cb {cb_index = 2, block_count = 6} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %storage_3 = ttl.bind_cb {cb_index = 3, block_count = 5} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %storage_4 = ttl.bind_cb {cb_index = 4, block_count = 4} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %storage_5 = ttl.bind_cb {cb_index = 5, block_count = 4} {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>

    %produced_0 = ttl.cb_reserve %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %produced_2 = ttl.cb_reserve %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %produced_3 = ttl.cb_reserve %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %consumed_2 = ttl.cb_wait %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %produced_4 = ttl.cb_reserve %storage_4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_4 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %produced_5 = ttl.cb_reserve %storage_5
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_5 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %consumed_3 = ttl.cb_wait %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %consumed_0 = ttl.cb_wait %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %produced_1 = ttl.cb_reserve %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %consumed_1 = ttl.cb_wait %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 6>
    %consumed_5 = ttl.cb_wait %storage_5
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_5 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %consumed_4 = ttl.cb_wait %storage_4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_4 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}
