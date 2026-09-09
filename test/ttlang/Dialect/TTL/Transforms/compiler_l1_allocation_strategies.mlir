// Verifies deterministic strategy selection on a fragmented conflict graph.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing})' | FileCheck %s --check-prefix=FIRST
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing l1-budget-override=75000})' | FileCheck %s --check-prefix=BEST
// RUN: not ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing l1-budget-override=75000})' 2>&1 | FileCheck %s --check-prefix=FIRST-BUDGET

// Best-fit uses a finite gap for DFB 1 and reduces the arena by five tiles.
// FIRST: module attributes {ttl.dfb_allocations = [
// FIRST-SAME: l1_payload_offset = 64 : i64
// FIRST-SAME: l1_payload_offset = 71744 : i64
// FIRST-SAME: l1_payload_offset = 41024 : i64
// FIRST-SAME: l1_payload_offset = 20544 : i64
// FIRST-SAME: l1_payload_offset = 57408 : i64
// FIRST-SAME: l1_payload_offset = 64 : i64
// FIRST-SAME: ttl.l1_arena_bytes = 81984 : i64
// FIRST-LABEL: func.func @fragmented_schedule
// FIRST-BUDGET: error: 'ttl.bind_cb' op compiler-l1 placement exceeds SRAM budget 75000 bytes
// FIRST-BUDGET-SAME: first-fit-decreasing placement does not prove infeasibility

// BEST: module attributes {ttl.dfb_allocations = [
// BEST-SAME: l1_payload_offset = 64 : i64
// BEST-SAME: l1_payload_offset = 64 : i64
// BEST-SAME: l1_payload_offset = 41024 : i64
// BEST-SAME: l1_payload_offset = 20544 : i64
// BEST-SAME: l1_payload_offset = 57408 : i64
// BEST-SAME: l1_payload_offset = 41024 : i64
// BEST-SAME: ttl.l1_arena_bytes = 71744 : i64
// BEST-LABEL: func.func @fragmented_schedule

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fragmented_schedule()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %storage_0 = ttl.bind_cb {cb_index = 0, block_count = 10} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %storage_1 = ttl.bind_cb {cb_index = 1, block_count = 5} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %storage_2 = ttl.bind_cb {cb_index = 2, block_count = 8} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 8>
    %storage_3 = ttl.bind_cb {cb_index = 3, block_count = 10} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %storage_4 = ttl.bind_cb {cb_index = 4, block_count = 7} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 7>
    %storage_5 = ttl.bind_cb {cb_index = 5, block_count = 7} {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 7>

    %produced_4 = ttl.cb_reserve %storage_4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_4 : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
    %produced_0 = ttl.cb_reserve %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %produced_3 = ttl.cb_reserve %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %produced_2 = ttl.cb_reserve %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 8>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 8>
    %consumed_0 = ttl.cb_wait %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %produced_1 = ttl.cb_reserve %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %consumed_2 = ttl.cb_wait %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 8>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 8>
    %produced_5 = ttl.cb_reserve %storage_5
        : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_5 : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
    %consumed_1 = ttl.cb_wait %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 5>
    %consumed_3 = ttl.cb_wait %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 10>
    %consumed_5 = ttl.cb_wait %storage_5
        : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_5 : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
    %consumed_4 = ttl.cb_wait %storage_4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_4 : <[1, 1], !ttcore.tile<32x32, bf16>, 7>
    return
  }
}
