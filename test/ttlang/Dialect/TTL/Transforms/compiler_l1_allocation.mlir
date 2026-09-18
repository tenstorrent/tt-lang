// Verifies byte reuse across formats, distinct control records, and both memory models.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})' | FileCheck %s --check-prefix=L1
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 reuse-user-dfbs=false})' | FileCheck %s --check-prefix=DISTINCT
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=metal-cb})' | FileCheck %s --check-prefix=METAL

// A completed BF16 acquisition can share FP32 payload bytes but retains its own state.
// L1: module attributes {ttl.dfb_allocations = [
// L1-SAME: l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64
// L1-SAME: l1_allocation_bytes = 4096 : i64, l1_offset = 8 : i64, l1_payload_offset = 64 : i64
// L1-SAME: ttl.l1_arena_bytes = 4160 : i64
// L1-SAME: ttl.memory_model = "compiler-l1"
// L1-LABEL: func.func @ordered_mixed_formats
// L1-SAME: ttl.base_cta_index = 1 : i32
// L1-NEXT: %[[FIRST:.*]] = ttl.bind_cb
// L1-NEXT: %[[SECOND:.*]] = ttl.bind_cb
// L1-NEXT: %{{.*}} = ttl.cb_reserve %[[FIRST]]

// DISTINCT: l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 4160 : i64
// DISTINCT-SAME: l1_allocation_bytes = 4096 : i64, l1_offset = 8 : i64, l1_payload_offset = 64 : i64
// DISTINCT-SAME: ttl.l1_arena_bytes = 6208 : i64
// DISTINCT-LABEL: func.func @ordered_mixed_formats
// DISTINCT-SAME: ttl.base_cta_index = 1 : i32

// METAL-NOT: l1_payload_offset
// METAL-NOT: ttl.memory_model
// METAL-LABEL: func.func @ordered_mixed_formats
// METAL-SAME: ttl.base_cta_index = 2 : i32

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @ordered_mixed_formats()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>

    %first_produced = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_consumed = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>

    %second_produced = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %second_consumed = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    return
  }
}
