// Verifies compiler-managed allocation with Wormhole alignment and no lifecycle boundary.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=multi-order-decreasing})' | FileCheck %s

// CHECK-LABEL: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: l1_offset = 0 : i64, l1_payload_offset = 32 : i64
// CHECK-SAME: l1_offset = 8 : i64, l1_payload_offset = 32 : i64
// CHECK-SAME: ttl.l1_arena_bytes = 2080 : i64
// CHECK-SAME: ttl.memory_model = "compiler-l1"
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @wormhole_transfer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_produced = ttl.cb_reserve %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_consumed = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_produced = ttl.cb_reserve %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_consumed = ttl.cb_wait %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
