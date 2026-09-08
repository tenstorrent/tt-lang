// Per-core allocation omits inactive payloads; uniform allocation retains one layout.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 sram-allocation-mode=uniform})' | FileCheck %s --check-prefix=UNIFORM
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 sram-allocation-mode=per-core})' | FileCheck %s --check-prefix=INDEPENDENT

// UNIFORM: ttl.l1_arena_bytes = 32832 : i64
// UNIFORM-NOT: sram_core_layouts
// UNIFORM-NOT: ttl.sram_allocation_mode
// UNIFORM-LABEL: func.func @uneven
// UNIFORM-NEXT: %[[LARGE:.*]] = ttl.bind_cb
// UNIFORM-NEXT: %[[SMALL:.*]] = ttl.bind_cb
// UNIFORM: ttl.cb_reserve %[[LARGE]]
// UNIFORM: ttl.cb_reserve %[[SMALL]]
// INDEPENDENT: arena_bytes = 32832 : i64, domain = 0 : i64, node = [0, 0], payload_offset = 64 : i64, payload_present = true
// INDEPENDENT-SAME: arena_bytes = 2112 : i64, domain = 1 : i64, node = [1, 0], payload_offset = 0 : i64, payload_present = false
// INDEPENDENT-SAME: arena_bytes = 32832 : i64, domain = 0 : i64, node = [0, 0], payload_offset = 8 : i64, payload_present = false
// INDEPENDENT-SAME: arena_bytes = 2112 : i64, domain = 1 : i64, node = [1, 0], payload_offset = 64 : i64, payload_present = true
// INDEPENDENT-SAME: ttl.sram_allocation_mode = "per-core"
// INDEPENDENT-LABEL: func.func @uneven
// INDEPENDENT-NEXT: %[[LARGE:.*]] = ttl.bind_cb
// INDEPENDENT-NEXT: %[[SMALL:.*]] = ttl.bind_cb
// INDEPENDENT: ttl.cb_reserve %[[LARGE]]
// INDEPENDENT: ttl.cb_reserve %[[SMALL]]
module attributes {ttl.launch_grid = [2, 1]} {
  func.func @uneven() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %large = ttl.bind_cb {cb_index = 0, block_count = 16} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 16>
    %small = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %column = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first = arith.cmpi eq, %column, %zero : index
    scf.if %first {
      %large_produced = ttl.cb_reserve %large : <[1, 1], !ttcore.tile<32x32, bf16>, 16> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %large : <[1, 1], !ttcore.tile<32x32, bf16>, 16>
      %large_consumed = ttl.cb_wait %large : <[1, 1], !ttcore.tile<32x32, bf16>, 16> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %large : <[1, 1], !ttcore.tile<32x32, bf16>, 16>
    } else {
      %small_produced = ttl.cb_reserve %small : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %small : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %small_consumed = ttl.cb_wait %small : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %small : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}
