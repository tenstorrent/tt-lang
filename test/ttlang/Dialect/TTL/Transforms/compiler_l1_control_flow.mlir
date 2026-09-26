// Branch-exclusive and loop-sequential transactions retain separate storage
// when the completion analysis cannot prove reuse across control flow.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram})' | FileCheck %s

// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: l1_payload_offset = 64 : i64
// CHECK-SAME: l1_payload_offset = 2112 : i64
// CHECK-SAME: ttl.l1_arena_bytes = 4160 : i64
// CHECK-LABEL: func.func @branch
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @branch(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %left = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %right = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %condition {
      %produced = ttl.cb_reserve %left : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %left : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed = ttl.cb_wait %left : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %left : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    } else {
      %produced = ttl.cb_reserve %right : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %right : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed = ttl.cb_wait %right : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %right : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}

// -----

// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: l1_payload_offset = 64 : i64
// CHECK-SAME: l1_payload_offset = 2112 : i64
// CHECK-SAME: ttl.l1_arena_bytes = 4160 : i64
// CHECK-LABEL: func.func @loop
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @loop() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %start = arith.constant 0 : index
    %end = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %start to %end step %step {
      %produced_first = ttl.cb_reserve %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed_first = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %produced_second = ttl.cb_reserve %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed_second = ttl.cb_wait %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}
