// A 2048-byte payload plus the 64-byte control prefix fits exactly in 2112 bytes.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-budget-override=2112},ttl-validate-cb-budget{l1-budget-override=2112})' | FileCheck %s
// CHECK: ttl.l1_arena_bytes = 2112 : i64
// CHECK-LABEL: func.func @boundary
// CHECK-NEXT: %{{.*}} = ttl.bind_cb
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @boundary() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
