// Verifies that external tensor payloads consume only compiler control storage.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=multi-order-decreasing})' | FileCheck %s

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes =
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_capacity_pages = 2 : i32
// CHECK-SAME: storage_segments = [{nodes = [{{.*}}], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 4096>}]
// CHECK-SAME: ttl.l1_arena_bytes = 64 : i64
// CHECK-SAME: ttl.memory_model = "compiler-l1"
// CHECK-LABEL: func.func @tensor_backed
// CHECK-NEXT: %[[STORAGE:.*]] = ttl.bind_cb
// CHECK-SAME: cb_index = 0
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 4096>
// CHECK-NEXT: %[[BLOCK:.*]] = ttl.cb_reserve %[[STORAGE]]
// CHECK-NEXT: ttl.cb_push %[[STORAGE]]
// CHECK-NEXT: return

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @tensor_backed()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = [0]} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %storage
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
