// Tensor-backed storage uses only a control record on Wormhole B0.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram})' | FileCheck %s

// CHECK-LABEL: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_capacity_pages = 1 : i32
// CHECK-SAME: storage_segments = [{nodes = [{{.*}}], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]
// CHECK-SAME: ttl.l1_arena_bytes = 32 : i64
// CHECK-SAME: ttl.memory_model = "compiler-sram"

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @tensor_backed()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = [0]} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %block = ttl.cb_reserve %storage
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
