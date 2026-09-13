// Verifies compiler-SRAM storage ownership for validated allocation groups.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=multi-order-decreasing reuse-user-dfbs=true})' | FileCheck %s

// Sequential scratch members share one control record and the largest payload
// envelope while retaining their logical descriptor geometry.

// CHECK: module attributes {ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32
// CHECK-SAME: l1_allocation_bytes = 8192 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64
// CHECK-SAME: storage_capacity_pages = 4 : i32, storage_index = 0 : i32}, {block_count = 4 : i32, dfb_index = 1 : i32
// CHECK-SAME: l1_allocation_bytes = 8192 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64
// CHECK-SAME: storage_capacity_pages = 4 : i32, storage_index = 0 : i32}], ttl.l1_arena_bytes = 8256 : i64
// CHECK-LABEL: func.func @scratch_capacity_envelope
// CHECK-NEXT: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, block_count = 4}

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @scratch_capacity_envelope()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 4}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %first_producer = ttl.cb_reserve %first
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %first_consumer = ttl.cb_wait %first
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %second_producer = ttl.cb_reserve %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second_consumer = ttl.cb_wait %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}

// -----

// Sequential tensor-backed members share one control record without adding
// payload bytes to the compiler-owned arena.

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_capacity_pages = 2 : i32, storage_index = 0 : i32
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 1 : i32
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_capacity_pages = 2 : i32, storage_index = 0 : i32
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>
// CHECK-SAME: ttl.l1_arena_bytes = 64 : i64
// CHECK-LABEL: func.func @tensor_backed_group
// CHECK-NEXT: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2}

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @tensor_backed_group()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = [0]} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
