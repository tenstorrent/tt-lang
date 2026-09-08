// Rejects unsupported storage groups and insufficient byte budgets before materialization.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-budget-override=1024 reuse-user-dfbs=false})'

// A complete page plus its control record cannot fit a 1024-byte budget.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @insufficient_budget() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 placement exceeds SRAM budget 1024 bytes}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Partially overlapping external ranges do not define one exact alias.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @partial_tensor_backing_overlap() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{compiler-l1 tensor-backed DFB byte ranges partially overlap on a shared launch node}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 4096>}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %first_block = ttl.cb_reserve %first
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %second_block = ttl.cb_reserve %second
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Tensor-backed storage requires at least one exact launch node.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @empty_tensor_backing(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 tensor backing requires an exact non-empty launch-node domain}}
    %tensor_backed_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 2 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %one : index
    %inactive_condition = arith.andi %runtime_condition, %outside_grid : i1
    scf.if %inactive_condition {
      ttl.opaque_call "inactive_tensor_access" (%tensor_backed_dfb)
          {header = "inactive_tensor_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Identical tensor ranges cannot back distinct storage with overlapping lifetimes.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @duplicate_tensor_range() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{compiler-l1 identical tensor-backed DFB ranges have overlapping lifetimes on a shared launch node}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_block = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_block = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Runtime and generated-kernel fields cannot represent this page count.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @unrepresentable_page_count() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 storage size is not representable}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[2147483648, 1], !ttcore.tile<1x16, bf16>, 1>
    return
  }
}

// -----

// Allocation groups require storage reuse because their members share one owner.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @unsupported_group() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{DFB allocation groups require user DFB reuse to be enabled}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
