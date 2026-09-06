// Rejects unsupported storage groups and insufficient byte budgets before materialization.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-budget-override=1024})'

// A complete page plus its control record cannot fit a 1024-byte budget.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @insufficient_budget() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 placement exceeds L1 budget 1024 bytes}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Tensor-backed storage has external ownership and cannot use arena offsets.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @unsupported_tensor_backing() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 does not yet support tensor-backed storage or allocation groups}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
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

// Explicit allocation groups cannot establish control-record handoff in this backend.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @unsupported_group() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    // expected-error @below {{compiler-l1 does not yet support tensor-backed storage or allocation groups}}
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
