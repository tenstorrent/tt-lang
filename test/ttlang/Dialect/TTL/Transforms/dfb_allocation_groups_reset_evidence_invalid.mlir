// Verifies an assumable domain conflict cannot hide a selected-reset write.
// RUN: ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})'

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_evidence">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %selected_active = arith.cmpi eq, %runtime_sum, %zero : index
    %crossing_active = arith.cmpi eq, %core_x, %one : index
    scf.if %selected_active {
      %slot = ttl.cb_reserve %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.if %crossing_active {
      %slot = ttl.cb_reserve %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_evidence">, <kind = data_movement, identity = "reader", operation = "reset_evidence">, <kind = data_movement, identity = "writer", operation = "reset_evidence">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @compute(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_evidence">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %selected_active = arith.cmpi eq, %runtime_sum, %zero : index
    %crossing_active = arith.cmpi eq, %core_x, %one : index
    scf.if %selected_active {
      %slot = ttl.cb_wait %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_evidence">, <kind = data_movement, identity = "reader", operation = "reset_evidence">, <kind = data_movement, identity = "writer", operation = "reset_evidence">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    scf.if %crossing_active {
      // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] cannot alias logical DFBs 0 and 1: reset-domain-write}}
      %slot = ttl.cb_wait %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_evidence">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_evidence">, <kind = data_movement, identity = "reader", operation = "reset_evidence">, <kind = data_movement, identity = "writer", operation = "reset_evidence">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
