// Tests hard errors for allocation groups expanded by selected DFB resets.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})'

// A selected reset cannot expand to a tensor-backed allocation-group member.

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "tensor_backed_member">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %tensor_backed = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{selected synchronized DFB reset targeting allocation group #ttl.dfb_allocation_group<0> requires scratch-backed members; logical DFB 1 is tensor-backed}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_backed_member">, <kind = data_movement, identity = "reader", operation = "tensor_backed_member">, <kind = data_movement, identity = "writer", operation = "tensor_backed_member">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "tensor_backed_member">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_backed_member">, <kind = data_movement, identity = "reader", operation = "tensor_backed_member">, <kind = data_movement, identity = "writer", operation = "tensor_backed_member">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "tensor_backed_member">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_backed_member">, <kind = data_movement, identity = "reader", operation = "tensor_backed_member">, <kind = data_movement, identity = "writer", operation = "tensor_backed_member">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// An unmatched reserve repeated before each reset overlaps the interval from
// the first reset through the last reset and cannot be assumed safe.

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @repeated_overlap_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "repeated_unproven_overlap">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 8 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %overlapping = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 9 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<4> members=[8, 9] cannot alias logical DFBs 8 and 9: reset-domain-write}}
      %overlapping_slot = ttl.cb_reserve %overlapping
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.reset_dfbs <4, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_overlap">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @repeated_overlap_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "repeated_unproven_overlap">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 8 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <4, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_overlap">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @repeated_overlap_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "repeated_unproven_overlap">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 8 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <4, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_overlap">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_overlap">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }
}

// -----

// A reserve cannot be separated from its matching push by a reset.

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reserve_push_crossing">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<2> members=[4, 5] cannot alias logical DFBs 4 and 5: reset-domain-write}}
    %slot = ttl.cb_reserve %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reserve_push_crossing">, <kind = data_movement, identity = "reader", operation = "reserve_push_crossing">, <kind = data_movement, identity = "writer", operation = "reserve_push_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    ttl.cb_push %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reserve_push_crossing">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reserve_push_crossing">, <kind = data_movement, identity = "reader", operation = "reserve_push_crossing">, <kind = data_movement, identity = "writer", operation = "reserve_push_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reserve_push_crossing">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reserve_push_crossing">, <kind = data_movement, identity = "reader", operation = "reserve_push_crossing">, <kind = data_movement, identity = "writer", operation = "reserve_push_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// A wait cannot be separated from its matching pop by a reset.

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "wait_pop_crossing">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "wait_pop_crossing">, <kind = data_movement, identity = "reader", operation = "wait_pop_crossing">, <kind = data_movement, identity = "writer", operation = "wait_pop_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "wait_pop_crossing">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 7 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<3> members=[6, 7] cannot alias logical DFBs 6 and 7: reset-domain-write}}
    %slot = ttl.cb_wait %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "wait_pop_crossing">, <kind = data_movement, identity = "reader", operation = "wait_pop_crossing">, <kind = data_movement, identity = "writer", operation = "wait_pop_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    ttl.cb_pop %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "wait_pop_crossing">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "wait_pop_crossing">, <kind = data_movement, identity = "reader", operation = "wait_pop_crossing">, <kind = data_movement, identity = "writer", operation = "wait_pop_crossing">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// Independently conditioned producer and consumer activity can cross the
// reset. Failure to prove condition equivalence must remain a hard conflict.

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader(%producer_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "independent_conditions">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %producer_sum = arith.addi %core_x, %producer_offset : index
    %producer_active = arith.cmpi eq, %producer_sum, %zero : index
    scf.if %producer_active {
      %slot = ttl.cb_reserve %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "independent_conditions">, <kind = data_movement, identity = "reader", operation = "independent_conditions">, <kind = data_movement, identity = "writer", operation = "independent_conditions">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @compute(%consumer_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "independent_conditions">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %consumer_sum = arith.addi %core_x, %consumer_offset : index
    %consumer_active = arith.cmpi eq, %consumer_sum, %one : index
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "independent_conditions">, <kind = data_movement, identity = "reader", operation = "independent_conditions">, <kind = data_movement, identity = "writer", operation = "independent_conditions">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    scf.if %consumer_active {
      // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<1> members=[2, 3] cannot alias logical DFBs 2 and 3: reset-domain-write}}
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
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "independent_conditions">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "independent_conditions">, <kind = data_movement, identity = "reader", operation = "independent_conditions">, <kind = data_movement, identity = "writer", operation = "independent_conditions">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
