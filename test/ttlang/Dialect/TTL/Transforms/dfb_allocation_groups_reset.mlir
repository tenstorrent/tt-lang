// Tests synchronized resets applied to complete allocation groups.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// Resetting one member separates two balanced group-member lifecycles. Both
// logical DFBs receive the same physical index.

// CHECK-LABEL: func.func @reader
// CHECK: %[[SELECTED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK-NEXT: %[[CURRENT:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "grouped_selected_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %current = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %selected_slot = ttl.cb_reserve %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_selected_reset">, <kind = data_movement, identity = "reader", operation = "grouped_selected_reset">, <kind = data_movement, identity = "writer", operation = "grouped_selected_reset">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %current_slot = ttl.cb_reserve %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "grouped_selected_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %current = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %selected_slot = ttl.cb_wait %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_selected_reset">, <kind = data_movement, identity = "reader", operation = "grouped_selected_reset">, <kind = data_movement, identity = "writer", operation = "grouped_selected_reset">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %current_slot = ttl.cb_wait %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "grouped_selected_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_selected_reset">, <kind = data_movement, identity = "reader", operation = "grouped_selected_reset">, <kind = data_movement, identity = "writer", operation = "grouped_selected_reset">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// An all-DFB reset separates two balanced group-member lifecycles.

// CHECK-LABEL: func.func @all_reader
// CHECK: %[[ALL_BEFORE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
// CHECK-NEXT: %[[ALL_AFTER:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @all_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "grouped_all_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %before = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %after = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before_slot = ttl.cb_reserve %before
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %before
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_all_reset">, <kind = data_movement, identity = "reader", operation = "grouped_all_reset">, <kind = data_movement, identity = "writer", operation = "grouped_all_reset">]>
    %after_slot = ttl.cb_reserve %after
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %after
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @all_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "grouped_all_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %before = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %after = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before_slot = ttl.cb_wait %before
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %before
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_all_reset">, <kind = data_movement, identity = "reader", operation = "grouped_all_reset">, <kind = data_movement, identity = "writer", operation = "grouped_all_reset">]>
    %after_slot = ttl.cb_wait %after
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %after
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @all_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "grouped_all_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "grouped_all_reset">, <kind = data_movement, identity = "reader", operation = "grouped_all_reset">, <kind = data_movement, identity = "writer", operation = "grouped_all_reset">]>
    return
  }
}
