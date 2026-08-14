// Tests allocation groups whose members alternate across synchronized reset epochs.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// The group alternates A0, B0, A1, B1 across three logical kernels. Each
// reset establishes canonical state before ownership changes between NOC0,
// compute, and NOC1. Native acquire operations in later epochs must retain the
// next same-kind acquire as their ownership boundary.

// CHECK: module attributes {ttl.dfb_allocations = [{block_count = 3 : i32, dfb_index = 0 : i32
// CHECK: %{{.*}} = ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK: %{{.*}} = ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> launch_node=(0,0) epoch_order=[0:0, 1:0, 0:1, 1:1]
// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// REPORT: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "allocation_group_epochs">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = [0 : i32]} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %a0 = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %a1 = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "allocation_group_epochs">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %a0 = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %b0 = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %a1 = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %b1 = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "allocation_group_epochs">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = [1 : i32]} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %b0 = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "allocation_group_epochs">, <kind = data_movement, identity = "reader", operation = "allocation_group_epochs">, <kind = data_movement, identity = "writer", operation = "allocation_group_epochs">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %b1 = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }
}
