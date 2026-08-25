// Tests allocation-group handoff from a reset-terminated producer epoch.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// The first epoch has advanced only the write pointer when the synchronized
// reset establishes canonical state. The second DFB may then use the group.
// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> launch_node=(0,0) epoch_order=[0:0, 1:0]
// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_terminal">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %stale_slot = ttl.cb_reserve %stale
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_terminal">, <kind = data_movement, identity = "reader", operation = "reset_terminal">, <kind = data_movement, identity = "writer", operation = "reset_terminal">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_reserve %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_terminal">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_terminal">, <kind = data_movement, identity = "reader", operation = "reset_terminal">, <kind = data_movement, identity = "writer", operation = "reset_terminal">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_wait %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_terminal">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_terminal">, <kind = data_movement, identity = "reader", operation = "reset_terminal">, <kind = data_movement, identity = "writer", operation = "reset_terminal">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}
