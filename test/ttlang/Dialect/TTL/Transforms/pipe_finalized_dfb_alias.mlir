// Summary: Verifies finalized DFB reuse preserves logical producer ownership
// while accounting for physical write-pointer phase.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-verify-pipenet-schedule,ttl-finalize-dfb-indices{reuse-user-dfbs=true},convert-ttl-to-ttkernel)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-verify-pipenet-schedule,ttl-finalize-dfb-indices{reuse-user-dfbs=true},convert-ttl-to-ttkernel)' -debug-only=ttl-pipe-capacity-analysis 2>&1 >/dev/null | FileCheck %s --check-prefix=CAPACITY

// A full foreign DFB advance returns the shared physical write pointer to its
// original phase. Capacity analysis counts only the target logical DFB pop.

// CHECK-LABEL: func.func @full_ring
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
// CAPACITY: PipeCapacity: accept src(0, 0) -> receiver(0, 0) DFB 2 at physical index 1 capacity 1: sends=1 pops=1

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @full_ring()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 4 : i32,
                  ttl.crta_indices = []} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %foreign = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>

    %foreign_block = ttl.cb_reserve %foreign
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %foreign : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %foreign_ready = ttl.cb_wait %foreign
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %foreign : <[1, 1], !ttcore.tile<1x16, bf16>, 1>

    %target_block = ttl.cb_reserve %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    %post = ttl.copy %pipe, %target_block
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<1x16, bf16>>)
        -> !ttl.receive_request
    %send = ttl.copy %source, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %target : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target_ready = ttl.cb_wait %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    return
  }
}

// -----

// A partial foreign DFB advance changes the shared physical write-pointer
// phase, so the point-to-point transfer publishes its receiver address. The
// target capacity count remains independent of the foreign logical DFB pop.

// CHECK-LABEL: func.func @partial_ring
// CHECK-NOT: ttl.pipe_computed_address_dfb_indices
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write
// CHECK: return
// CAPACITY: PipeCapacity: accept src(0, 0) -> receiver(0, 0) DFB 2 at physical index 1 capacity 2: sends=1 pops=1

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @partial_ring()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 4 : i32,
                  ttl.crta_indices = []} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 3>
    %foreign = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>

    %foreign_block = ttl.cb_reserve %foreign
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %foreign : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %foreign_ready = ttl.cb_wait %foreign
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %foreign : <[1, 1], !ttcore.tile<1x16, bf16>, 2>

    %target_block = ttl.cb_reserve %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    %post = ttl.copy %pipe, %target_block
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<1x16, bf16>>)
        -> !ttl.receive_request
    %send = ttl.copy %source, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 3>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %target : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target_ready = ttl.cb_wait %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
