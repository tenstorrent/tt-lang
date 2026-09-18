// Summary: Verifies PipeGraph distinguishes declared physical aliases from
// logical DFB producer ownership.
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s --check-prefix=LOWERING
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel -debug-only=ttl-pipe-capacity-analysis 2>&1 >/dev/null | FileCheck %s --check-prefix=CAPACITY

// A different logical DFB at the same declared physical index does not have
// allocator proof. Neither computed addressing nor capacity synchronization
// may assume that the foreign and target lifecycles are disjoint.

// LOWERING-LABEL: func.func @unproven_alias
// LOWERING-NOT: ttl.pipe_computed_address_dfb_indices
// LOWERING: ttkernel.load_from_l1
// LOWERING: ttkernel.noc_async_write
// LOWERING: return
// CAPACITY: PipeCapacity: reject src(0, 0) -> receiver(0, 0) DFB 2 at physical index 1 capacity 1: receiver DFB producer stream is not proven pipe-only

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @unproven_alias()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %foreign = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 2 : index}
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

// An unmatched full-ring push on the target logical DFB preserves physical
// phase but invalidates the pipe-only ownership required by capacity sync.

// LOWERING-LABEL: func.func @same_logical_unmatched_push
// LOWERING-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// LOWERING-NOT: ttkernel.load_from_l1
// LOWERING: ttkernel.noc_async_write
// LOWERING: return
// CAPACITY: PipeCapacity: reject src(0, 0) -> receiver(0, 0) DFB 1 capacity 1: receiver DFB producer stream is not proven pipe-only

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @same_logical_unmatched_push()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %local_block = ttl.cb_reserve %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %target : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
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
    %ready0 = ttl.cb_wait %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %ready1 = ttl.cb_wait %target
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    return
  }
}

// -----

// Distinct logical DFBs remain distinct receiver nodes when they share a
// physical index. Both require finalized alias proof before capacity
// synchronization can use either lifecycle.

// CAPACITY: PipeCapacity: 2 receiver DFB node(s), 2 receiver endpoint(s)
// CAPACITY: PipeCapacity: reject src(0, 0) -> receiver(0, 0) DFB 1 at physical index 2 capacity 1: receiver DFB producer stream is not proven pipe-only
// CAPACITY: PipeCapacity: reject src(0, 0) -> receiver(0, 0) DFB 2 capacity 1: receiver DFB producer stream is not proven pipe-only

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @logical_aliases_have_independent_capacity()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source0 = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %source1 = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target0 = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target1 = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %target0_block = ttl.cb_reserve %target0
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    %post0 = ttl.copy %pipe0, %target0_block
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<1x16, bf16>>)
        -> !ttl.receive_request
    %send0 = ttl.copy %source0, %pipe0
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %target0 : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target0_ready = ttl.cb_wait %target0
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target0 : <[1, 1], !ttcore.tile<1x16, bf16>, 1>

    %target1_block = ttl.cb_reserve %target1
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    %post1 = ttl.copy %pipe1, %target1_block
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
           tensor<1x1x!ttcore.tile<1x16, bf16>>)
        -> !ttl.receive_request
    %send1 = ttl.copy %source1, %pipe1
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %target1 : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target1_ready = ttl.cb_wait %target1
        : <[1, 1], !ttcore.tile<1x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    return
  }
}
