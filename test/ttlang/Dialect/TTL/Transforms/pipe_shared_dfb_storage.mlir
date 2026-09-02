// Summary: Computed PipeNet addresses require dedicated DFB storage.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-verify-pipenet-schedule,convert-ttl-to-ttkernel{pipe-computed-addresses=true})' | FileCheck %s

// DFBs 1 and 2 use distinct physical descriptors backed by storage 1. The
// receiver therefore publishes its TT-Metal-assigned address instead of using
// a separately allocated computed-address backing tensor.

// CHECK-LABEL: func.func @shared_storage_receiver
// CHECK-NOT: ttl.pipe_computed_address_dfb_indices
// CHECK: ttkernel.store_to_l1
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write
// CHECK: return

module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0]], block_count = 2 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>,
     num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>,
     num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 2 : i32, element_type = !ttcore.tile<1x16, bf16>,
     num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32}],
  ttl.launch_grid = [1, 1]
} {
  func.func @shared_storage_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
