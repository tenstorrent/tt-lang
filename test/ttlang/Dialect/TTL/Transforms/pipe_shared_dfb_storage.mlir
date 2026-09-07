// Summary: Verifies computed PipeNet addressing for shared DFB storage.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-verify-pipenet-schedule,convert-ttl-to-ttkernel{pipe-computed-addresses=true})' | FileCheck %s

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

// -----

// Compiler-managed indices retain fixed bases when an allocation group shares
// one storage owner, so the sender can compute the receiver address.

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

// CHECK-LABEL: func.func @compiler_l1_shared_storage_receiver
// CHECK-SAME: ttl.fabric_routes = [
// CHECK-SAME: ttl.fabric_runtime_arg_base_common_index = 3 : i64
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: return

module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0]], block_count = 2 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>,
     l1_allocation_bytes = 64 : i64, l1_offset = 0 : i64,
     l1_payload_offset = 64 : i64, num_tiles = 1 : i32,
     page_size = 32 : i32, storage_capacity_pages = 2 : i32,
     storage_index = 0 : i32},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>,
     l1_allocation_bytes = 32 : i64, l1_offset = 8 : i64,
     l1_payload_offset = 128 : i64, num_tiles = 1 : i32,
     page_size = 32 : i32, storage_capacity_pages = 1 : i32,
     storage_index = 1 : i32},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 2 : i32, element_type = !ttcore.tile<1x16, bf16>,
     l1_allocation_bytes = 32 : i64, l1_offset = 8 : i64,
     l1_payload_offset = 128 : i64, num_tiles = 1 : i32,
     page_size = 32 : i32, storage_capacity_pages = 1 : i32,
     storage_index = 1 : i32}],
  ttl.l1_arena_bytes = 160 : i64,
  ttl.launch_grid = [1, 1],
  ttl.memory_model = "compiler-l1",
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @compiler_l1_shared_storage_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %foreign = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %target = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {deviceTransfer = #transfer}
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
