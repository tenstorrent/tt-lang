// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' | FileCheck %s --check-prefix=COMPUTED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s --check-prefix=PUBLISHED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' | FileCheck %s --check-prefix=RECEIVER-POST

// Summary: Verifies that the PipeNet options select receiver-published or
// computed addresses and receiver-post or capacity-counter synchronization.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @point_to_point_pipe
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.noc_async_write
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED-NOT: ttkernel.load_from_l1
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @point_to_point_pipe
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.load_from_l1
  // PUBLISHED: ttkernel.noc_async_write
  func.func @point_to_point_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_dst = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %src_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A physical DFB index that changes from tensor-backed to compiler-managed
// storage requires the receiver-published address protocol in every epoch.
module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 1 : i32, page_size = 2048 : i32,
     storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 2048>}]},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 1 : i32, page_size = 2048 : i32}],
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [
      {configurations = [
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
         num_tiles = 1 : i32, page_size = 2048 : i32,
         storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 2048>}]},
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
         entry_reconfiguration = 0 : i64, num_tiles = 1 : i32,
         page_size = 2048 : i32, storage_segments = [{nodes = [[0, 0]]}]}],
       dfb_index = 0 : i32},
      {configurations = [
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
         num_tiles = 1 : i32, page_size = 2048 : i32,
         storage_segments = [{nodes = [[0, 0]]}]}],
       dfb_index = 1 : i32}]},
  ttl.launch_grid = [1, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @compute() attributes {
      ttl.base_cta_index = 3 : i32,
      ttl.crta_indices = [],
      ttl.kernel_thread = #ttkernel.thread<compute>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute_kernel", operation = "reconfigured_receiver">
  } {
    ttl.dfb_reconfiguration <0, participants[
      <kind = compute, identity = "compute_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "reader_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "writer_kernel", operation = "reconfigured_receiver">]>
    return
  }

  // COMPUTED-LABEL: func.func @reconfigured_receiver
  // COMPUTED-NOT: ttl.pipe_computed_address_dfb_indices
  // The first epoch publishes and consumes the tensor-backed reservation.
  // COMPUTED: ttkernel.store_to_l1
  // COMPUTED: ttkernel.load_from_l1
  // COMPUTED: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"
  // The second epoch publishes and consumes the compiler-managed reservation.
  // COMPUTED: ttkernel.store_to_l1
  // COMPUTED: ttkernel.load_from_l1
  // COMPUTED: return
  func.func @reconfigured_receiver(
      %input: tensor<1x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
      attributes {
        ttl.base_cta_index = 3 : i32,
        ttl.crta_indices = [0 : i32, 2 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader_kernel", operation = "reconfigured_receiver">,
        ttl.noc_index = 0 : i32
      } {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> -> !ttl.pipe_transfer
    %scratch = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %send_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %tensor_backed = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

    %tensor_reservation = ttl.cb_reserve %tensor_backed
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %tensor_block = ttl.attach_cb %tensor_reservation, %tensor_backed
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %tensor_token = ttl.pipe_transfer.post %transfer, %tensor_block
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %send_reservation_0 = ttl.cb_reserve %send_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %send_block_0 = ttl.attach_cb %send_reservation_0, %send_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input_row = arith.constant 0 : index
    %input_column_0 = arith.constant 0 : index
    %input_tile_0 = ttl.tensor_slice %input[%input_row, %input_column_0]
        : tensor<1x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %read_0 = ttl.copy %input_tile_0, %send_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> !ttl.transfer_handle<read>
    ttl.wait %read_0 : !ttl.transfer_handle<read>
    ttl.cb_push %send_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %ready_0 = ttl.cb_wait %send_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %send_0 = ttl.pipe_transfer.send %transfer, %send_dfb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_0 : !ttl.transfer_handle<write>
    ttl.cb_pop %send_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipe_transfer.wait %tensor_token : !ttl.pipe_token<net 0>
    ttl.cb_push %tensor_backed : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %tensor_ready = ttl.cb_wait %tensor_backed
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %tensor_backed : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

    ttl.dfb_reconfiguration <0, participants[
      <kind = compute, identity = "compute_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "reader_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "writer_kernel", operation = "reconfigured_receiver">]>

    %scratch_reservation = ttl.cb_reserve %scratch
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scratch_block = ttl.attach_cb %scratch_reservation, %scratch
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scratch_token = ttl.pipe_transfer.post %transfer, %scratch_block
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %send_reservation_1 = ttl.cb_reserve %send_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %send_block_1 = ttl.attach_cb %send_reservation_1, %send_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input_column_1 = arith.constant 1 : index
    %input_tile_1 = ttl.tensor_slice %input[%input_row, %input_column_1]
        : tensor<1x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %read_1 = ttl.copy %input_tile_1, %send_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> !ttl.transfer_handle<read>
    ttl.wait %read_1 : !ttl.transfer_handle<read>
    ttl.cb_push %send_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %ready_1 = ttl.cb_wait %send_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %send_1 = ttl.pipe_transfer.send %transfer, %send_dfb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_1 : !ttl.transfer_handle<write>
    ttl.cb_pop %send_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipe_transfer.wait %scratch_token : !ttl.pipe_token<net 0>
    ttl.cb_push %scratch : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %scratch_ready = ttl.cb_wait %scratch
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_row = arith.constant 0 : index
    %output_column = arith.constant 0 : index
    %output_tile = ttl.tensor_slice %output[%output_row, %output_column]
        : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %write = ttl.copy %scratch, %output_tile
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
           tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
        -> !ttl.transfer_handle<write>
    ttl.wait %write : !ttl.transfer_handle<write>
    ttl.cb_pop %scratch : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @synchronize() attributes {
      ttl.base_cta_index = 3 : i32,
      ttl.crta_indices = [],
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer_kernel", operation = "reconfigured_receiver">,
      ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration <0, participants[
      <kind = compute, identity = "compute_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "reader_kernel", operation = "reconfigured_receiver">,
      <kind = data_movement, identity = "writer_kernel", operation = "reconfigured_receiver">]>
    return
  }
}

// -----

// Tensor-backed receiver storage uses the runtime address published by the
// receiver instead of a separately allocated computed-address base.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @tensor_backed_receiver_uses_published_address
  // COMPUTED-NOT: ttl.pipe_computed_address_dfb_indices
  // COMPUTED: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.load_from_l1
  // COMPUTED: ttkernel.noc_async_write

  // PUBLISHED-LABEL: func.func @tensor_backed_receiver_uses_published_address
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.load_from_l1
  // PUBLISHED: ttkernel.noc_async_write
  func.func @tensor_backed_receiver_uses_published_address(
      %tensor: tensor<1x1x!ttcore.tile<32x32, f32>>)
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Two two-block reservations exactly fill a four-block receiver DFB. The
// second reservation reaches the physical end without advancing past it.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @repeated_reservation_reaches_dfb_end
  // COMPUTED-NOT: ttl.pipe_computed_address_dfb_indices
  // COMPUTED-DAG: %[[COMPUTED_TWO_I32:.*]] = arith.constant 2 : i32
  // COMPUTED-DAG: %[[COMPUTED_TWO:.*]] = arith.constant 2 : index
  // COMPUTED-DAG: %[[COMPUTED_DST:.*]] = ttkernel.get_compile_time_arg_val(1)
  // COMPUTED: scf.for {{.*}} to %[[COMPUTED_TWO]]
  // COMPUTED: ttkernel.cb_reserve_back(%[[COMPUTED_DST]], %[[COMPUTED_TWO_I32]])
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.experimental.semaphore_wait_min
  // COMPUTED-NEXT: ttkernel.cb_push_back(%[[COMPUTED_DST]], %[[COMPUTED_TWO_I32]])
  // COMPUTED: return
  // COMPUTED-LABEL: func.func @repeated_reservation_reaches_dfb_end_sender
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-DAG: %[[BLOCK_COUNT:.*]] = arith.constant 4 : i32
  // COMPUTED-DAG: %[[REPEAT_STRIDE:.*]] = arith.constant 2 : i32
  // COMPUTED-DAG: %[[BLOCK_BYTES:.*]] = arith.constant 4096 : i32
  // COMPUTED: ttkernel.noc_async_write_one_packet_set_state({{.*}}, %[[BLOCK_BYTES]]
  // COMPUTED: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr
  // COMPUTED: %[[SLOT:.*]] = memref.load %[[SLOT_COUNTER:.*]]
  // COMPUTED-NEXT: %[[SLOT_OFFSET:.*]] = arith.muli %[[SLOT]], %[[BLOCK_BYTES]]
  // COMPUTED-NEXT: %[[DST_ADDR:.*]] = arith.addi {{.*}}, %[[SLOT_OFFSET]]
  // COMPUTED-NEXT: %[[ADVANCED_SLOT:.*]] = arith.addi %[[SLOT]], %[[REPEAT_STRIDE]]
  // COMPUTED-NEXT: %[[NEXT_SLOT:.*]] = arith.remui %[[ADVANCED_SLOT]], %[[BLOCK_COUNT]]
  // COMPUTED-NEXT: memref.store %[[NEXT_SLOT]], %[[SLOT_COUNTER]]
  // COMPUTED-NEXT: ttkernel.noc_async_write_one_packet_with_state(%[[SRC_ADDR]], %[[DST_ADDR]]
  // COMPUTED-NOT: ttkernel.load_from_l1
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @repeated_reservation_reaches_dfb_end
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED-DAG: %[[PUBLISHED_TWO_I32:.*]] = arith.constant 2 : i32
  // PUBLISHED-DAG: %[[PUBLISHED_TWO:.*]] = arith.constant 2 : index
  // PUBLISHED-DAG: %[[PUBLISHED_DST:.*]] = ttkernel.get_compile_time_arg_val(1)
  // PUBLISHED: scf.for {{.*}} to %[[PUBLISHED_TWO]]
  // PUBLISHED: ttkernel.cb_reserve_back(%[[PUBLISHED_DST]], %[[PUBLISHED_TWO_I32]])
  // PUBLISHED: %[[PUBLISHED_ADDR:.*]] = ttkernel.get_write_ptr(%[[PUBLISHED_DST]])
  // PUBLISHED: ttkernel.noc_inline_dw_write({{.*}}, %[[PUBLISHED_ADDR]]
  // PUBLISHED: ttkernel.experimental.semaphore_wait_min
  // PUBLISHED-NEXT: ttkernel.cb_push_back(%[[PUBLISHED_DST]], %[[PUBLISHED_TWO_I32]])
  // PUBLISHED: return
  // PUBLISHED-LABEL: func.func @repeated_reservation_reaches_dfb_end_sender
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED-DAG: %[[PUBLISHED_BLOCK_BYTES:.*]] = arith.constant 4096 : i32
  // PUBLISHED: %[[ADDRESS_TABLE:.*]] = ttkernel.get_common_arg_val
  // PUBLISHED-NEXT: %[[ADDRESS_TABLE_PTR:.*]] = ttkernel.reinterpret_cast(%[[ADDRESS_TABLE]])
  // PUBLISHED: ttkernel.noc_async_write_one_packet_set_state({{.*}}, %[[PUBLISHED_BLOCK_BYTES]]
  // PUBLISHED: %[[PUBLISHED_SRC_ADDR:.*]] = ttkernel.get_write_ptr
  // PUBLISHED-NEXT: %[[PUBLISHED_DST_ADDR:.*]] = ttkernel.load_from_l1(%[[ADDRESS_TABLE_PTR]]
  // PUBLISHED-NEXT: ttkernel.noc_async_write_one_packet_with_state(%[[PUBLISHED_SRC_ADDR]], %[[PUBLISHED_DST_ADDR]]
  // PUBLISHED: return
  func.func @repeated_reservation_reaches_dfb_end()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 4} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lb to %ub step %step {
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %slot = tensor.extract_slice %reserved[0, 0] [1, 1] [1, 1]
            : tensor<1x2x!ttcore.tile<32x32, f32>>
              to tensor<1x1x!ttcore.tile<32x32, f32>>
        %receive = ttl.copy %pipe, %slot
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        ttl.cb_push %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
      }
    }
    func.return
  }

  // Match the receiver loop with two sends from the source node.
  func.func @repeated_reservation_reaches_dfb_end_sender()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lb to %ub step %step {
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %send = ttl.copy %src, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// A receiver on the source core publishes its address with a local L1 store;
// an inline NoC write does not update the issuing core's SRAM.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  // COMPUTED-LABEL: func.func @loopback_point_to_point
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-NOT: ttkernel.store_to_l1
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.noc_async_write
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @loopback_point_to_point
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.store_to_l1
  // PUBLISHED-NOT: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.noc_async_write
  // PUBLISHED: return
  func.func @loopback_point_to_point()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %recv_dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %recv = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.receive_request
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}

// -----

// A loopback collective stores locally on the source receiver and uses an
// inline NoC write for every remote receiver.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @loopback_collective
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-NOT: ttkernel.store_to_l1
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.noc_async_write_multicast_loopback_src
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @loopback_collective
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: %[[ZERO:.*]] = arith.constant 0 : index
  // PUBLISHED-NEXT: %[[ZERO_I32:.*]] = arith.constant 0 : i32
  // PUBLISHED-DAG: %[[SOURCE_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated(%[[ZERO]])
  // PUBLISHED-DAG: %[[SOURCE_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated(%[[ZERO]])
  // PUBLISHED-DAG: %[[TABLE_ADDRESS:.*]] = ttkernel.get_common_arg_val(%[[ZERO]])
  // PUBLISHED: %[[CURRENT_X:.*]] = ttkernel.my_logical_x_
  // PUBLISHED-NEXT: %[[CURRENT_Y:.*]] = ttkernel.my_logical_y_
  // PUBLISHED-NEXT: %[[X_MATCHES:.*]] = arith.cmpi eq, %[[CURRENT_X]], %[[ZERO]] : index
  // PUBLISHED-NEXT: %[[Y_MATCHES:.*]] = arith.cmpi eq, %[[CURRENT_Y]], %[[ZERO]] : index
  // PUBLISHED-NEXT: %[[RECEIVER_IS_SOURCE:.*]] = arith.andi %[[X_MATCHES]], %[[Y_MATCHES]] : i1
  // PUBLISHED: %[[TABLE_PTR:.*]] = ttkernel.reinterpret_cast(%[[TABLE_ADDRESS]])
  // PUBLISHED: %[[PUBLISHED_ADDRESS:.*]] = ttkernel.get_write_ptr
  // PUBLISHED-NEXT: scf.if %[[RECEIVER_IS_SOURCE]] {
  // PUBLISHED-NEXT:   ttkernel.store_to_l1(%[[PUBLISHED_ADDRESS]], %[[TABLE_PTR]], %[[ZERO_I32]])
  // PUBLISHED-NEXT: } else {
  // PUBLISHED-NEXT:   ttkernel.noc_inline_dw_write(core[%[[SOURCE_X]], %[[SOURCE_Y]]], %[[TABLE_ADDRESS]], %[[PUBLISHED_ADDRESS]], {{.*}})
  // PUBLISHED-NEXT: }
  // PUBLISHED: ttkernel.noc_async_write_multicast_loopback_src
  // PUBLISHED: return
  func.func @loopback_collective()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
      %recv_dst = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
      %send = ttl.copy %src_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Disabling computed addresses keeps receiver-published multicast available
// when every receiver DFB is proven to advance identically.
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  // COMPUTED-LABEL: func.func @uniform_multicast
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.noc_async_write_multicast
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @uniform_multicast
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.load_from_l1
  // PUBLISHED: ttkernel.noc_async_write_multicast
  // PUBLISHED: return

  // RECEIVER-POST-LABEL: func.func @uniform_multicast
  // RECEIVER-POST-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // RECEIVER-POST-NOT: ttkernel.noc_inline_dw_write
  // RECEIVER-POST: ttkernel.noc_async_write_multicast
  // RECEIVER-POST: return
  func.func @uniform_multicast()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Computed addresses require a receiver DFB stream whose physical ring movement
// is fully modeled by pipe receives. A non-pipe push on the receiver DFB keeps
// the receiver-published address protocol even when computed addresses are
// enabled.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @mixed_receiver_dfb_uses_published_address
  // COMPUTED-NOT: ttl.pipe_computed_address_dfb_indices
  // COMPUTED: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.load_from_l1
  // COMPUTED: ttkernel.noc_async_write
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @mixed_receiver_dfb_uses_published_address
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.load_from_l1
  // PUBLISHED: ttkernel.noc_async_write
  func.func @mixed_receiver_dfb_uses_published_address()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>

    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %local = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>

      %recv_dst = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %src_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// The capacity protocol requires computed addressing, so disabling the option
// also disables capacity: the computed case emits sender-local capacity-counter
// operations, while the published case uses receiver-post synchronization.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @capacity_pipe
  // COMPUTED: ttkernel.experimental.semaphore_wait_min
  // COMPUTED-NOT: ttkernel.store_to_l1
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: return

  // PUBLISHED-LABEL: func.func @capacity_pipe
  // PUBLISHED-NOT: arith.subi
  // PUBLISHED-NOT: ttkernel.store_to_l1
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED-NOT: arith.subi
  // PUBLISHED-NOT: ttkernel.store_to_l1
  // PUBLISHED: return

  // RECEIVER-POST-LABEL: func.func @capacity_pipe
  // RECEIVER-POST-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // Receiver post increments sender-ready before the receiver completion wait.
  // RECEIVER-POST: ttkernel.noc_semaphore_inc
  // RECEIVER-POST: ttkernel.experimental.semaphore_wait_min
  // RECEIVER-POST: ttkernel.cb_push_back
  // RECEIVER-POST: ttkernel.cb_pop_front
  // The pop does not release capacity; the sender consumes the ready post.
  // RECEIVER-POST-NOT: ttkernel.noc_semaphore_inc
  // RECEIVER-POST: ttkernel.experimental.semaphore_wait(
  // RECEIVER-POST: ttkernel.noc_semaphore_set
  func.func @capacity_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
