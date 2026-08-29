// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-verify-pipenet-schedule,convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' | FileCheck %s

// Summary: Verifies computed receiver addressing when runtime control is
// bounded by PipeNet roles and shared by the complete transfer protocol.

// The runtime flag controls matching source and destination protocol
// occurrences. The receiver recurrence remains valid for either flag value.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // CHECK-LABEL: func.func @matching_runtime_receiver_context
  // CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // CHECK-NOT: ttkernel.noc_inline_dw_write
  // CHECK: ttkernel.noc_async_write
  // CHECK-NOT: ttkernel.noc_inline_dw_write
  // CHECK: return
  func.func @matching_runtime_receiver_context(%runtime_y: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %node_y = ttl.core_y : index
    %runtime_selected = arith.cmpi eq, %node_y, %runtime_y : index
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.if %runtime_selected {
        %recv = ttl.cb_reserve %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %receive = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.if %runtime_selected {
        %send = ttl.copy %src_cb, %pipe
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

// Record-selected source and destination callbacks retain the same bounded
// runtime condition after the record-loop factor is removed from each count.
#selected_domain = #ttl.device_domain<
    components = <name = "device", extent = [2]>>
#selected_records = #ttl.pipenet_records<net 0 name "bounded_selected" pipes [
  #ttl.pipe_record<
      srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 1,
      dstEndX = 0, dstEndY = 1,
      deviceTransfer = <
        domain = #selected_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 2, 2>} {
  // CHECK-LABEL: func.func @matching_selected_runtime_receiver_context
  // CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // CHECK-NOT: ttkernel.noc_inline_dw_write
  // CHECK: ttkernel.routing_plane.fused_write_atomic_inc
  // CHECK-NOT: ttkernel.noc_inline_dw_write
  // CHECK: return
  func.func @matching_selected_runtime_receiver_context(%runtime_y: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.pipenet_foreach_dst attributes {
        records = #selected_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source_x, %source_y = ttl.selected_pipe_source_coordinates %pipe
          : !ttl.selected_pipe_dst
      %source_sum = arith.addi %source_x, %source_y : index
      %destination_selected = arith.cmpi eq, %source_sum, %runtime_y : index
      scf.if %destination_selected {
        %recv = ttl.cb_reserve %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %receive = ttl.copy %pipe, %recv
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.yield
    }
    ttl.pipenet_foreach_src attributes {
        records = #selected_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination_x, %destination_y, %destination_end_x, %destination_end_y =
          ttl.selected_pipe_destination_coordinates %pipe
              : !ttl.selected_pipe_src
      %destination_sum = arith.addi %destination_x, %destination_y : index
      %source_selected = arith.cmpi eq, %destination_sum, %runtime_y : index
      scf.if %source_selected {
        %send = ttl.copy %src_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.yield
    }
    func.return
  }
}
