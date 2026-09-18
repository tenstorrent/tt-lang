// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify nested graph callbacks restrict receiver address sequencing
// to selected-record combinations that execute on the same logical device.

// Each device selects one outer destination record and the same inner record.
// The receiver therefore posts once per edge and needs one DFB slot.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-LABEL: func.func @senders
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-NOT: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @receivers
// CHECK: ttkernel.cb_reserve_back
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: %[[MANAGER:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[CONNECTION_COUNT:.*]] = ttkernel.routing_plane.open_connections %[[MANAGER]],
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc(%[[MANAGER]], %[[CONNECTION_COUNT]],
// CHECK: ttkernel.routing_plane.close_connections(%[[MANAGER]],
// CHECK-NEXT: ttkernel.noc_semaphore_set
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK: return

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "nested_graph" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @senders() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @receivers()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%outer_pipe: !ttl.selected_pipe_dst):
      ttl.pipenet_foreach_dst attributes {records = #records} {
      ^bb0(%inner_pipe: !ttl.selected_pipe_dst):
        %reserved = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %inner_pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        ttl.yield
      }
      ttl.yield
    }
    func.return
  }
}
