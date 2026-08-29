// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify selected identity predicates preserve one send/post pair per
// device-transfer record through schedule and receiver-address analysis.

// Sender and receiver callbacks describe each record with complementary
// endpoint properties. Per-record analysis resolves both predicates before
// comparing execution counts.

// CHECK-LABEL: func.func @sender
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [1, 0] : index
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @receiver
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 1] : index
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.cb_reserve_back

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "identity_schedule" pipes [
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
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_zero = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %src_one = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      %select_zero = arith.cmpi eq, %destination, %zero : index
      scf.if %select_zero {
        %send = ttl.copy %src_zero, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      } else {
        %send = ttl.copy %src_one, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.yield
    }
    func.return
  }

  func.func @receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst_zero = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_one = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      %select_zero = arith.cmpi eq, %source, %zero : index
      scf.if %select_zero {
        %reserved = ttl.cb_reserve %dst_zero
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %dst_zero : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      } else {
        %reserved = ttl.cb_reserve %dst_one
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %dst_one : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.yield
    }
    func.return
  }
}
