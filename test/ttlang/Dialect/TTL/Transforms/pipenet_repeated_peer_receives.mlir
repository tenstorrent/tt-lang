// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s
// RUN: ttlang-opt %s -ttl-verify-pipenet-schedule -o /dev/null

// Repeated incoming peers share callback IR but require distinct post/wait pairs.
// CHECK-LABEL: func.func @sender
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @receiver
// CHECK: ttkernel.cb_reserve_back

#domain = #ttl.device_domain<components = <name = "device", extent = [3]>>
#records = #ttl.pipenet_records<net 0 name "repeated_peers" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [2]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [2]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %rounds = arith.constant 2 : index
    scf.for %round = %zero to %rounds step %one {
      ttl.pipenet_foreach_src attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.yield
      }
    }
    func.return
  }

  func.func @receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %rounds = arith.constant 2 : index
    scf.for %round = %zero to %rounds step %one {
      ttl.pipenet_foreach_dst attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        ttl.yield
      }
    }
    func.return
  }
}
