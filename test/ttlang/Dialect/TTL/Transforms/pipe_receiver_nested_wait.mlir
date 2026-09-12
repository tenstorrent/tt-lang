// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verify that a receive wait nested in selected-pipe control precedes the
// enclosing reservation publication.

// CHECK-LABEL: func.func @nested_receive_wait_precedes_push
// CHECK: ttkernel.experimental.semaphore_wait_min

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @nested_receive_wait_precedes_push()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %destination_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %reserved = ttl.cb_reserve %destination_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %token = ttl.pipe_transfer.post %transfer, %reserved
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.cb_push %destination_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %source_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
