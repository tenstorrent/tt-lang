// A pipe send of a waited block reads through the read pointer, also when a
// later reservation of the same DFB is published before the send.
// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// CHECK-LABEL: func.func @sender
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_wait_front(%[[SRC_DFB]]
// CHECK: ttkernel.cb_reserve_back(%[[SRC_DFB]]
// CHECK: ttkernel.cb_push_back(%[[SRC_DFB]]
// CHECK-NOT: ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.get_read_ptr(%[[SRC_DFB]])
// CHECK-NOT: ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.cb_pop_front(%[[SRC_DFB]]
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %current = ttl.cb_wait %src
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %next = ttl.cb_reserve %src
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %src : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.cb_pop %src : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }

  func.func @receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }
}
