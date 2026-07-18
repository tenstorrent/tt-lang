// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies that computed receiver addresses are used only when one
// receiver control context proves the complete DFB reservation sequence.

// A post inside a loop can execute more times than a post after the loop.
// Lexical occurrence count therefore cannot assign their receiver slots.

// CHECK-LABEL: func.func @loop_multiplicity_falls_back() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK-COUNT-2: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @loop_multiplicity_falls_back() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %pipe_a = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    scf.for %iter = %zero to %two step %one {
      ttl.if_dst %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %recv_a = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_a = ttl.copy %pipe_a, %recv_a
            : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post_a : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
      }
      ttl.if_src %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %send_a = ttl.copy %src, %pipe_a
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send_a : !ttl.transfer_handle<write>
      }
    }
    ttl.if_dst %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %recv_b = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_b = ttl.copy %pipe_b, %recv_b
          : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_b : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    }
    ttl.if_src %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %send_b = ttl.copy %src, %pipe_b
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_b : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A consumer thread can pop the receiver DFB without changing the reservation
// order established by the data-movement thread. The cross-thread pop must not
// disable the uniform computed address required by multicast.

// CHECK-LABEL: func.func @multicast_post
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast
// CHECK-LABEL: func.func @multicast_consume
// CHECK: ttkernel.cb_pop_front
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @multicast_post() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @multicast_consume() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %ready = ttl.cb_wait %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}

// -----

// Mutually exclusive branches do not establish an order between receiver
// posts, even when their regions have a fixed lexical order.

// CHECK-LABEL: func.func @branch_order_falls_back() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK-COUNT-2: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @branch_order_falls_back() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %condition = arith.constant true
    scf.if %condition {
      ttl.if_dst %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %recv_a = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_a = ttl.copy %pipe_a, %recv_a
            : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post_a : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %send_a = ttl.copy %src, %pipe_a
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send_a : !ttl.transfer_handle<write>
      }
    } else {
      ttl.if_dst %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
        %recv_b = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_b = ttl.copy %pipe_b, %recv_b
            : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post_b : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
        %send_b = ttl.copy %src, %pipe_b
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send_b : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// Two static occurrences of one PipeKey cannot be represented by one initial
// receiver slot and one batch stride. Every writer to the shared DFB falls
// back, including the following distinct PipeKey.

// CHECK-LABEL: func.func @repeated_pipe_key_falls_back() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK-COUNT-3: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @repeated_pipe_key_falls_back() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %pipe_a0 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_a1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %recv_a0 = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post_a0 = ttl.copy %pipe_a0, %recv_a0
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send_a0 = ttl.copy %src, %pipe_a0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_a0 : !ttl.transfer_handle<write>
    ttl.wait %post_a0 : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    %recv_a1 = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post_a1 = ttl.copy %pipe_a1, %recv_a1
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send_a1 = ttl.copy %src, %pipe_a1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_a1 : !ttl.transfer_handle<write>
    ttl.wait %post_a1 : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    %recv_b = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post_b = ttl.copy %pipe_b, %recv_b
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send_b = ttl.copy %src, %pipe_b
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_b : !ttl.transfer_handle<write>
    ttl.wait %post_b : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    func.return
  }
}

// -----

// Posts in one loop body execute in a fixed per-iteration order. The common
// control context proves the two-slot receiver batch.

// CHECK-LABEL: func.func @common_loop_computes_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @common_loop_computes_addresses() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    scf.for %iter = %zero to %two step %one {
      ttl.if_dst %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %recv_a = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_a = ttl.copy %pipe_a, %recv_a
            : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post_a : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
        %recv_b = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_b = ttl.copy %pipe_b, %recv_b
            : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post_b : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %pipe_a : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %send_a = ttl.copy %src, %pipe_a
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send_a : !ttl.transfer_handle<write>
      }
      ttl.if_src %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
        %send_b = ttl.copy %src, %pipe_b
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send_b : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// Receiver posts in different data-movement functions have no common program
// order. Reusing one physical DFB index therefore requires address publication.

// CHECK-LABEL: func.func @receiver_a() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK: ttkernel.noc_inline_dw_write
// CHECK-LABEL: func.func @receiver_b() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK: ttkernel.noc_inline_dw_write
// CHECK-LABEL: func.func @sender_a() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK: ttkernel.load_from_l1
// CHECK-LABEL: func.func @sender_b() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @receiver_a() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }

  func.func @receiver_b() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    ttl.if_dst %pipe : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }

  func.func @sender_a() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @sender_b() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    ttl.if_src %pipe : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
