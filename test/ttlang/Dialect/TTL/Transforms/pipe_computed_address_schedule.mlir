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

// Pipe A occurs once and uses slot 0 at every receiver. Receivers 1 and 2 also
// reserve slot 1 for Pipe B, but that later reservation cannot change Pipe A's
// one-element address sequence.

// CHECK-LABEL: func.func @partial_overlap_one_shot_computes_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-COUNT-2: ttkernel.noc_async_write_multicast
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.remui
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 6, 1>} {
  func.func @partial_overlap_one_shot_computes_addresses() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(1, 0) to(4, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(4, 0) net 0>
    %pipe_b = ttl.create_pipe src(5, 0) dst(1, 0) to(2, 0) net 1
        : !ttl.pipe<src(5, 0) dst(1, 0) to(2, 0) net 1>
    ttl.if_dst %pipe_a : !ttl.pipe<src(0, 0) dst(1, 0) to(4, 0) net 0> {
      %recv_a = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_a = ttl.copy %pipe_a, %recv_a
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(4, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_a : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_dst %pipe_b : !ttl.pipe<src(5, 0) dst(1, 0) to(2, 0) net 1> {
      %recv_b = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_b = ttl.copy %pipe_b, %recv_b
          : (!ttl.pipe<src(5, 0) dst(1, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_b : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe_a : !ttl.pipe<src(0, 0) dst(1, 0) to(4, 0) net 0> {
      %send_a = ttl.copy %src, %pipe_a
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(4, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_a : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe_b : !ttl.pipe<src(5, 0) dst(1, 0) to(2, 0) net 1> {
      %send_b = ttl.copy %src, %pipe_b
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(5, 0) dst(1, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_b : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A consumer thread can pop one block between two multicast receives without
// changing their address sequence. Both sends reuse the computed
// block-zero address at every receiver.
// Address setup may be hoisted, but each write must use its computed address.

// CHECK-LABEL: func.func @multicast_post
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[MULTI_ZERO:.*]] = arith.constant 0 : index
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK-DAG: %[[MULTI_DST_A:.*]] = ttkernel.get_common_arg_val(%[[MULTI_ZERO]])
// CHECK-DAG: %[[MULTI_DST_B:.*]] = ttkernel.get_common_arg_val(%[[MULTI_ZERO]])
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK: ttkernel.noc_async_write_multicast({{.*}}, %[[MULTI_DST_A]], noc {{.*}})
// CHECK-NOT: ttkernel.noc_async_write_multicast
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK: ttkernel.noc_async_write_multicast({{.*}}, %[[MULTI_DST_B]], noc {{.*}})
// CHECK-NOT: ttkernel.noc_async_write_multicast
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK-LABEL: func.func @multicast_consume
// CHECK: %[[MULTI_ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[MULTI_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: scf.if
// CHECK-NEXT: ttkernel.cb_wait_front(%[[MULTI_DFB]], %[[MULTI_ONE]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[MULTI_DFB]], %[[MULTI_ONE]])
// CHECK-NEXT: ttkernel.cb_wait_front(%[[MULTI_DFB]], %[[MULTI_ONE]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[MULTI_DFB]], %[[MULTI_ONE]])
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
      %recv_again = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_again = ttl.copy %pipe, %recv_again
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_again : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %send_again = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_again : !ttl.transfer_handle<write>
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
      %ready_again = ttl.cb_wait %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}

// -----

// Two sequential producer reservations can reuse one DFB block even when a
// different kernel thread pops the first block. Both senders must use the
// block-zero computed address without receiver address publication.
// Address setup may be hoisted, but each write must use its computed address.

// CHECK-LABEL: func.func @cross_thread_single_slot_posts
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NOT: ttkernel.noc_async_write {{.*}}
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK: %[[DST_A:.*]] = ttkernel.get_common_arg_val(%[[ZERO]])
// CHECK: ttkernel.noc_async_write %{{.*}}, core[{{.*}}], %[[DST_A]], %{{.*}}, noc %{{.*}}
// CHECK-NOT: ttkernel.noc_async_write {{.*}}
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK: %[[DST_B:.*]] = ttkernel.get_common_arg_val(%[[ZERO]])
// CHECK: ttkernel.noc_async_write %{{.*}}, core[{{.*}}], %[[DST_B]], %{{.*}}, noc %{{.*}}
// CHECK-NOT: ttkernel.noc_async_write {{.*}}
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: arith.muli
// CHECK-LABEL: func.func @cross_thread_single_slot_consumer
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: ttkernel.cb_wait_front(%[[DST_DFB]], %[[ONE]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[DST_DFB]], %[[ONE]])
// CHECK-NEXT: ttkernel.cb_wait_front(%[[DST_DFB]], %[[ONE]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[DST_DFB]], %[[ONE]])
// CHECK-NEXT: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @cross_thread_single_slot_posts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe_a = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
    %pipe_b = ttl.create_pipe src(2, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(2, 0) dst(0, 0) to(0, 0) net 0>
    ttl.if_dst %pipe_a : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0> {
      %recv_a = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_a = ttl.copy %pipe_a, %recv_a
          : (!ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_a : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_dst %pipe_b : !ttl.pipe<src(2, 0) dst(0, 0) to(0, 0) net 0> {
      %recv_b = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_b = ttl.copy %pipe_b, %recv_b
          : (!ttl.pipe<src(2, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_b : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe_a : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0> {
      %send_a = ttl.copy %src, %pipe_a
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_a : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe_b : !ttl.pipe<src(2, 0) dst(0, 0) to(0, 0) net 0> {
      %send_b = ttl.copy %src, %pipe_b
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(2, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_b : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @cross_thread_single_slot_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %ready_a = ttl.cb_wait %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %ready_b = ttl.cb_wait %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}

// -----

// Mutually exclusive branches do not establish an order between receiver
// posts, even when their regions have a fixed lexical order.

// CHECK-LABEL: func.func @branch_order_falls_back(%{{.*}}: i1) attributes {ttkernel.thread = #ttkernel.thread<noc>} {
// CHECK-COUNT-2: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @branch_order_falls_back(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
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

// Distinct transfer definitions for one PipeKey receive distinct initial slots
// and therefore use independent computed-address state.

// CHECK-LABEL: func.func @repeated_pipe_key_computes_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @repeated_pipe_key_computes_addresses() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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

// Posts in one loop body execute in a fixed per-iteration order. The two posts
// advance the DFB by two slots per iteration, which wraps a two-slot DFB.

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

// An unknown loop trip count still permits computed addressing when every
// iteration follows the same receiver reservation recurrence.

// CHECK-LABEL: func.func @invariant_dynamic_loop_computes_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: arith.remui
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @invariant_dynamic_loop_computes_addresses(%upper: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iter = %zero to %upper step %one {
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post : !ttl.transfer_handle
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
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
