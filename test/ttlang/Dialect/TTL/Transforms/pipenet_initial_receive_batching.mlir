// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel='pipe-computed-addresses=false' | FileCheck %s --check-prefix=PUBLISHED --implicit-check-not=ttl.pipenet_initial_receive_capacity

// Verify initial-storage and counter-ownership proofs before logical DFB identities are lowered.

// Distinct senders occupy five initially empty receiver slots.
// PUBLISHED-LABEL: func.func @initial_gather_receiver
// CHECK-LABEL: func.func @initial_gather_receiver
// CHECK: ttl.pipenet_initial_receive_capacity = 5 : i64
// CHECK: return
#records = #ttl.pipenet_records<net 0 name "initial_gather" pipes [
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 3, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 4, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 5, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 6, 1>} {
  func.func @initial_gather_sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %write = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %read = ttl.cb_wait %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %send = ttl.copy %source, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    return
  }
  func.func @initial_gather_receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 5} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %write = ttl.cb_reserve %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %write : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5>
      ttl.yield
    }
    return
  }
}

// -----

// A repeated receive sequence has no initial-empty proof for later iterations.
// PUBLISHED-LABEL: func.func @repeated_gather_receiver
// CHECK-LABEL: func.func @repeated_gather_receiver
// CHECK-NOT: ttl.pipenet_initial_receive_capacity
// CHECK: return
#records = #ttl.pipenet_records<net 0 name "repeated_gather" pipes [
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 3, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 4, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 5, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 6, 1>} {
  func.func @repeated_gather_sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %iteration = %zero to %two step %one {
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %write = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %read = ttl.cb_wait %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %send = ttl.copy %source, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    }
    return
  }
  func.func @repeated_gather_receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 5} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %iteration = %zero to %two step %one {
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %write = ttl.cb_reserve %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %write : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5>
      ttl.yield
    }
    }
    return
  }
}

// -----

// Earlier producer traffic prevents a larger initial reservation, even after a full pointer cycle.
// PUBLISHED-LABEL: func.func @prior_payload_receiver
// CHECK-LABEL: func.func @prior_payload_receiver
// CHECK-NOT: ttl.pipenet_initial_receive_capacity
// CHECK: return
#records = #ttl.pipenet_records<net 0 name "prior_payload" pipes [
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 3, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 4, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 5, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 6, 1>} {
  func.func @prior_payload_sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %write = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %read = ttl.cb_wait %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %send = ttl.copy %source, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    return
  }
  func.func @prior_payload_receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 5} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %prior = ttl.cb_reserve %destination {num_tiles = 5 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 5> -> tensor<1x5x!ttcore.tile<32x32, f32>>
    ttl.cb_push %destination {num_tiles = 5 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 5>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %write = ttl.cb_reserve %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %write : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5>
      ttl.yield
    }
    return
  }
}

// -----

// Records from one sender remain batchable when each has its own readiness word.
// PUBLISHED-LABEL: func.func @distinct_sender_counters_receiver
// CHECK-LABEL: func.func @distinct_sender_counters_receiver
// CHECK: %[[READY_INDEX:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [5, 6, 7, 8, 9] : index
// CHECK-NEXT: %[[READY_WORD:.*]] = ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK: %[[READY_ADDRESS:.*]] = ttkernel.get_noc_addr({{.*}}, %[[READY_WORD]], {{.*}})
// CHECK-NEXT: ttkernel.noc_semaphore_inc(%[[READY_ADDRESS]],
// CHECK: ttl.pipenet_initial_receive_capacity = 5 : i64
// CHECK: return
#records = #ttl.pipenet_records<net 0 name "distinct_sender_counters" pipes [
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 6, 1>} {
  func.func @distinct_sender_counters_sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %write = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %read = ttl.cb_wait %source : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %send = ttl.copy %source, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    return
  }
  func.func @distinct_sender_counters_receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 5} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %write = ttl.cb_reserve %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %write : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, f32>, 5>
      ttl.yield
    }
    return
  }
}
