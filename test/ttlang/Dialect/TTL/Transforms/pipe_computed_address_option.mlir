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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %recv = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
  // COMPUTED: %[[SLOT:.*]] = memref.load %[[SLOT_COUNTER:.*]]
  // COMPUTED-NEXT: %[[SLOT_OFFSET:.*]] = arith.muli %[[SLOT]], %[[BLOCK_BYTES]]
  // COMPUTED-NEXT: %[[DST_ADDR:.*]] = arith.addi {{.*}}, %[[SLOT_OFFSET]]
  // COMPUTED-NEXT: %[[ADVANCED_SLOT:.*]] = arith.addi %[[SLOT]], %[[REPEAT_STRIDE]]
  // COMPUTED-NEXT: %[[NEXT_SLOT:.*]] = arith.remui %[[ADVANCED_SLOT]], %[[BLOCK_COUNT]]
  // COMPUTED-NEXT: memref.store %[[NEXT_SLOT]], %[[SLOT_COUNTER]]
  // COMPUTED-NEXT: ttkernel.noc_async_write {{.*}}, %[[DST_ADDR]], %[[BLOCK_BYTES]]
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
  // PUBLISHED: %[[ADDRESS_TABLE:.*]] = ttkernel.get_common_arg_val
  // PUBLISHED-NEXT: %[[ADDRESS_TABLE_PTR:.*]] = ttkernel.reinterpret_cast(%[[ADDRESS_TABLE]])
  // PUBLISHED-NEXT: %[[PUBLISHED_DST_ADDR:.*]] = ttkernel.load_from_l1(%[[ADDRESS_TABLE_PTR]]
  // PUBLISHED: ttkernel.noc_async_write {{.*}}, %[[PUBLISHED_DST_ADDR]]
  // PUBLISHED: return
  func.func @repeated_reservation_reaches_dfb_end()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 4}
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
            -> !ttl.transfer_handle
        ttl.wait %receive : !ttl.transfer_handle
        ttl.cb_push %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
      }
    }
    func.return
  }

  // Match the receiver loop with two sends from the source node.
  func.func @repeated_reservation_reaches_dfb_end_sender()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %recv_dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %recv = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.transfer_handle
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
  // PUBLISHED-DAG: %[[PUBLISHED_ADDRESS:.*]] = ttkernel.get_write_ptr
  // PUBLISHED-DAG: %[[TABLE_ADDRESS:.*]] = ttkernel.get_common_arg_val(%[[ZERO]])
  // PUBLISHED-DAG: %[[CURRENT_X:.*]] = ttkernel.my_logical_x_
  // PUBLISHED-DAG: %[[CURRENT_Y:.*]] = ttkernel.my_logical_y_
  // PUBLISHED: %[[X_MATCHES:.*]] = arith.cmpi eq, %[[CURRENT_X]], %[[ZERO]] : index
  // PUBLISHED-NEXT: %[[Y_MATCHES:.*]] = arith.cmpi eq, %[[CURRENT_Y]], %[[ZERO]] : index
  // PUBLISHED-NEXT: %[[RECEIVER_IS_SOURCE:.*]] = arith.andi %[[X_MATCHES]], %[[Y_MATCHES]] : i1
  // PUBLISHED-NEXT: scf.if %[[RECEIVER_IS_SOURCE]] {
  // PUBLISHED-NEXT:   %[[TABLE_PTR:.*]] = ttkernel.reinterpret_cast(%[[TABLE_ADDRESS]])
  // PUBLISHED-NEXT:   ttkernel.store_to_l1(%[[PUBLISHED_ADDRESS]], %[[TABLE_PTR]], %[[ZERO_I32]])
  // PUBLISHED-NEXT: } else {
  // PUBLISHED-NEXT:   ttkernel.noc_inline_dw_write(core[%[[SOURCE_X]], %[[SOURCE_Y]]], %[[TABLE_ADDRESS]], %[[PUBLISHED_ADDRESS]], {{.*}})
  // PUBLISHED-NEXT: }
  // PUBLISHED: ttkernel.noc_async_write_multicast_loopback_src
  // PUBLISHED: return
  func.func @loopback_collective()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    %recv_dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %recv = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>

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
        -> !ttl.transfer_handle
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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
