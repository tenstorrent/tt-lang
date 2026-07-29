// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -convert-ttl-to-ttkernel
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})'

// Summary: Negative tests for pipe value provenance, receiver DFB validation,
// and synchronization resource diagnostics in ttl-convert-ttl-to-ttkernel.

// Pipe values with conflicting transfer contracts cannot select one lowering
// protocol safely.

func.func @conflicting_pipe_transfer_contracts(%condition: i1)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %point_to_point = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %collective = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 {
      isCollective = true}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %pipe = scf.if %condition
      -> (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    scf.yield %point_to_point
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  } else {
    scf.yield %collective
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  }
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{requires a consistent transfer contract for all possible pipe values}}
  %xf = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  func.return
}

// -----

// An untyped wait cannot select between two distinct pipe receive operations.

func.func @wait_with_distinct_pipe_receive_sources(%condition: i1)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst0 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf0 = ttl.copy %pipe, %dst0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  %dst1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %pipe, %dst1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  %xf = scf.if %condition -> (!ttl.transfer_handle) {
    scf.yield %xf0 : !ttl.transfer_handle
  } else {
    scf.yield %xf1 : !ttl.transfer_handle
  }
  // expected-error @below {{requires either every possible source to be the same pipe receive ttl.copy or no source to be a pipe receive}}
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// PipeNet address analysis represents receiver reservations in whole DFB
// blocks and must reject a partial block instead of rounding its span up.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @partial_receiver_block_rejected()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst {num_tiles = 2 : i64}
          : <[4, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<2x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{PipeNet receiver DFB reserve must contain a whole number of DFB blocks; reserve contains 2 tile(s), but each DFB block contains 4 tile(s)}}
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<2x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A receiver reservation may reach the physical DFB end exactly, but it cannot
// advance the TT-Metal write pointer past that end.

func.func @gather_receiver_reservation_past_dfb_end()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 3}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
  %src = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>
  %recv1 = ttl.cb_reserve %cb {num_tiles = 2 : i64}
      : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x2x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %send1 = ttl.copy %src, %p1
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  %recv2 = ttl.cb_reserve %cb {num_tiles = 2 : i64}
      : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{pipe receiver DFB reservation sequence reaches slot 2 with a span of 2 blocks, which advances the DFB producer write pointer past block_count=3; increase block_count or change the reservation sizes}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x2x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  %send2 = ttl.copy %src, %p2
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}

// -----

// A loop recurrence must validate every executed receiver reservation, not only
// the first reservation represented by the static post operation.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @repeated_receiver_reservation_past_dfb_end()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 3}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lb to %ub step %step {
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 3>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %slot = tensor.extract_slice %reserved[0, 0] [1, 1] [1, 1]
            : tensor<1x2x!ttcore.tile<32x32, f32>>
              to tensor<1x1x!ttcore.tile<32x32, f32>>
        // expected-error @below {{pipe receiver DFB reservation sequence reaches slot 2 with a span of 2 blocks, which advances the DFB producer write pointer past block_count=3; increase block_count or change the reservation sizes}}
        %receive = ttl.copy %pipe, %slot
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %receive : !ttl.transfer_handle
        ttl.cb_push %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      }
    }
    func.return
  }

  func.func @repeated_receiver_reservation_past_dfb_end_sender()
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

// A completion wait cannot select between independent transfer completion
// semaphores.
func.func @pipe_wait_requires_one_static_post(%condition: i1)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %src = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %reserved0 = ttl.cb_reserve %dst
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer, %reserved0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  %send0 = ttl.pipe_transfer.send %transfer, %src
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  %reserved1 = ttl.cb_reserve %dst
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer, %reserved1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer, %src
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  %token = scf.if %condition -> (!ttl.pipe_token<net 0>) {
    scf.yield %token0 : !ttl.pipe_token<net 0>
  } else {
    scf.yield %token1 : !ttl.pipe_token<net 0>
  }
  // expected-error @below {{requires exactly one possible receiver post; found 2}}
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// A collective pipe cannot publish different receiver DFB slice offsets because
// NoC multicast uses one destination SRAM address for all receivers.

#receiverOne = affine_set<(d0) : (d0 == 1)>
#receiverTwo = affine_set<(d0) : (d0 == 2)>
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_destination_addresses_differ_by_destination()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %core_x = ttl.core_x : index
    affine.if #receiverOne(%core_x) {
      %recv_group = ttl.cb_reserve %dst
          : <[1, 2], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %recv = tensor.extract_slice %recv_group[0, 0] [1, 1] [1, 1]
          : tensor<1x2x!ttcore.tile<32x32, f32>>
          to tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-note @below {{receiver core_x=1, core_y=0 uses DFB 1: post is not consumed by a receiver push}}
      %post = ttl.copy %p, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
    }
    affine.if #receiverTwo(%core_x) {
      %recv_group = ttl.cb_reserve %dst
          : <[1, 2], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %recv = tensor.extract_slice %recv_group[0, 1] [1, 1] [1, 1]
          : tensor<1x2x!ttcore.tile<32x32, f32>>
          to tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %p, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
    }
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
      %send = ttl.copy %src, %p
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Collective destination addresses must be statically traceable because NoC
// multicast uses one destination SRAM address for all receivers.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_destination_address_dynamic_offset_rejected(
      %offset: index) attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv_group = ttl.cb_reserve %dst
          : <[1, 2], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %recv = tensor.extract_slice %recv_group[0, %offset] [1, 1] [1, 1]
          : tensor<1x2x!ttcore.tile<32x32, f32>>
          to tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{collective pipe destination address could not be determined statically; TT-Metal NoC multicast requires one statically proven destination SRAM address for all receivers}}
      %post = ttl.copy %p, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
    }
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.copy %src, %p
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Collective receiver posts in different dynamic branches do not establish
// one address sequence shared by every receiver.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_branch_schedule_rejected(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(2, 0) dst(1, 0) to(2, 0) net 1
        : !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>
    scf.if %condition {
      %recv_a = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-note @below {{receiver core_x=1, core_y=0 uses DFB 0: post is not consumed by a receiver push}}
      %post_a = ttl.copy %pipe_a, %recv_a
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
      %send_a = ttl.copy %src, %pipe_a
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
    } else {
      %recv_b = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_b = ttl.copy %pipe_b, %recv_b
          : (!ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      %send_b = ttl.copy %src, %pipe_b
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Multicast cannot use a receiver-published address when one receiver can
// advance its destination DFB independently. The published addresses can then
// differ even though every receiver uses the same DFB index and tile offset.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_asymmetric_non_pipe_traffic_rejected()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %collective = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %receiver_one = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %transfer = ttl.pipe_transfer.create %collective
        {kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %receiver_one
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
      %local = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %local_ready = ttl.cb_wait %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_dst %collective
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-note @below {{receiver core_x=1, core_y=0 uses DFB 1: push reserve owns no matching receiver post}}
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %collective
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
      %send = ttl.pipe_transfer.send %transfer, %src
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// The collective uses slot 0 at both receivers on its first occurrence. On its
// second occurrence, receiver 1 wraps to slot 0 after an intervening unicast
// reservation while receiver 2 advances to slot 1.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_nonuniform_receiver_sequence_rejected()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %collective = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %receiver_one = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %collective_transfer = ttl.pipe_transfer.create %collective
        {kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    %receiver_one_transfer = ttl.pipe_transfer.create %receiver_one
        {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> -> !ttl.pipe_transfer
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.if_dst %collective
          : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
        %recv = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %collective_transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %collective
          : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
        // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
        %send = ttl.pipe_transfer.send %collective_transfer, %src
            : (!ttl.pipe_transfer,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %receiver_one
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
        %recv = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %receiver_one_transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.pipe_token<net 1>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 1>
        ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %receiver_one
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
        %send = ttl.pipe_transfer.send %receiver_one_transfer, %src
            : (!ttl.pipe_transfer,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// A pipe receiver must use the sender payload element type. Pipe values do not
// carry this type, so transfer correspondence validates it before lowering.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @pipe_payload_element_type_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{pipe receiver destination 'tensor<1x1x!ttcore.tile<32x32, bf16>>' cannot hold sender DFB block with 1 element(s) of type '!ttcore.tile<32x32, f32>'}}
    %post = ttl.copy %pipe, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // expected-note @below {{corresponding pipe send is here}}
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// A pipe receiver destination must contain at least the number of elements
// written from the sender DFB block.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @pipe_payload_exceeds_receiver_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe receiver destination 'tensor<1x1x!ttcore.tile<32x32, f32>>' cannot hold sender DFB block with 2 element(s) of type '!ttcore.tile<32x32, f32>'}}
    %post = ttl.copy %pipe, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    // expected-note @below {{corresponding pipe send is here}}
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// Pipe payload writes require a tile element type because the NoC transfer size
// is derived from the tile storage size.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @pipe_payload_requires_tile_element_type()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], f32, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], f32, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %dst
        : <[1, 1], f32, 1> -> tensor<1x1xf32>
    %post = ttl.copy %pipe, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1xf32>)
        -> !ttl.transfer_handle
    // expected-error @below {{pipe transfer source DFB element type must be tile}}
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], f32, 1>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}
