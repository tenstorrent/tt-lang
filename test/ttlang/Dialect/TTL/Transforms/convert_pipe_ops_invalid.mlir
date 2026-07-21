// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -convert-ttl-to-ttkernel

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
  // expected-error @below {{untyped transfer handle wait requires every possible source to be the same pipe receive ttl.copy}}
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Two unicast pipes converging on node (1, 0) cannot reuse the same physical
// receiver DFB slot until a receiver pop releases the previous receive.

func.func @gather_block_count_too_small()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>
  %recv1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{gather pipe receiver DFB reuses slot 0 before a receiver pop releases it; add a receiver pop before reusing the DFB slot or increase block_count}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}

// -----

// A multi-block receive cannot wrap around the physical DFB ring. The receiver
// must pop before reusing earlier slots or use a larger block_count.

func.func @gather_receive_span_would_wrap_block_count()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 3}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
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
  %recv2 = ttl.cb_reserve %cb {num_tiles = 2 : i64}
      : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{gather pipe receiver DFB reserve at slot 2 spans 2 block(s), which would wrap block_count=3}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x2x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}

// -----

// A receiver pop cannot release part of a live pipe receive slot. Partial
// release would make the low blocks reusable while the high blocks remain live.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @gather_partial_receiver_pop_rejected()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %cb {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %p, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x2x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    // expected-error @below {{pipe receiver DFB pop releases 1 block(s), but oldest live receive slot spans 2 block(s); receiver pops must release whole DFB slots}}
    ttl.cb_pop %cb {num_tiles = 1 : i64}
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return
  }
}

// -----

// An extra receiver DFB pop after the tracked pipe receive has already been
// released cannot be mapped to a live pipe slot.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @gather_extra_receiver_pop_rejected()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %p, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.cb_pop %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    // expected-error @below {{pipe receiver DFB pop releases 1 block(s), but only 0 live pipe receive block(s) are tracked; receiver pops must release only live pipe receive slots}}
    ttl.cb_pop %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return
  }
}

// -----

// A receiver pop cannot release blocks that are not tracked as live pipe receive
// slots. Otherwise the compiler cannot keep static receiver slots synchronized
// with the hardware DFB ring.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @gather_overlarge_receiver_pop_rejected()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %p, %recv
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    // expected-error @below {{pipe receiver DFB pop releases 2 block(s), but only 1 live pipe receive block(s) are tracked; receiver pops must release only live pipe receive slots}}
    ttl.cb_pop %cb {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return
  }
}

// -----

// A collective pipe cannot publish different receiver DFB slice offsets because
// NoC multicast uses one destination SRAM address for all receivers.

func.func @collective_destination_addresses_differ_by_destination()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  %recv_group = ttl.cb_reserve %cb
      : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv_group[0, 0] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-note @below {{previous collective receive post for this pipe was here}}
  %xf0 = ttl.copy %p, %recv0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf0 : !ttl.transfer_handle
  %recv1 = tensor.extract_slice %recv_group[0, 1] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective pipe receive posts publish different destination addresses; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
  %xf1 = ttl.copy %p, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  func.return
}

// -----

// Collective destination addresses must be statically traceable because NoC
// multicast uses one destination SRAM address for all receivers.

func.func @collective_destination_address_dynamic_offset_rejected(%offset: index)
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  %recv_group = ttl.cb_reserve %cb
      : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %recv = tensor.extract_slice %recv_group[0, %offset] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective pipe destination address could not be determined statically; TT-Metal NoC multicast requires one statically proven destination SRAM address for all receivers}}
  %xf = ttl.copy %p, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Collective receiver posts in mutually exclusive branches do not establish
// one uniform multicast destination slot.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_branch_schedule_rejected()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(2, 0) dst(1, 0) to(2, 0) net 1
        : !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>
    %condition = arith.constant true
    scf.if %condition {
      %recv_a = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{collective pipe receiver posts do not have one proven DFB slot; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
      %post_a = ttl.copy %pipe_a, %recv_a
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
    } else {
      %recv_b = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_b = ttl.copy %pipe_b, %recv_b
          : (!ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
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
        {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
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
      // expected-error @below {{collective pipe receiver DFB write pointers are not proven equal; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
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

// Multicast receiver DFBs must advance by the same number of blocks per loop
// iteration. Otherwise later iterations publish different destination slots.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_nonuniform_receiver_batch_rejected()
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
        {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    %receiver_one_transfer = ttl.pipe_transfer.create %receiver_one
        {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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
        // expected-error @below {{collective pipe receiver DFB write pointers are not proven equal; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
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

// The current lowering has queue depth 1 for each logical pipe. A second
// receive post before the first send would overwrite the sender-visible
// destination address table entry.

func.func @same_pipe_two_posts_before_send_rejected()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p, %recv0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  %recv1 = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one live receive post per pipe before each send}}
  %post1 = ttl.copy %p, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.wait %post1 : !ttl.transfer_handle
  func.return
}

// -----

// Receive-ahead posts in different blocks are rejected because the current
// queue-depth-1 lowering cannot prove that the first post is consumed before
// the second one publishes a new destination address.

func.func @same_pipe_receive_ahead_across_blocks_rejected()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  scf.if %cond {
    %recv1 = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one live receive post per pipe before each send}}
    %token1 = ttl.pipe_transfer.post %transfer, %recv1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
  }
  %send0 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
  func.return
}

// -----

// Two receive posts in one loop-body iteration before the send are rejected:
// both posts are live within a single iteration, exceeding the queue-depth-1
// limit just as two posts in a straight-line block would.

func.func @same_pipe_two_posts_in_loop_body_rejected()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  scf.for %iter = %zero to %one step %one {
    %recv0 = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token0 = ttl.pipe_transfer.post %transfer, %recv0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    %recv1 = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one live receive post per pipe before each send}}
    %token1 = ttl.pipe_transfer.post %transfer, %recv1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  }
  %send0 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  func.return
}

// -----

// Receiver completion still uses local semaphore ids. A PipeNet id above the
// local limit is rejected even when sender-ready counters use GlobalSemaphore
// allocation.

// expected-error @below {{pipe synchronization requires 17 hardware semaphore ids, exceeding TT hardware limit of 16; issue #619 tracks scalable pipe synchronization allocation}}
// expected-note @below {{highest allocated semaphore id is 16 for receiver-completion counter}}
module {
  func.func @unicast_pipe_sync_exceeds_hardware_semaphore_limit()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %p2 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %p3 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
    %p4 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
    %p5 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
    %p6 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
    %p7 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
    %p8 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
    %p9 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
    %p10 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
    %p11 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
    %p12 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
    %p13 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
    %p14 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
    %p15 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
    %p16 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 16
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>
    %recv1 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf1 = ttl.copy %p1, %recv1
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf1 : !ttl.transfer_handle
    %recv2 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf2 = ttl.copy %p2, %recv2
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf2 : !ttl.transfer_handle
    %recv3 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf3 = ttl.copy %p3, %recv3
        : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf3 : !ttl.transfer_handle
    %recv4 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf4 = ttl.copy %p4, %recv4
        : (!ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf4 : !ttl.transfer_handle
    %recv5 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf5 = ttl.copy %p5, %recv5
        : (!ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf5 : !ttl.transfer_handle
    %recv6 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf6 = ttl.copy %p6, %recv6
        : (!ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf6 : !ttl.transfer_handle
    %recv7 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf7 = ttl.copy %p7, %recv7
        : (!ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf7 : !ttl.transfer_handle
    %recv8 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf8 = ttl.copy %p8, %recv8
        : (!ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf8 : !ttl.transfer_handle
    %recv16 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf16 = ttl.copy %p16, %recv16
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf16 : !ttl.transfer_handle
    func.return
  }
}

// -----

// Two collective pipes whose destinations overlap at node (1, 0) cannot reuse
// the same physical receiver DFB slot until a receiver pop releases the
// previous receive.

func.func @collective_overlap_block_count_too_small()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %recv1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective overlap pipe receiver DFB reuses slot 0 before a receiver pop releases it; add a receiver pop before reusing the DFB slot or increase block_count}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}
