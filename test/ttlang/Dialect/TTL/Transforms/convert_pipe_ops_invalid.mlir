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
  // expected-error @below {{requires either every possible source to be the same pipe receive ttl.copy or no source to be a pipe receive}}
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Two unicast pipes converging on node (1, 0) need distinct slots in the
// receiver DFB. With block_count=1 the second pipe's assigned slot exceeds the
// DFB capacity.

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
  // expected-error @below {{gather pipe receiver DFB has block_count=1 but slot 1 is assigned to this pipe; block_count must be >= 2}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
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
  // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one outstanding receiver post per pipe}}
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
    // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one outstanding receiver post per pipe}}
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
    // expected-error @below {{pipe transfer for pipe net 0 src(0, 0) dst(1, 0) to(1, 0) requires queue depth greater than 1; current lowering supports one outstanding receiver post per pipe}}
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

// Seventeen pipe endpoint relations share one receiver and therefore require
// distinct local completion counters. Sender-ready counters can use global
// storage, but completion counter 16 exceeds the local semaphore limit.

// expected-error @below {{pipe synchronization requires 17 hardware semaphore ids, exceeding TT hardware limit of 16; issue #619 tracks scalable pipe synchronization allocation}}
// expected-note @below {{highest allocated semaphore id is 16 for receiver-completion counter}}
module {
  func.func @completion_counters_exceed_hardware_semaphore_limit()
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 17}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 17>
    %dst = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 17>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %pipe_0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer_0 = ttl.pipe_transfer.create %pipe_0 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %token_0 = ttl.pipe_transfer.post %transfer_0, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    %pipe_1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %transfer_1 = ttl.pipe_transfer.create %pipe_1 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
        -> !ttl.pipe_transfer
    %token_1 = ttl.pipe_transfer.post %transfer_1, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 1>
    %pipe_2 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 2
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2>
    %transfer_2 = ttl.pipe_transfer.create %pipe_2 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2>
        -> !ttl.pipe_transfer
    %token_2 = ttl.pipe_transfer.post %transfer_2, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 2>
    %pipe_3 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 3
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 3>
    %transfer_3 = ttl.pipe_transfer.create %pipe_3 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 3>
        -> !ttl.pipe_transfer
    %token_3 = ttl.pipe_transfer.post %transfer_3, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 3>
    %pipe_4 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 4
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 4>
    %transfer_4 = ttl.pipe_transfer.create %pipe_4 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 4>
        -> !ttl.pipe_transfer
    %token_4 = ttl.pipe_transfer.post %transfer_4, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 4>
    %pipe_5 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 5
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 5>
    %transfer_5 = ttl.pipe_transfer.create %pipe_5 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 5>
        -> !ttl.pipe_transfer
    %token_5 = ttl.pipe_transfer.post %transfer_5, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 5>
    %pipe_6 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 6
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 6>
    %transfer_6 = ttl.pipe_transfer.create %pipe_6 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 6>
        -> !ttl.pipe_transfer
    %token_6 = ttl.pipe_transfer.post %transfer_6, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 6>
    %pipe_7 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 7
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 7>
    %transfer_7 = ttl.pipe_transfer.create %pipe_7 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 7>
        -> !ttl.pipe_transfer
    %token_7 = ttl.pipe_transfer.post %transfer_7, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 7>
    %pipe_8 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 8
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 8>
    %transfer_8 = ttl.pipe_transfer.create %pipe_8 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 8>
        -> !ttl.pipe_transfer
    %token_8 = ttl.pipe_transfer.post %transfer_8, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 8>
    %pipe_9 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 9
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 9>
    %transfer_9 = ttl.pipe_transfer.create %pipe_9 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 9>
        -> !ttl.pipe_transfer
    %token_9 = ttl.pipe_transfer.post %transfer_9, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 9>
    %pipe_10 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 10
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 10>
    %transfer_10 = ttl.pipe_transfer.create %pipe_10 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 10>
        -> !ttl.pipe_transfer
    %token_10 = ttl.pipe_transfer.post %transfer_10, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 10>
    %pipe_11 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 11
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 11>
    %transfer_11 = ttl.pipe_transfer.create %pipe_11 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 11>
        -> !ttl.pipe_transfer
    %token_11 = ttl.pipe_transfer.post %transfer_11, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 11>
    %pipe_12 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 12
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 12>
    %transfer_12 = ttl.pipe_transfer.create %pipe_12 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 12>
        -> !ttl.pipe_transfer
    %token_12 = ttl.pipe_transfer.post %transfer_12, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 12>
    %pipe_13 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 13
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 13>
    %transfer_13 = ttl.pipe_transfer.create %pipe_13 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 13>
        -> !ttl.pipe_transfer
    %token_13 = ttl.pipe_transfer.post %transfer_13, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 13>
    %pipe_14 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 14
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>
    %transfer_14 = ttl.pipe_transfer.create %pipe_14 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>
        -> !ttl.pipe_transfer
    %token_14 = ttl.pipe_transfer.post %transfer_14, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 14>
    %pipe_15 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
    %transfer_15 = ttl.pipe_transfer.create %pipe_15 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
        -> !ttl.pipe_transfer
    %token_15 = ttl.pipe_transfer.post %transfer_15, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 15>
    %pipe_16 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 16
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 16>
    %transfer_16 = ttl.pipe_transfer.create %pipe_16 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 16>
        -> !ttl.pipe_transfer
    %token_16 = ttl.pipe_transfer.post %transfer_16, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 16>
    func.return
  }
}

// -----

// Two collective pipes whose destinations overlap at node (1, 0) each need a
// distinct slot in the receiver DFB. With block_count=1 the second pipe's
// assigned slot (1) exceeds the DFB capacity.

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
  // expected-error @below {{collective overlap pipe receiver DFB has block_count=1 but slot 1 is assigned to this pipe; block_count must be >= 2}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}

// -----

// A merged internal token must retain one transfer creation so resource
// planning cannot select state from an unrelated transfer.

func.func @merged_token_requires_one_transfer_creation(%condition: i1)
    attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %pipe0 {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %transfer1 = ttl.pipe_transfer.create %pipe1 {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %token = scf.if %condition -> (!ttl.pipe_token<net 0>) {
    %dst0 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token0 = ttl.pipe_transfer.post %transfer0, %dst0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %token0 : !ttl.pipe_token<net 0>
  } else {
    %dst1 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token1 = ttl.pipe_transfer.post %transfer1, %dst1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %token1 : !ttl.pipe_token<net 0>
  }
  // expected-error @below {{'ttl.pipe_transfer.wait' op requires all possible receive posts to derive from one ttl.pipe_transfer.create}}
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}
