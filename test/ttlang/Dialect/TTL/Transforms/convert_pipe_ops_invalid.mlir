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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{requires a consistent transfer contract for all possible pipe values}}
  %xf = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  func.return
}

// -----

// Pipe transfer expansion rejects a pipe value whose possible definitions
// identify different logical-device transfers.

#device_transfer_0 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 1]>>>
#device_transfer_1 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 2]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @conflicting_pipe_device_transfers(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #device_transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #device_transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe = scf.if %condition
        -> (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
      scf.yield %pipe0
          : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    } else {
      scf.yield %pipe1
          : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    }
    // expected-error @below {{requires every possible pipe definition to use the same logical-device transfer}}
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// A helper cannot encode one logical-device transfer when its callers supply
// pipes from different device edges.

#callsite_transfer_0 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 1]>>>
#callsite_transfer_1 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 2]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func private @pipe_callsite_receiver(
      %pipe: !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
    %dst_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{requires every possible pipe definition to use the same logical-device transfer}}
    %handle = ttl.copy %pipe, %dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    func.return
  }

  func.func @pipe_callsite_caller()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #callsite_transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #callsite_transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    func.call @pipe_callsite_receiver(%pipe0)
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
    func.call @pipe_callsite_receiver(%pipe1)
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
    func.return
  }
}

// -----

// Fabric cannot use receiver-published addresses. Report the specific DFB
// producer-stream property that prevented computed addressing.

#fabric_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#fabric_transfer = #ttl.device_transfer<
    domain = #fabric_domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @fabric_requires_pipe_only_receiver_stream()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 {
        deviceTransfer = #fabric_transfer}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>

    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %local = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>

      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-note @below {{receiver DFB 1: non-pipe push does not advance one full physical DFB}}
      %post = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{fabric pipe transfer requires computed receiver DFB addresses}}
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Role predicates must reference a PipeNet declared by high-level pipe IR.

func.func @unknown_role_predicate()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{references unknown PipeNet 7}}
  %is_src = ttl.is_src {pipe_net_id = 7 : i64}
  scf.if %is_src {
    scf.yield
  }
  func.return
}

// -----

// A send and its corresponding receiver post must identify the same
// logical-device transfer.

#send_device_transfer = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 1]>>>
#post_device_transfer = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [1, 4]>>,
    edge = <source = <coordinates = [0, 0]>,
            destination = <coordinates = [0, 2]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @conflicting_device_transfers()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %send_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #send_device_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %post_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #post_device_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_transfer = ttl.pipe_transfer.create %send_pipe {
        deviceTransfer = #send_device_transfer,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
        -> !ttl.pipe_transfer
    %post_transfer = ttl.pipe_transfer.create %post_pipe {
        deviceTransfer = #post_device_transfer,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
        -> !ttl.pipe_transfer
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe receiver post has no corresponding send for its device transfer}}
    %post = ttl.pipe_transfer.post %post_transfer, %reserved
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    // expected-note @below {{this send has the same PipeKey but a different device transfer}}
    %send = ttl.pipe_transfer.send %send_transfer, %src
        : (!ttl.pipe_transfer,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// An untyped wait cannot select between two distinct pipe receive operations.

func.func @wait_with_distinct_pipe_receive_sources(%condition: i1)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst0 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf0 = ttl.copy %pipe, %dst0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %dst1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %pipe, %dst1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %xf = scf.if %condition -> (!ttl.receive_request) {
    scf.yield %xf0 : !ttl.receive_request
  } else {
    scf.yield %xf1 : !ttl.receive_request
  }
  // expected-error @below {{requires either every possible source to be the same pipe receive ttl.copy or no source to be a pipe receive}}
  ttl.wait %xf : !ttl.receive_request
  func.return
}

// -----

// PipeNet address analysis represents receiver reservations in whole DFB
// blocks and must reject a partial block instead of rounding its span up.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @partial_receiver_block_rejected()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
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
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
  %src = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
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
      -> !ttl.receive_request
  ttl.wait %xf1 : !ttl.receive_request
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
      -> !ttl.receive_request
  %send2 = ttl.copy %src, %p2
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  ttl.wait %xf2 : !ttl.receive_request
  func.return
}

// -----

// A loop recurrence must validate every executed receiver reservation, not only
// the first reservation represented by the static post operation.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @repeated_receiver_reservation_past_dfb_end()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
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
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        ttl.cb_push %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      }
    }
    func.return
  }

  func.func @repeated_receiver_reservation_past_dfb_end_sender()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
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
  %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
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
      // expected-note @below {{receiver core_x=1, core_y=0 uses receiver DFB 1: post is not consumed by a receiver push}}
      %post = ttl.copy %p, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
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
          -> !ttl.receive_request
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

// Matched sender and receiver definitions must describe the same number of
// original DFB blocks.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_pipe_transfer_block_spans()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_transfer = ttl.pipe_transfer.create %pipe {
        block_span = 2 : i64,
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %post_transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{pipe send and receiver post use different transfer block spans}}
      %token = ttl.pipe_transfer.post %post_transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send uses block_span=2}}
      %send = ttl.pipe_transfer.send %send_transfer, %src
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Matched sender and receiver definitions must select the same number of
// destination transfer groups.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_pipe_transfer_destination_group_depths()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_transfer = ttl.pipe_transfer.create %pipe {
        destination_group_depth = 2 : i64,
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %post_transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{pipe send and receiver post use different destination group depths}}
      %token = ttl.pipe_transfer.post %post_transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send uses destination_group_depth=2}}
      %send = ttl.pipe_transfer.send %send_transfer, %src
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
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
          -> !ttl.receive_request
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
    %src = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_a = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(2, 0) dst(1, 0) to(2, 0) net 1
        : !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>
    scf.if %condition {
      ttl.if_dst %pipe_a : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
        %recv_a = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        // expected-note @below {{receiver core_x=1, core_y=0 uses receiver DFB 0: post is not consumed by a receiver push}}
        %post_a = ttl.copy %pipe_a, %recv_a
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
      }
      ttl.if_src %pipe_a : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
        // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence; TT-Metal NoC multicast requires one destination SRAM address for all receivers}}
        %send_a = ttl.copy %src, %pipe_a
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
            -> !ttl.transfer_handle<write>
      }
    } else {
      ttl.if_dst %pipe_b : !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1> {
        %recv_b = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post_b = ttl.copy %pipe_b, %recv_b
            : (!ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
      }
      ttl.if_src %pipe_b : !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1> {
        %send_b = ttl.copy %src, %pipe_b
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(2, 0) dst(1, 0) to(2, 0) net 1>)
            -> !ttl.transfer_handle<write>
      }
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
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
      // expected-note @below {{receiver core_x=1, core_y=0 uses receiver DFB 1: non-pipe push does not advance one full physical DFB}}
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{pipe receiver destination 'tensor<1x1x!ttcore.tile<32x32, bf16>>' cannot hold sender DFB block with 1 element(s) of type '!ttcore.tile<32x32, f32>'}}
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send is here}}
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A pipe receiver destination must contain at least the number of elements
// written from the sender DFB block.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @pipe_payload_exceeds_receiver_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{pipe receiver destination 'tensor<1x1x!ttcore.tile<32x32, f32>>' cannot hold sender DFB block with 2 element(s) of type '!ttcore.tile<32x32, f32>'}}
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send is here}}
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Pipe payload writes require a tile element type because the NoC transfer size
// is derived from the tile storage size.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @pipe_payload_requires_tile_element_type()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], f32, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], f32, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], f32, 1> -> tensor<1x1xf32>
      %post = ttl.copy %pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1xf32>)
          -> !ttl.receive_request
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{pipe transfer source DFB element type must be tile}}
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], f32, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Two publications that can execute together cannot consume one receive post.
module attributes {ttl.launch_grid = array<i64: 2, 2>} {
  func.func @coexecuting_receiver_publications(%condition0: i1,
                                               %condition1: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %landing = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 1) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      %block = ttl.cb_reserve %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-note @below {{receiver core_x=1, core_y=0 uses receiver DFB 0: post is consumed by multiple co-executing pushes}}
      %request = ttl.copy %pipe, %block
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %request : !ttl.receive_request
      scf.if %condition0 {
        ttl.cb_push %landing
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      scf.if %condition1 {
        ttl.cb_push %landing
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
    }
    func.return
  }

  func.func @coexecuting_receiver_publications_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 1) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      // expected-error @below {{collective pipe receiver address sequences are not proven equal for every transfer occurrence}}
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
