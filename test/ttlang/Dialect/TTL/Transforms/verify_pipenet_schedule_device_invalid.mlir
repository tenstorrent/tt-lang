// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-schedule

// Summary: Reject device-specialized PipeNet events whose dynamic occurrence
// counts differ.

// The source executes one send, while the destination executes two posts.

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [3]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_source = ttl.is_device <coordinates = [0]> in #domain : i1
    scf.if %is_source {
      // expected-error @below {{cannot prove a one-to-one synchronization schedule}}
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_destination = ttl.is_device <coordinates = [3]> in #domain : i1
    scf.if %is_destination {
      scf.for %iteration = %c0 to %c2 step %c1 {
        %reserved = ttl.cb_reserve %dst
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %post = ttl.copy %pipe, %reserved
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// A destination range has no single logical-device execution location. Reject
// it until PipeNet lowering expands the scatter into point transfers.

#range_domain = #ttl.device_domain<
    components = <name = "device", extent = [4]>>
#range_transfer = #ttl.device_transfer<
    domain = #range_domain,
    edge = <source = <coordinates = [0]>,
            destinationRange = <lo = <coordinates = [1]>,
                                hi = <coordinates = [4]>>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @range_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #range_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @range_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #range_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{device-range fabric transfers require scatter target lowering}}
    %post = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    func.return
  }
}

// -----

// A pipe selected from different logical-device transfers does not have one
// execution location for schedule analysis.

#device_transfer_0 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [4]>>,
    edge = <source = <coordinates = [0]>,
            destination = <coordinates = [1]>>>
#device_transfer_1 = #ttl.device_transfer<
    domain = <components = <name = "device", extent = [4]>>,
    edge = <source = <coordinates = [2]>,
            destination = <coordinates = [3]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @ambiguous_device_transfer(%condition: i1)
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
    // expected-error @below {{requires every possible pipe definition at this call site to use the same logical-device transfer}}
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }
}
