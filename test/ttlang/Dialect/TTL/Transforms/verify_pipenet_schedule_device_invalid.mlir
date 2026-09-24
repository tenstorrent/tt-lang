// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-schedule

// Summary: Reject device-specialized PipeNet events whose dynamic occurrence
// counts differ or whose fabric receiver posts form a wait-for cycle.

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
        // expected-error @below {{PipeNet net_0 requires one static receiver post definition for each static send definition at receiver core_x=0, core_y=0; found 2 static receiver post definition(s) and 1 static send definition(s)}}
        // expected-note @below {{this receiver post has no corresponding send}}
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

// -----

// Each device sends before it posts its own receive. Every iteration reuses
// one DRAM tile, so each fabric send waits for the peer's receiver post, which
// follows the peer's own send.

#layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#forward = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>
#backward = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [1]>, destination = <coordinates = [0]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @reused_dram_exchange(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>)
      attributes {
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %forward = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #forward}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %backward = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1 {
        deviceTransfer = #backward}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_device_zero = ttl.is_device <coordinates = [0]> in #domain : i1
    %is_device_one = ttl.is_device <coordinates = [1]> in #domain : i1
    scf.for %iteration = %zero to %two step %one {
      %slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      scf.if %is_device_zero {
        // expected-error @below {{pipe schedule contains a wait-for cycle on PipeNet net_0}}
        // expected-note @below {{sender waits for receiver post at core_x=0, core_y=0 before send at core_x=0, core_y=0}}
        %send = ttl.copy %src, %forward
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        // expected-note @below {{program order requires receiver post at core_x=0, core_y=0 after send at core_x=0, core_y=0}}
        %receive = ttl.copy %backward, %slice
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
      scf.if %is_device_one {
        // expected-note @below {{sender waits for receiver post at core_x=0, core_y=0 before send at core_x=0, core_y=0}}
        %send = ttl.copy %src, %backward
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        // expected-note @below {{program order requires receiver post at core_x=0, core_y=0 after send at core_x=0, core_y=0}}
        %receive = ttl.copy %forward, %slice
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// The all-gather relay reuses one DRAM tile for each half. Each fabric send
// then waits for the receiver post, and row 0 posts the right half only after
// row 1 of its device posts the left multicast, which follows row 1's
// previous fabric send of the right half.

#layout = #ttl.layout<
  shape = [128, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
!output = tensor<4x1x!ttcore.tile<32x32, bf16>, #layout>
!slice = tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
!block = tensor<1x1x!ttcore.tile<32x32, bf16>>
!dfb = !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
!left_pipe = !ttl.pipe<src(0, 0) dst(0, 1) to(0, 2) net 3>
!right_pipe = !ttl.pipe<src(0, 0) dst(0, 1) to(0, 2) net 4>
#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#ring_left = #ttl.pipenet_records<net 1 name "ring_left" pipes [
  #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
          edge = <source = <coordinates = [0]>,
                  destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
          edge = <source = <coordinates = [1]>,
                  destination = <coordinates = [0]>>>>
]>
#ring_right = #ttl.pipenet_records<net 2 name "ring_right" pipes [
  #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
          edge = <source = <coordinates = [0]>,
                  destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <domain = #domain,
          edge = <source = <coordinates = [1]>,
                  destination = <coordinates = [0]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 3>} {
  func.func @reused_dram_relay(
      %output: !output)
      attributes {
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %local_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !dfb
    %row_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !dfb
    %relay_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !dfb
    %left = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 2) net 3
        : !left_pipe
    %right = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 2) net 4
        : !right_pipe
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_src attributes {records = #ring_left} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %local_dfb, %pipe
          : (!dfb, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.pipenet_foreach_src attributes {records = #ring_right} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %local_dfb, %pipe
          : (!dfb, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    scf.for %iteration = %zero to %two step %one {
      %left_slice = ttl.tensor_slice %output[%zero, %zero]
          : !output -> !slice
      %right_slice = ttl.tensor_slice %output[%one, %zero]
          : !output -> !slice
      %forward = arith.cmpi ult, %iteration, %one : index
      ttl.pipenet_foreach_dst attributes {records = #ring_left} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        %receive = ttl.copy %pipe, %left_slice
            : (!ttl.selected_pipe_dst, !slice) -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %row_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> !block
        %read = ttl.copy %left_slice, %row_dfb
            : (!slice, !dfb) -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.yield
      }
      ttl.if_src %left : !left_pipe {
        // expected-error @below {{pipe schedule contains a wait-for cycle on PipeNet net_3}}
        // expected-note @below {{sender waits for receiver post at core_x=0, core_y=1 before send at core_x=0, core_y=0}}
        %send = ttl.copy %row_dfb, %left
            : (!dfb, !left_pipe)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.pipenet_foreach_dst attributes {records = #ring_right} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        // expected-note @below {{program order requires receiver post at core_x=0, core_y=0 after send at core_x=0, core_y=0}}
        %receive = ttl.copy %pipe, %right_slice
            : (!ttl.selected_pipe_dst, !slice) -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %row_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> !block
        %read = ttl.copy %right_slice, %row_dfb
            : (!slice, !dfb) -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.yield
      }
      ttl.if_src %right : !right_pipe {
        %send = ttl.copy %row_dfb, %right
            : (!dfb, !right_pipe)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %left : !left_pipe {
        %reserved = ttl.cb_reserve %relay_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> !block
        // expected-note @below {{program order requires receiver post at core_x=0, core_y=1 after send at core_x=0, core_y=1}}
        %receive = ttl.copy %left, %reserved
            : (!left_pipe, !block)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
      scf.if %forward {
        ttl.pipenet_foreach_src attributes {records = #ring_left} {
        ^bb0(%pipe: !ttl.selected_pipe_src):
          %send = ttl.copy %relay_dfb, %pipe
              : (!dfb, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
          ttl.yield
        }
      }
      ttl.if_dst %right : !right_pipe {
        %reserved = ttl.cb_reserve %relay_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> !block
        %receive = ttl.copy %right, %reserved
            : (!right_pipe, !block)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
      scf.if %forward {
        ttl.pipenet_foreach_src attributes {records = #ring_right} {
        ^bb0(%pipe: !ttl.selected_pipe_src):
          // expected-note @below {{sender waits for receiver post at core_x=0, core_y=0 before send at core_x=0, core_y=1}}
          %send = ttl.copy %relay_dfb, %pipe
              : (!dfb, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
          ttl.yield
        }
      }
    }
    func.return
  }
}
