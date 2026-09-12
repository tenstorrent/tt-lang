// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-to-ttkernel-pipeline

// A tensor read must observe the receive completion signal.

#layout0 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain0 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer0 = #ttl.device_transfer<
  domain = #domain0,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @read_before_receive_wait(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout0>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %read_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout0>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout0>
      // expected-note @below {{pipe destination is here}}
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout0>)
          -> !ttl.receive_request
      %reserved = ttl.cb_reserve %read_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{reads a pipe destination tensor region before the matching receive wait}}
      %read = ttl.copy %output_slice, %read_dfb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout0>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      ttl.wait %receive : !ttl.receive_request
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// Reusing one static tensor destination would overwrite its first transfer.

#layout5 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain5 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer5 = #ttl.device_transfer<
  domain = #domain5,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @repeated_destination(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout5>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer5}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout5>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout5>
      scf.for %iteration = %zero to %two step %one {
        // expected-error @below {{pipe receive tensor_slice destination currently requires exactly one statically proven transfer occurrence}}
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #layout5>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %send = ttl.copy %send_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    return
  }
}

// -----

// Each receiver-owned tensor region has one pipe producer.

#layout4 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain4 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer4 = #ttl.device_transfer<
  domain = #domain4,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @overlapping_pipe_destinations(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout4>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer4}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1 {
        deviceTransfer = #transfer4}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %zero = arith.constant 0 : index
    %output_slice = ttl.tensor_slice %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout4>
    ttl.if_dst %pipe0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      // expected-note @below {{overlapping destination is here}}
      %receive0 = ttl.copy %pipe0, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout4>)
          -> !ttl.receive_request
      ttl.wait %receive0 : !ttl.receive_request
    }
    ttl.if_dst %pipe1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1> {
      // expected-error @below {{pipe receive tensor_slice overlaps another pipe destination for tensor 0}}
      %receive1 = ttl.copy %pipe1, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout4>)
          -> !ttl.receive_request
      ttl.wait %receive1 : !ttl.receive_request
    }
    ttl.if_src %pipe0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %send0 = ttl.copy %send_dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1> {
      %send1 = ttl.copy %send_dfb, %pipe1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// Sender-side DRAM address generation requires a static destination formula.

#layout2 = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain2 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer2 = #ttl.device_transfer<
  domain = #domain2,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @dynamic_destination_start(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout2>,
      %start: index)
      attributes {
        ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer2}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%start, %zero]
          : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout2>
      // expected-error @below {{pipe receive tensor_slice currently requires static start indices}}
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout2>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// A collective has multiple destinations and therefore cannot use the
// point-to-point ownership proof.

#layout3 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain3 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer3 = #ttl.device_transfer<
  domain = #domain3,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 2]} {
  func.func @collective_tensor_destination(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout3>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 1) net 0 {
        deviceTransfer = #transfer3}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 1) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 1) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout3>
      // expected-error @below {{pipe receive tensor_slice destination currently supports only point-to-point transfers}}
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 1) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout3>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 1) net 0> {
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 1) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// A local write cannot overlap a region owned by a pipe receive.

#layout1 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain1 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer1 = #ttl.device_transfer<
  domain = #domain1,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @overlapping_local_write(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>
      // expected-note @below {{pipe destination is here}}
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      // expected-error @below {{writes a tensor region also owned by a pipe receive destination}}
      %write = ttl.copy %send_dfb, %output_slice
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %write : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// An opaque tensor access has unknown region and access effects.

#layout1 = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain1 = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer1 = #ttl.device_transfer<
  domain = #domain1,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @opaque_access_to_pipe_destination_tensor(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>
      // expected-note @below {{pipe destination is here}}
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      // expected-error @below {{uses a tensor owned by a pipe receive destination through an unsupported operation}}
      ttl.opaque_call "touch_tensor" (%output) {header = "touch_tensor.hpp"}
          : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout1>) -> ()
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}
