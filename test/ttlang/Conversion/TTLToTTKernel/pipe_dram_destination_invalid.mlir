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

// A same-device graph edge lowers to NoC, which cannot write a DRAM tensor
// destination.

#same_device_input_layout = #ttl.layout<
  shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#same_device_output_layout = #ttl.layout<
  shape = [128, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @same_device_sender(
      %input: tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_input_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "direct_dram" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = <components = <name = "device", extent = [2, 2]>>,
           edge = <source = <coordinates = [0, 0]>,
                   destination = <coordinates = [0, 0]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %reserved = ttl.cb_reserve %send_dfb
          : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %block = ttl.attach_cb %reserved, %send_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %zero = arith.constant 0 : index
      %input_slice = ttl.tensor_slice %input[%zero, %zero]
          : tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_input_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_input_layout>
      %read = ttl.copy %input_slice, %send_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_input_layout>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      // expected-error @below {{computed DRAM pipe destination requires an inter-device transfer}}
      %send = ttl.copy %block, %pipe
          : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }

  func.func @same_device_receiver(
      %output: tensor<4x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 1 : i32
      } {
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "direct_dram" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = <components = <name = "device", extent = [2, 2]>>,
           edge = <source = <coordinates = [0, 0]>,
                   destination = <coordinates = [0, 0]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<4x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.selected_pipe_dst,
             tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      %reserved = ttl.cb_reserve %readback_dfb
          : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %read = ttl.copy %output_slice, %readback_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      %two = arith.constant 2 : index
      %disjoint_slice = ttl.tensor_slice %output[%two, %zero]
          : tensor<4x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>
      %write = ttl.copy %readback_dfb, %disjoint_slice
          : (!ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>,
             tensor<2x2x!ttcore.tile<32x32, bf16>, #same_device_output_layout>)
          -> !ttl.transfer_handle<write>
      ttl.wait %write : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// Receiver-owned DRAM address computation is a fabric transport protocol.

#local_layout = #ttl.layout<
  shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {ttl.launch_grid = [1, 2]} {
  func.func @local_pipe_to_dram(
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>, #local_layout>)
      attributes {
        ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 1) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, bf16>, #local_layout>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #local_layout>
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>, #local_layout>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      // expected-error @below {{computed DRAM pipe destination requires an inter-device transfer}}
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }
}

// -----

// Reusing one static tensor destination without reading each transfer would
// overwrite an unconsumed payload.

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
        // expected-error @below {{repeated pipe receive tensor_slice destination requires exactly one matching read before reuse}}
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

module attributes {ttl.launch_grid = [1, 2]} {
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
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 1) net 1 {
        deviceTransfer = #transfer4}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 1>
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
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 1> {
      // expected-error @below {{pipe receive tensor_slice overlaps another pipe destination for tensor 0}}
      %receive1 = ttl.copy %pipe1, %output_slice
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 1>,
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
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 1> {
      %send1 = ttl.copy %send_dfb, %pipe1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 1>)
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
      // expected-error @below {{pipe receive tensor_slice start index in dimension 0 is not statically enumerable}}
      %output_slice = ttl.tensor_slice %output[%start, %zero]
          : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout2>
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

// -----

// Wait-any completion tracks a reserved DFB block per candidate, so a receive
// into a DRAM tensor region cannot be a wait-any candidate.

#wait_any_input_layout = #ttl.layout<
  shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#wait_any_output_layout = #ttl.layout<
  shape = [128, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @wait_any_dram_sender(
      %input: tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_input_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "direct_dram" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = <components = <name = "device", extent = [2, 2]>>,
           edge = <source = <coordinates = [0, 0]>,
                   destination = <coordinates = [1, 1]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %reserved = ttl.cb_reserve %send_dfb
          : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %block = ttl.attach_cb %reserved, %send_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %zero = arith.constant 0 : index
      %input_slice = ttl.tensor_slice %input[%zero, %zero]
          : tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_input_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_input_layout>
      %read = ttl.copy %input_slice, %send_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_input_layout>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      %send = ttl.copy %block, %pipe
          : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }

  func.func @wait_any_dram_receiver(
      %output: tensor<4x2x!ttcore.tile<32x32, bf16>, #wait_any_output_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 1 : i32
      } {
    ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "direct_dram" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = <components = <name = "device", extent = [2, 2]>>,
           edge = <source = <coordinates = [0, 0]>,
                   destination = <coordinates = [1, 1]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<4x2x!ttcore.tile<32x32, bf16>, #wait_any_output_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_output_layout>
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.selected_pipe_dst,
             tensor<2x2x!ttcore.tile<32x32, bf16>, #wait_any_output_layout>)
          -> !ttl.receive_request
      // expected-error @below {{requires every candidate receive to target a reserved DFB block}}
      %ready = ttl.wait_any %receive start %zero
          : (!ttl.receive_request, index) -> !ttl.ready_receive
      ttl.wait %receive : !ttl.receive_request
    }
    return
  }
}

// -----

// A read in every iteration must follow the receive wait in every iteration.
// The receive executes only in the last iteration, and a no-rendezvous sender
// may write the region while the first iteration reads it.

#layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#forward = #ttl.device_transfer<
  domain = #domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @read_before_receive(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>,
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32, 1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %forward = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #forward}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_device_zero = ttl.is_device <coordinates = [0]> in #domain : i1
    %is_device_one = ttl.is_device <coordinates = [1]> in #domain : i1
    %input_slice = ttl.tensor_slice %input[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %output_slice = ttl.tensor_slice %output[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    scf.for %iteration = %zero to %two step %one {
      %is_last = arith.cmpi eq, %iteration, %one : index
      scf.if %is_device_zero {
        scf.if %is_last {
          %reserved = ttl.cb_reserve %send_dfb
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %read = ttl.copy %input_slice, %send_dfb
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read : !ttl.transfer_handle<read>
          %send = ttl.copy %send_dfb, %forward
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      scf.if %is_device_one {
        scf.if %is_last {
          // expected-note @below {{pipe destination is here}}
          %receive = ttl.copy %forward, %output_slice
              : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>)
              -> !ttl.receive_request
          ttl.wait %receive : !ttl.receive_request
        }
        %readback = ttl.cb_reserve %readback_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-error @below {{reads a pipe destination tensor region before the matching receive wait}}
        %read_output = ttl.copy %output_slice, %readback_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read_output : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// A read that executes only before the receive races with the incoming write,
// although the read and the receive never execute in the same iteration.

#layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#forward = #ttl.device_transfer<
  domain = #domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @read_before_only_receive(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>,
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32, 1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %forward = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #forward}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_device_zero = ttl.is_device <coordinates = [0]> in #domain : i1
    %is_device_one = ttl.is_device <coordinates = [1]> in #domain : i1
    %input_slice = ttl.tensor_slice %input[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %output_slice = ttl.tensor_slice %output[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    scf.for %iteration = %zero to %two step %one {
      %is_last = arith.cmpi eq, %iteration, %one : index
      scf.if %is_device_zero {
        scf.if %is_last {
          %reserved = ttl.cb_reserve %send_dfb
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %read = ttl.copy %input_slice, %send_dfb
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read : !ttl.transfer_handle<read>
          %send = ttl.copy %send_dfb, %forward
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      scf.if %is_device_one {
        scf.if %is_last {
          // expected-note @below {{pipe destination is here}}
          %receive = ttl.copy %forward, %output_slice
              : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>)
              -> !ttl.receive_request
          ttl.wait %receive : !ttl.receive_request
        }
        %is_first = arith.cmpi eq, %iteration, %zero : index
        scf.if %is_first {
          %readback = ttl.cb_reserve %readback_dfb
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          // expected-error @below {{reads a pipe destination tensor region before the matching receive wait}}
          %read_output = ttl.copy %output_slice, %readback_dfb
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read_output : !ttl.transfer_handle<read>
        }
      }
    }
    return
  }
}

// -----

// A slice defined in the loop body still names one region in every iteration,
// so the first-iteration read races with the last-iteration receive.

#layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#forward = #ttl.device_transfer<
  domain = #domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @read_shared_region_in_loop(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>,
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32, 1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %forward = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #forward}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_device_zero = ttl.is_device <coordinates = [0]> in #domain : i1
    %is_device_one = ttl.is_device <coordinates = [1]> in #domain : i1
    %input_slice = ttl.tensor_slice %input[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    scf.for %iteration = %zero to %two step %one {
      %is_last = arith.cmpi eq, %iteration, %one : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<2x1x!ttcore.tile<32x32, bf16>, #layout>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      scf.if %is_device_zero {
        scf.if %is_last {
          %reserved = ttl.cb_reserve %send_dfb
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %read = ttl.copy %input_slice, %send_dfb
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read : !ttl.transfer_handle<read>
          %send = ttl.copy %send_dfb, %forward
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      scf.if %is_device_one {
        scf.if %is_last {
          // expected-note @below {{pipe destination is here}}
          %receive = ttl.copy %forward, %output_slice
              : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>)
              -> !ttl.receive_request
          ttl.wait %receive : !ttl.receive_request
        }
        %readback = ttl.cb_reserve %readback_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-error @below {{reads a pipe destination tensor region before the matching receive wait}}
        %read_output = ttl.copy %output_slice, %readback_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read_output : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// A receive loop with more iterations than the enumeration bound is rejected
// before its occurrences are enumerated.

#bound_layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#bound_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#bound_transfer = #ttl.device_transfer<
  domain = #bound_domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 2], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @over_bound_sender(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #bound_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 1) net 0 {
        deviceTransfer = #bound_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>
    %zero = arith.constant 0 : index
    %count = arith.constant 1048577 : index
    %one = arith.constant 1 : index
    %input_slice = ttl.tensor_slice %input[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #bound_layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #bound_layout>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      scf.for %iteration = %zero to %count step %one {
        %reserved = ttl.cb_reserve %send_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %input_slice, %send_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #bound_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        %send = ttl.copy %send_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    return
  }

  func.func @over_bound_receiver(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #bound_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 1 : i32
      } {
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 1) net 0 {
        deviceTransfer = #bound_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>
    %zero = arith.constant 0 : index
    %count = arith.constant 1048577 : index
    %one = arith.constant 1 : index
    %core_y = ttl.core_y : index
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      scf.for %iteration = %zero to %count step %one {
        // expected-error @below {{pipe receive tensor_slice enumeration supports at most 1048576 loop iterations and executions}}
        %output_slice = ttl.tensor_slice %output[%core_y, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #bound_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #bound_layout>
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #bound_layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %readback_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %output_slice, %readback_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #bound_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// A receive loop whose trip count is static but whose bounds depend on a
// runtime value cannot be enumerated.

#runtime_layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#runtime_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#runtime_transfer = #ttl.device_transfer<
  domain = #runtime_domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @runtime_loop_bounds(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #runtime_layout>,
      %start: index)
      attributes {
        ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #runtime_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %end = arith.addi %start, %two overflow<nsw> : index
      scf.for %iteration = %start to %end step %one {
        // expected-error @below {{pipe receive tensor_slice requires statically enumerable enclosing loops}}
        %output_slice = ttl.tensor_slice %output[%zero, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #runtime_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #runtime_layout>
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #runtime_layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %end = arith.addi %start, %two overflow<nsw> : index
      scf.for %iteration = %start to %end step %one {
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
