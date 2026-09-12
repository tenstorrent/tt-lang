// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s

// A multi-page PipeNet receive targeting receiver-owned interleaved DRAM uses
// scatter packets followed by one completion increment.

// CHECK-LABEL: func.func @sender_node
// CHECK-SAME: ttl.crta_indices = [0 : i32, 1 : i32]
// CHECK: ttkernel.TensorAccessorArgs{{.*}}get_tensor_accessor_args_cta_offset<1, 2>()
// CHECK: %[[DESTINATION0:.*]] = ttkernel.tensor_accessor.get_noc_addr
// CHECK: %[[DESTINATION1:.*]] = ttkernel.tensor_accessor.get_noc_addr
// CHECK: %[[DESTINATION2:.*]] = ttkernel.tensor_accessor.get_noc_addr
// CHECK: %[[DESTINATION3:.*]] = ttkernel.tensor_accessor.get_noc_addr
// CHECK: %[[SOURCE:.*]] = ttkernel.get_read_ptr
// CHECK-NEXT: ttkernel.routing_plane.scatter_write
// CHECK-SAME: %[[SOURCE]]
// CHECK-SAME: %[[DESTINATION0]]
// CHECK-SAME: %[[DESTINATION1]]
// CHECK-SAME: %[[DESTINATION2]]
// CHECK-SAME: %[[DESTINATION3]]
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc
// CHECK-NOT: ttl.pipe_transfer

// CHECK-LABEL: func.func @receiver_node
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NOT: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-NOT: ttl.pipe_transfer

#dram_layout = #ttl.layout<
  shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#output_layout = #ttl.layout<
  shape = [128, 64], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @sender_node(
      %input: tensor<2x2x!ttcore.tile<32x32, bf16>, #dram_layout>)
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
          : tensor<2x2x!ttcore.tile<32x32, bf16>, #dram_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #dram_layout>
      %read = ttl.copy %input_slice, %send_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>, #dram_layout>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      %send = ttl.copy %send_dfb, %pipe
          : (!ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    return
  }

  func.func @receiver_node(
      %output: tensor<4x2x!ttcore.tile<32x32, bf16>, #output_layout>)
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
                   destination = <coordinates = [1, 1]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %zero = arith.constant 0 : index
      %output_slice = ttl.tensor_slice %output[%zero, %zero]
          : tensor<4x2x!ttcore.tile<32x32, bf16>, #output_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #output_layout>
      %receive = ttl.copy %pipe, %output_slice
          : (!ttl.selected_pipe_dst,
             tensor<2x2x!ttcore.tile<32x32, bf16>, #output_layout>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      %reserved = ttl.cb_reserve %readback_dfb
          : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>>
      %read = ttl.copy %output_slice, %readback_dfb
          : (tensor<2x2x!ttcore.tile<32x32, bf16>, #output_layout>,
             !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %read : !ttl.transfer_handle<read>
      %two = arith.constant 2 : index
      %disjoint_slice = ttl.tensor_slice %output[%two, %zero]
          : tensor<4x2x!ttcore.tile<32x32, bf16>, #output_layout>
          -> tensor<2x2x!ttcore.tile<32x32, bf16>, #output_layout>
      %write = ttl.copy %readback_dfb, %disjoint_slice
          : (!ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>,
             tensor<2x2x!ttcore.tile<32x32, bf16>, #output_layout>)
          -> !ttl.transfer_handle<write>
      ttl.wait %write : !ttl.transfer_handle<write>
    }
    return
  }
}
