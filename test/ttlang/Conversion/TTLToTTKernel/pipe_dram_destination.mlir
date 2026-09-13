// RUN: ttlang-opt %s --split-input-file --ttl-to-ttkernel-pipeline | FileCheck %s

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
// CHECK-NOT: ttkernel.experimental.semaphore_wait
// CHECK-NEXT: ttkernel.routing_plane.scatter_write
// CHECK-SAME: %[[SOURCE]]
// CHECK-SAME: %[[DESTINATION0]]
// CHECK-SAME: %[[DESTINATION1]]
// CHECK-SAME: %[[DESTINATION2]]
// CHECK-SAME: %[[DESTINATION3]]
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc
// CHECK-NOT: ttl.pipe_transfer

// CHECK-LABEL: func.func @receiver_node
// CHECK-NOT: ttkernel.routing_plane
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

// -----

// Multiple selected records for one send share its occurrence counter. A
// later send therefore receives the next dense counter index.

// CHECK-LABEL: func.func @dense_counter_sender
// CHECK: %[[ONE_INDEX:.*]] = arith.constant 1 : index
// CHECK: %[[ZERO_VALUE:.*]] = arith.constant 0 : i32
// CHECK: %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// CHECK: %[[COUNTERS:.*]] = memref.alloca() : memref<2xi32>
// CHECK-NEXT: memref.store %[[ZERO_VALUE]], %[[COUNTERS]][%[[ZERO_INDEX]]]
// CHECK-NEXT: memref.store %[[ZERO_VALUE]], %[[COUNTERS]][%[[ONE_INDEX]]]
// CHECK: arith.remui
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 4, 0, 4] : index

#dense_counter_layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#dense_counter_output_layout = #ttl.layout<
  shape = [224, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#dense_counter_domain = #ttl.device_domain<
  components = <name = "device", extent = [3]>>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @dense_counter_sender(
      %input0: tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>,
      %input1: tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32, 1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb0 = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %send_dfb1 = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "dense_counter0" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [1]>>>>,
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [2]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      scf.for %outer = %zero to %two step %one {
        scf.for %inner = %zero to %two step %one {
          %input_slice = ttl.tensor_slice %input0[%inner, %zero]
              : tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
          %reserved = ttl.cb_reserve %send_dfb0
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %read = ttl.copy %input_slice, %send_dfb0
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read : !ttl.transfer_handle<read>
          %send = ttl.copy %send_dfb0, %pipe
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.selected_pipe_src)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
    }
    ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 1 name "dense_counter1" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [1]>>>>,
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [2]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      scf.for %iteration = %zero to %two step %one {
        %input_slice = ttl.tensor_slice %input1[%iteration, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
        %reserved = ttl.cb_reserve %send_dfb1
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %input_slice, %send_dfb1
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        %send = ttl.copy %send_dfb1, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    return
  }

  func.func @dense_counter_receiver(
      %output0: tensor<7x1x!ttcore.tile<32x32, bf16>, #dense_counter_output_layout>,
      %output1: tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [2 : i32, 3 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 1 : i32
      } {
    %readback_dfb0 = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %readback_dfb1 = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "dense_counter0" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [1]>>>>,
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [2]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      scf.for %outer = %zero to %two step %one {
        scf.for %inner = %zero to %two step %one {
          %outer_offset = arith.muli %outer, %two : index
          %inner_offset = arith.muli %inner, %four : index
          %row = arith.addi %outer_offset, %inner_offset : index
          %output_slice = ttl.tensor_slice %output0[%row, %zero]
              : tensor<7x1x!ttcore.tile<32x32, bf16>, #dense_counter_output_layout>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_output_layout>
          %receive = ttl.copy %pipe, %output_slice
              : (!ttl.selected_pipe_dst,
                 tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_output_layout>)
              -> !ttl.receive_request
          ttl.wait %receive : !ttl.receive_request
          %reserved = ttl.cb_reserve %readback_dfb0
              : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %read = ttl.copy %output_slice, %readback_dfb0
              : (tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_output_layout>,
                 !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
              -> !ttl.transfer_handle<read>
          ttl.wait %read : !ttl.transfer_handle<read>
        }
      }
    }
    ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 1 name "dense_counter1" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [1]>>>>,
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = #dense_counter_domain,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [2]>>>>
      ]>
    } {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      scf.for %iteration = %zero to %two step %one {
        %output_slice = ttl.tensor_slice %output1[%iteration, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %readback_dfb1
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %output_slice, %readback_dfb1
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #dense_counter_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// Statically enumerated, disjoint DRAM destinations use one sender-local
// occurrence counter and do not require receiver readiness signals.

// CHECK-LABEL: func.func @disjoint_sender
// CHECK: memref.alloca() : memref<1xi32>
// CHECK: scf.for
// CHECK-NOT: ttkernel.experimental.constant_table_lookup
// CHECK-NOT: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc

// CHECK-LABEL: func.func @disjoint_receiver
// CHECK: scf.for
// CHECK-NOT: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.noc_async_read
// CHECK: ttkernel.noc_async_read_barrier

#disjoint_layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#disjoint_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#disjoint_transfer = #ttl.device_transfer<
  domain = #disjoint_domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @disjoint_sender(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [0 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 0 : i32
      } {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #disjoint_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %input_slice = ttl.tensor_slice %input[%iteration, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>
        %reserved = ttl.cb_reserve %send_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %input_slice, %send_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        %send = ttl.copy %send_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    return
  }

  func.func @disjoint_receiver(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>)
      attributes {
        ttl.base_cta_index = 2 : i32,
        ttl.crta_indices = [1 : i32],
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.noc_index = 1 : i32
      } {
    %readback_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #disjoint_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %output_slice = ttl.tensor_slice %output[%iteration, %zero]
            : tensor<2x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %readback_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %output_slice, %readback_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #disjoint_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// A receiver may reuse a core-specific static DRAM region after it waits for
// and finishes reading each payload.

// CHECK-LABEL: func.func @repeated_sender
// CHECK: scf.for
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc

// CHECK-LABEL: func.func @repeated_receiver
// CHECK: scf.for
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.noc_async_read
// CHECK: ttkernel.noc_async_read_barrier

#repeated_layout = #ttl.layout<
  shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>,
  buffer = dram, grid = [1, 1], memory = interleaved>
#repeated_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#repeated_transfer = #ttl.device_transfer<
  domain = #repeated_domain,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = [1, 2], ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @repeated_sender(
      %input: tensor<2x1x!ttcore.tile<32x32, bf16>, #repeated_layout>)
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
        deviceTransfer = #repeated_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    %input_slice = ttl.tensor_slice %input[%zero, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #repeated_layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #repeated_layout>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %reserved = ttl.cb_reserve %send_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %input_slice, %send_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #repeated_layout>,
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

  func.func @repeated_receiver(
      %output: tensor<2x1x!ttcore.tile<32x32, bf16>, #repeated_layout>)
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
        deviceTransfer = #repeated_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    %core_y = ttl.core_y : index
    %output_slice = ttl.tensor_slice %output[%core_y, %zero]
        : tensor<2x1x!ttcore.tile<32x32, bf16>, #repeated_layout>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>, #repeated_layout>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %receive = ttl.copy %pipe, %output_slice
            : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 1) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>, #repeated_layout>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
        %reserved = ttl.cb_reserve %readback_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %read = ttl.copy %output_slice, %readback_dfb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>, #repeated_layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
      }
    }
    return
  }
}
