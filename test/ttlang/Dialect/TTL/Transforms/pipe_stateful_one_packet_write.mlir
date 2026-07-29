// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies that pipe send lowering uses stateful one-packet NoC
// writes only when one loop owns a single eligible pipe send.

// A one-tile unicast pipe with a computed receiver DFB address can program the
// destination command state once before the loop and reuse it per transfer.
// CHECK-LABEL: func.func @stateful_one_packet_write_in_loop
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: %[[SRC:.+]] = ttkernel.get_write_ptr
// CHECK: ttkernel.noc_async_write_one_packet_with_state(%[[SRC]]
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @stateful_one_packet_write_in_loop()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %ready = ttl.cb_wait %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %send = ttl.pipe_transfer.send %transfer, %src_cb
            : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// The inner loop reprograms the NoC write command before the outer send, so
// only the inner send can reuse resident command state.
// CHECK-LABEL: func.func @nested_sends_use_generic_writes
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: ttkernel.noc_async_write{{[ (]}}
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @nested_sends_use_generic_writes()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb0 = ttl.bind_cb {cb_index = 2, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst_cb1 = ttl.bind_cb {cb_index = 3, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1>
    %transfer0 = ttl.pipe_transfer.create %pipe0
        {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    %transfer1 = ttl.pipe_transfer.create %pipe1
        {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      ttl.if_dst %pipe0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_cb0
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer0, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst_cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %ready = ttl.cb_wait %dst_cb0
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst_cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
      ttl.if_dst %pipe1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> {
        %recv = ttl.cb_reserve %dst_cb1
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer1, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 1>
        ttl.cb_push %dst_cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %ready = ttl.cb_wait %dst_cb1
            : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst_cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
      scf.for %inner = %c0 to %c1 step %c1 {
        ttl.if_src %pipe0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
          %send = ttl.pipe_transfer.send %transfer0, %src_cb0
              : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      ttl.if_src %pipe1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> {
        %send = ttl.pipe_transfer.send %transfer1, %src_cb1
            : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// A tensor write in the transfer loop overwrites the NoC write command state,
// so the pipe send must keep programming the complete command.
// CHECK-LABEL: func.func @tensor_write_in_loop_uses_generic_pipe_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write{{[ (]}}
// CHECK: ttkernel.noc_async_write_tile
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
#output_layout = #ttl.layout<
    shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @tensor_write_in_loop_uses_generic_pipe_write(
      %output: tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>)
      attributes {
        ttl.base_cta_index = 1 : i32,
        ttl.crta_indices = [0],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer, %recv
            : (!ttl.pipe_transfer,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %ready = ttl.cb_wait %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %send = ttl.pipe_transfer.send %transfer, %src_cb
            : (!ttl.pipe_transfer,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        %slice = ttl.tensor_slice %output[%c0, %c0]
            : tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>
        %write = ttl.copy %src_cb, %slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}
