// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies that TTKernel cleanup selects stateful one-packet NoC
// writes only when the loop preserves the write command on every executing
// core.

// A one-tile unicast pipe can reuse write state while its destination address
// advances through a receiver DFB. The receiver's tensor write executes on a
// disjoint core and therefore does not invalidate the sender's command.
// CHECK-LABEL: func.func @stateful_one_packet_write_in_loop
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: } {ttkernel.execution_core_ranges = [#ttcore.core_range<(0,0), (0,0)>]}
// CHECK: scf.for
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.noc_async_write_tile
// CHECK: %[[SRC:.+]] = ttkernel.get_write_ptr
// CHECK: ttkernel.noc_async_write_one_packet_with_state(%[[SRC]]
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
#output_layout = #ttl.layout<
    shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @stateful_one_packet_write_in_loop(
      %output: tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>)
      attributes {
        ttl.base_cta_index = 1 : i32,
        ttl.crta_indices = [0],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
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
            : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %dst_cb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %slice = ttl.tensor_slice %output[%c0, %c0]
            : tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>
        %write = ttl.copy %dst_cb, %slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #output_layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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

// An inline word write uses a separate NoC command buffer, so it does not
// prevent reuse of the resident asynchronous-write command on the same core.
// CHECK-LABEL: func.func @inline_word_write_preserves_async_write_state
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
func.func @inline_word_write_preserves_async_write_state(
    %src_addr: i32, %dst_addr: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i8 = arith.constant 0 : i8
  %c15_i8 = arith.constant 15 : i8
  %value = arith.constant 1 : i32
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.noc_inline_dw_write(
        core[%c1, %c0], %dst_addr, %value, %c15_i8, noc %c0_i8)
        : (index, index, i32, i32, i8, i8) -> ()
    ttkernel.noc_async_write
        %src_addr, core[%c1, %c0], %dst_addr, %size, noc %c0_i8
        : (i32, index, index, i32, i32, i8) -> ()
  }
  func.return
}

// -----

// A call in the loop may execute another NoC write and must prevent resident
// write-command reuse across iterations.
// CHECK-LABEL: func.func private @reprogram_write
// CHECK: ttkernel.noc_async_write
// CHECK-LABEL: func.func @call_reprograms_write_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write
// CHECK: func.call @reprogram_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
func.func private @reprogram_write(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  %size = arith.constant 2048 : i32
  ttkernel.noc_async_write
      %src, core[%x, %y], %dst, %size, noc %noc
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

func.func @call_reprograms_write_state(
    %src: i32, %dst: i32, %other_dst: i32, %x: index, %y: index,
    %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.noc_async_write
        %src, core[%x, %y], %dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
    func.call @reprogram_write(%src, %other_dst, %x, %y, %noc)
        : (i32, i32, index, index, i8) -> ()
  }
  func.return
}

// -----

// A resolved helper with no write-command effects does not prevent state reuse.
// CHECK-LABEL: func.func @pure_call_preserves_write_state
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: func.call @add_one
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
func.func private @add_one(%value: i32) -> i32 {
  %one = arith.constant 1 : i32
  %result = arith.addi %value, %one : i32
  func.return %result : i32
}

func.func @pure_call_preserves_write_state(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    %unused = func.call @add_one(%dst) : (i32) -> i32
    ttkernel.noc_async_write
        %src, core[%x, %y], %dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
  }
  func.return
}

// -----

// An external callable has unknown command effects and must prevent state
// reuse.
// CHECK-LABEL: func.func @external_call_invalidates_write_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write
// CHECK: func.call @external_side_effect
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
func.func private @external_side_effect(i32)

func.func @external_call_invalidates_write_state(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.noc_async_write
        %src, core[%x, %y], %dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
    func.call @external_side_effect(%dst) : (i32) -> ()
  }
  func.return
}

// -----

// An opaque device call may reprogram NoC command registers and must prevent
// state reuse.
// CHECK-LABEL: func.func @opaque_call_invalidates_write_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.opaque_call "may_write_noc"
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
func.func @opaque_call_invalidates_write_state(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.noc_async_write
        %src, core[%x, %y], %dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
    ttkernel.opaque_call "may_write_noc" ()
        {header = "custom_noc.hpp"} : () -> ()
  }
  func.return
}

// -----

// A recursive call component remains unknown until every member is analyzed.
// The separately called component member must not reuse an incomplete cached
// result that hides a generic write in another member.
// CHECK-LABEL: func.func private @recursive_write_a
// CHECK: ttkernel.noc_async_write
// CHECK-LABEL: func.func private @recursive_write_b
// CHECK: call @recursive_write_a
// CHECK-LABEL: func.func @recursive_call_cycle_reprograms_write_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: call @recursive_write_a
// CHECK: call @recursive_write_b
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
func.func private @recursive_write_a(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  func.call @recursive_write_b(%src, %dst, %x, %y, %noc)
      : (i32, i32, index, index, i8) -> ()
  %size = arith.constant 2048 : i32
  ttkernel.noc_async_write
      %src, core[%x, %y], %dst, %size, noc %noc
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

func.func private @recursive_write_b(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  func.call @recursive_write_a(%src, %dst, %x, %y, %noc)
      : (i32, i32, index, index, i8) -> ()
  func.return
}

func.func @recursive_call_cycle_reprograms_write_state(
    %condition: i1, %src: i32, %dst: i32, %x: index, %y: index,
    %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 2048 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    scf.if %condition {
      func.call @recursive_write_a(%src, %dst, %x, %y, %noc)
          : (i32, i32, index, index, i8) -> ()
    } {ttkernel.execution_core_ranges = [#ttcore.core_range<(1,0), (1,0)>]}
    func.call @recursive_write_b(%src, %dst, %x, %y, %noc)
        : (i32, i32, index, index, i8) -> ()
    scf.if %condition {
      ttkernel.noc_async_write
          %src, core[%x, %y], %dst, %size, noc %noc
          : (i32, index, index, i32, i32, i8) -> ()
    } {ttkernel.execution_core_ranges = [#ttcore.core_range<(0,0), (0,0)>]}
  }
  func.return
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
    %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb0 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst_cb1 = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index}
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
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
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

// -----

// A transfer larger than the conservative Wormhole B0 one-packet limit keeps
// the generic write when the module does not identify its target.
// CHECK-LABEL: func.func @default_target_rejects_large_one_packet_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
func.func @default_target_rejects_large_one_packet_write(
    %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %size = arith.constant 12288 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.noc_async_write
        %src, core[%x, %y], %dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
  }
  func.return
}

// -----

// Blackhole permits the same 12288-byte transfer in one NoC packet.
// CHECK-LABEL: func.func @blackhole_accepts_large_one_packet_write
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @blackhole_accepts_large_one_packet_write(
      %src: i32, %dst: i32, %x: index, %y: index, %noc: i8) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %size = arith.constant 12288 : i32
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttkernel.noc_async_write
          %src, core[%x, %y], %dst, %size, noc %noc
          : (i32, index, index, i32, i32, i8) -> ()
    }
    func.return
  }
}

// -----

// Hoisting setup across a zero-trip loop would overwrite the resident state
// even though the generic write never executes.
// CHECK-LABEL: func.func @zero_trip_loop_preserves_prior_state
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK-NEXT: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
func.func @zero_trip_loop_preserves_prior_state(
    %src: i32, %prior_dst: i32, %new_dst: i32, %prior_x: index,
    %prior_y: index, %new_x: index, %new_y: index, %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = arith.constant 2048 : i32
  %zero = arith.constant 0 : i32
  %prior_noc_addr = ttkernel.get_noc_addr(
      %prior_x, %prior_y, %zero, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(
      %prior_noc_addr, %size, noc %noc)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  scf.for %iteration = %c0 to %c0 step %c1 {
    ttkernel.noc_async_write
        %src, core[%new_x, %new_y], %new_dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
  }
  ttkernel.noc_async_write_one_packet_with_state(
      %src, %prior_dst, noc %noc) : (i32, i32, i8) -> ()
  func.return
}

// -----

// A state-dependent write before the generic write must observe the setup that
// precedes the loop. Moving the generic write's setup before the loop changes
// that first write's destination and transfer size.
// CHECK-LABEL: func.func @state_user_before_write_prevents_hoisting
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: scf.for
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_with_state
// CHECK-NEXT: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
func.func @state_user_before_write_prevents_hoisting(
    %prior_src: i32, %new_src: i32, %prior_dst: i32, %new_dst: i32,
    %prior_x: index, %prior_y: index, %new_x: index, %new_y: index,
    %noc: i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = arith.constant 2048 : i32
  %zero = arith.constant 0 : i32
  %prior_noc_addr = ttkernel.get_noc_addr(
      %prior_x, %prior_y, %zero, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(
      %prior_noc_addr, %size, noc %noc)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  scf.for %iteration = %c0 to %c1 step %c1 {
    ttkernel.noc_async_write_one_packet_with_state(
        %prior_src, %prior_dst, noc %noc) : (i32, i32, i8) -> ()
    ttkernel.noc_async_write
        %new_src, core[%new_x, %new_y], %new_dst, %size, noc %noc
        : (i32, index, index, i32, i32, i8) -> ()
  }
  func.return
}
