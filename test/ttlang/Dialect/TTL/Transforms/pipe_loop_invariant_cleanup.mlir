// The computed-address protocol must not load a receiver-published address.
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s --implicit-check-not=ttkernel.load_from_l1

// Summary: Verifies that TTL-to-TTKernel cleanup hoists loop-invariant
// PipeNet value construction and selects stateful NoC writes when other NoC
// commands execute on disjoint cores.

// Sender-side pipe lowering inside a multi-tile transfer loop should compute
// the role predicate and receiver coordinates once. The DFB write pointer
// remains inside the loop because it depends on buffer state. The write command
// state is configured once because receiver-side commands execute on a
// different core.
// CHECK-LABEL: func.func @sender_loop_hoists_pipe_invariants
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[SLOT_COUNT:.+]] = arith.constant 2 : i32
// CHECK-DAG: %[[WRITE_SIZE:.+]] = arith.constant 8192 : i32
// CHECK: %[[SLOT_STATE:.+]] = memref.alloca() : memref<1xi32>
// CHECK-NEXT: memref.store %[[C0_I32]], %[[SLOT_STATE]][%[[C0]]] : memref<1xi32>
// CHECK-DAG: %[[CORE_X:.+]] = ttkernel.my_logical_x_
// CHECK-DAG: %[[CORE_Y:.+]] = ttkernel.my_logical_y_
// CHECK: %[[DST_X:.+]] = ttkernel.experimental.convert_logical_x_to_translated(%[[C1]])
// CHECK-NEXT: %[[DST_Y:.+]] = ttkernel.experimental.convert_logical_y_to_translated(%[[C0]])
// CHECK: %[[DST_BASE:.+]] = ttkernel.get_common_arg_val(%[[C0]]) : (index) -> i32
// CHECK: %[[SETUP_ADDR:.+]] = ttkernel.get_noc_addr(%[[DST_X]], %[[DST_Y]], %[[C0_I32]],
// CHECK: scf.if %[[IS_SRC:.+]]
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_set_state(%[[SETUP_ADDR]], %[[WRITE_SIZE]],
// CHECK: scf.for
// CHECK-NOT: ttkernel.my_logical_x_
// CHECK-NOT: ttkernel.my_logical_y_
// CHECK-NOT: ttkernel.experimental.convert_logical_x_to_translated
// CHECK-NOT: ttkernel.experimental.convert_logical_y_to_translated
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK-NOT: ttkernel.experimental.convert_logical_x_to_translated
// CHECK-NOT: ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[SRC:.+]] = ttkernel.get_write_ptr
// CHECK-NEXT: %[[SLOT:.+]] = memref.load %[[SLOT_STATE]][%[[C0]]] : memref<1xi32>
// CHECK-NEXT: %[[OFFSET:.+]] = arith.muli %[[SLOT]], %[[WRITE_SIZE]] : i32
// CHECK-NEXT: %[[DST_ADDR:.+]] = arith.addi %[[DST_BASE]], %[[OFFSET]] : i32
// CHECK-NEXT: %[[NEXT_SLOT:.+]] = arith.addi %[[SLOT]], %[[C1_I32]] : i32
// CHECK-NEXT: %[[WRAPPED_SLOT:.+]] = arith.remui %[[NEXT_SLOT]], %[[SLOT_COUNT]] : i32
// CHECK-NEXT: memref.store %[[WRAPPED_SLOT]], %[[SLOT_STATE]][%[[C0]]] : memref<1xi32>
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_with_state(%[[SRC]], %[[DST_ADDR]],
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
func.func @sender_loop_hoists_pipe_invariants()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
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
          : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x2x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb
          : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 2], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
  }
  func.return
}
}
