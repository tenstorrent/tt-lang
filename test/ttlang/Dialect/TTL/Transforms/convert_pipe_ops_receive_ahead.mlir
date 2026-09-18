// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies receive-ahead schedules with multiple transfer definitions
// for one logical pipe.

// A second receive post for the same logical pipe can be live before the first
// send in one control-flow branch. Each transfer definition receives distinct
// source-node address-table storage and sender-ready synchronization.
// CHECK-LABEL: func.func @same_pipe_receive_ahead_across_blocks_allocates_two_slots
// CHECK-DAG: %[[FIRST_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[SECOND_READY_IDX:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[SECOND_TABLE_OFF:.*]] = arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: arith.addi {{.*}}, %[[SECOND_TABLE_OFF]]
// CHECK: scf.if
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[FIRST_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[SECOND_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
func.func @same_pipe_receive_ahead_across_blocks_allocates_two_slots()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  scf.if %cond {
    %recv1 = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token1 = ttl.pipe_transfer.post %transfer, %recv1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
  }
  %send0 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
  func.return
}
