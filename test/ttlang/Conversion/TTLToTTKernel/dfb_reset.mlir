// Verifies selected and all-DFB reset lowering, masks, and state allocation.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false l1-budget-override=8256})' -o /dev/null

// CHECK: module attributes {
// CHECK-DAG: ttl.dfb_reset_count = 2 : i64
// CHECK-DAG: ttl.pipe_sram_scratch_bytes = 32 : i64
// CHECK-LABEL: func.func @reset_masks
// CHECK-DAG: %[[SELECTED_LOW:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[SELECTED_HIGH:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[SECOND_OFFSET:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[ALL_LOW:.*]] = arith.constant 4 : i32
// CHECK: %[[SCRATCH_BASE:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SCRATCH_BASE]], %[[SELECTED_LOW]], %[[SELECTED_HIGH]]) {dfb_resource_indices = array<i32: 33>,
// CHECK: %[[SECOND_BASE:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[SECOND_STATE:.*]] = arith.addi %[[SECOND_BASE]], %[[SECOND_OFFSET]] : i32
// CHECK: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SECOND_STATE]], %[[ALL_LOW]], %[[SELECTED_HIGH]]) {dfb_resource_indices = array<i32: 2, 33>,

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset_masks()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %low_dfb = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %high_dfb = ttl.bind_cb {cb_index = 33, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%high_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }
}

// -----

// Reset state follows PipeNet scratch instead of overlapping its address table.
// The 8,192-byte DFB footprint plus 64-byte combined scratch exactly fits the
// second RUN's 8,256-byte budget.
// CHECK-LABEL: module attributes {
// CHECK-SAME: ttl.dfb_reset_count = 1 : i64
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 64 : i64
// CHECK-LABEL: func.func @pipe_and_reset
// CHECK: %[[RESET_OFFSET:.*]] = arith.constant 32 : i32
// CHECK: %[[RESET_STATE:.*]] = arith.addi {{.*}}, %[[RESET_OFFSET]] : i32
// CHECK: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[RESET_STATE]], {{.*}}) {dfb_resource_indices = array<i32: 0>,
module attributes {
  ttl.launch_grid = array<i64: 2, 2>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @pipe_receive()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "reset_pipe" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %destination = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %transfer = ttl.copy %pipe, %destination
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %transfer : !ttl.receive_request
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    return
  }

  func.func @pipe_and_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "reset_pipe" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %transfer = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %transfer : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
    return
  }
}

// -----

// Three 16-byte reset records occupy one allocator-rounded 64-byte scratch
// allocation.
// CHECK-LABEL: module attributes {
// CHECK-SAME: ttl.dfb_reset_count = 3 : i64
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 64 : i64
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @three_reset_records()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}
