// Summary: Verify PipeTransport grouping retains scalar execution when tensor
// slice progression is not proven contiguous.
// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=2 l1-budget-override=29760' | FileCheck %s --check-prefix=CALLBACK-BUDGET

#layout = #ttl.layout<
    shape = [32, 256], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// Adjacent loop indices start overlapping two-tile blocks. Grouping cannot
// widen them into one contiguous block without changing the transferred tiles.
// CHECK-LABEL: func.func @overlapping_multitile_blocks
// CHECK-NOT: block_span
// CHECK: scf.for %[[ITER:.*]] = %{{.*}} to %{{.*}} step %[[ONE:.*]] {
// CHECK: ttl.tensor_slice %{{.*}}[%{{.*}}, %[[ITER]]]
// CHECK-SAME: -> tensor<1x2x!ttcore.tile<32x32, f32>,

// Tensor-backed DFB capacities are fixed by their backing ranges. A candidate
// that would resize either DFB retains the scalar transfer.
// CHECK-LABEL: func.func @tensor_backed_capacity
// CHECK: ttl.bind_cb{cb_index = 2, block_count = 2}
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 8192>
// CHECK: ttl.bind_cb{cb_index = 3, block_count = 1}
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4096>
// CHECK-NOT: block_span
// CHECK: scf.for
// CHECK: ttl.pipe_transfer.send

// CHECK-LABEL: func.func @callback_budget_static
// CHECK: ttl.pipe_transfer.create {{.*}}block_span = 2 : i64

// The scalar module fits exactly at 29760 bytes; capacity synchronization for
// both callback transfers makes the grouped candidate exceed the budget.
// CALLBACK-BUDGET-LABEL: func.func @callback_budget_static
// CALLBACK-BUDGET-NOT: block_span
// CALLBACK-BUDGET: scf.for
// CALLBACK-BUDGET: ttl.pipe_transfer.send

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @overlapping_multitile_blocks(
      %input: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x2x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_dfb
            : (tensor<1x2x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %src_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_dfb, %pipe
            : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x2x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        %received = ttl.cb_wait %dst_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x2x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_dfb, %output_slice
            : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>,
               tensor<1x2x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_dfb
            : <[1, 2], !ttcore.tile<32x32, f32>, 1>
      }
    }
    func.return
  }

  func.func @tensor_backed_capacity(
      %input: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 8192>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 3, block_count = 1}
        {dfb_id = 3 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_dfb
            : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %received = ttl.cb_wait %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_dfb, %output_slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }
    func.return
  }

  func.func @callback_budget_static(
      %input: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x8x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
        {dfb_id = 4 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 8192>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 5, block_count = 4}
        {dfb_id = 5 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 16384>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c4 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 2
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_dfb
            : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 2>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
        %received = ttl.cb_wait %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output[%c0, %iter]
            : tensor<1x8x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_dfb, %output_slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
      }
    }
    func.return
  }

  func.func private @callback_budget_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 6, block_count = 1} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 3 name "callback" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %send_again = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send_again : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }
}
