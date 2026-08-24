// Summary: Verify PipeTransport grouping retains scalar execution when tensor
// slice progression is not proven contiguous.
// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s

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
}
