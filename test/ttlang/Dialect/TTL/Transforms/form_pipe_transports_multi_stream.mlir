// Summary: Verify independent grouped PipeTransport streams receive disjoint
// scratch intervals before scalar receiver-address storage.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=2},convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=true})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=2},convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=true})' | FileCheck %s --check-prefix=LOWERED

#layout = #ttl.layout<
    shape = [32, 128], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// The two transport-owned rings require 16384 bytes each. The scalar stream's
// receiver-published address table follows both intervals.
// PLAN: PipeTransport: stream 0 transfer 0
// PLAN-NEXT: PipeTransport:   source
// PLAN-SAME: ownership=transport scratch_offset=0 scratch_bytes=8192
// PLAN-NEXT: PipeTransport:   endpoint 0
// PLAN-SAME: ownership=transport scratch_offset=0 scratch_bytes=16384
// PLAN: PipeTransport: stream 1 transfer 1
// PLAN-NEXT: PipeTransport:   source
// PLAN-SAME: ownership=transport scratch_offset=16384 scratch_bytes=8192
// PLAN-NEXT: PipeTransport:   endpoint 1
// PLAN-SAME: ownership=transport scratch_offset=16384 scratch_bytes=16384

// LOWERED: module attributes
// LOWERED-SAME: ttl.pipe_sram_scratch_bytes = 32800 : i64
// LOWERED-LABEL: func.func @multiple_streams
// LOWERED-DAG: %[[SECOND_SCRATCH_OFFSET:.*]] = arith.constant 16384 : i32
// LOWERED-DAG: %[[ADDRESS_TABLE_OFFSET:.*]] = arith.constant 32768 : i32
// LOWERED: arith.addi %{{.*}}, %[[SECOND_SCRATCH_OFFSET]]
// LOWERED: arith.addi %{{.*}}, %[[ADDRESS_TABLE_OFFSET]]

module attributes {ttl.launch_grid = array<i64: 4, 1>} {
  func.func @multiple_streams(
      %input_a: tensor<1x4x!ttcore.tile<32x32, f32>, #layout>,
      %output_a: tensor<1x4x!ttcore.tile<32x32, f32>, #layout>,
      %input_b: tensor<1x4x!ttcore.tile<32x32, f32>, #layout>,
      %output_b: tensor<1x4x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 4 : i32,
                  ttl.crta_indices = [0, 1, 2, 3],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_a = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_a = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %src_b = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_b = ttl.bind_cb {cb_index = 3, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %src_scalar = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_scalar = ttl.bind_cb {cb_index = 5, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    scf.for %iter = %c0 to %c4 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %src_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input_a[%c0, %iter]
            : tensor<1x4x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_a
            : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %src_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_a, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %received = ttl.cb_wait %dst_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output_a[%c0, %iter]
            : tensor<1x4x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_a, %output_slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_a
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }

    scf.for %iter = %c0 to %c4 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
          : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %reserved = ttl.cb_reserve %src_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input_b[%c0, %iter]
            : tensor<1x4x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_b
            : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %src_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_b, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
               !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
        %recv = ttl.cb_reserve %dst_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %received = ttl.cb_wait %dst_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output_b[%c0, %iter]
            : tensor<1x4x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_b, %output_slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_b
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }

    %scalar_pipe = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
    ttl.if_dst %scalar_pipe
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0> {
      %recv = ttl.cb_reserve %dst_scalar
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %recv_handle = ttl.copy %scalar_pipe, %recv
          : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %recv_handle : !ttl.transfer_handle
      ttl.cb_push %dst_scalar
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %received = ttl.cb_wait %dst_scalar
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_scalar
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %scalar_pipe
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0> {
      %send = ttl.copy %src_scalar, %scalar_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
