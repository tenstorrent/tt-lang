// Summary: Verify callback resources conservatively bound mixed static PipeNet grouping.
// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s --check-prefix=GROUPED
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=2 l1-budget-override=16640' | FileCheck %s --check-prefix=SCALAR
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=2 l1-budget-override=25728},convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=true pipe-global-semaphores-only=true l1-budget-override=25728})' | FileCheck %s --check-prefix=EXACT

#layout = #ttl.layout<
    shape = [32, 32000000000000], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// GROUPED-LABEL: func.func @static_transfer
// GROUPED: ttl.pipe_transfer.create {{.*}}block_span = 2 : i64

// SCALAR-LABEL: func.func @static_transfer
// SCALAR-NOT: block_span
// SCALAR: scf.for
// SCALAR: ttl.pipe_transfer.send

// EXACT: module attributes
// EXACT-SAME: ttl.pipe_global_semaphore_count = 6 : i64
// EXACT-SAME: ttl.pipe_sram_scratch_bytes = 16416 : i64
// EXACT-LABEL: func.func @static_transfer
// EXACT: ttkernel.noc_async_write

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @static_transfer(
      %input: tensor<1x1000000000000x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x1000000000000x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 8192>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 4}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 16384>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %c0 = arith.constant 0 : index
    %c1000000000000 = arith.constant 1000000000000 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c1000000000000 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x1000000000000x!ttcore.tile<32x32, f32>, #layout>
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
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %recv_handle : !ttl.receive_request
        ttl.cb_push %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
        %received = ttl.cb_wait %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 4>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output[%c0, %iter]
            : tensor<1x1000000000000x!ttcore.tile<32x32, f32>, #layout>
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

  func.func @callback_transfer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 1 name "callback" pipes [
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

  func.func @callback_receive()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 1 name "callback" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %first = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %reserved_again = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive_again = ttl.copy %pipe, %reserved_again
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive_again : !ttl.receive_request
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
