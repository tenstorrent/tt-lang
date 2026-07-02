// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true})' | FileCheck %s --check-prefix=COMPUTED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s --check-prefix=PUBLISHED

// Summary: Verifies that the pipe computed-address option selects between
// computed receiver DFB addresses and receiver-published destination addresses.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @point_to_point_pipe
  // COMPUTED-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write
  // COMPUTED: ttkernel.noc_async_write
  // COMPUTED-NOT: ttkernel.load_from_l1

  // PUBLISHED-LABEL: func.func @point_to_point_pipe
  // PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
  // PUBLISHED: ttkernel.noc_inline_dw_write
  // PUBLISHED: ttkernel.load_from_l1
  // PUBLISHED: ttkernel.noc_async_write
  func.func @point_to_point_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %recv = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}

// -----

// The capacity protocol requires computed addressing, so disabling the option
// also disables capacity: the computed case emits the sender-local capacity
// semaphore handshake, the published case falls back to sender-ready and emits
// no capacity ops.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMPUTED-LABEL: func.func @capacity_pipe
  // COMPUTED: ttkernel.experimental.semaphore_wait_min
  // COMPUTED: arith.subi
  // COMPUTED: ttkernel.store_to_l1
  // COMPUTED-NOT: ttkernel.noc_inline_dw_write

  // PUBLISHED-LABEL: func.func @capacity_pipe
  // PUBLISHED-NOT: arith.subi
  // PUBLISHED-NOT: ttkernel.store_to_l1
  // PUBLISHED: ttkernel.noc_inline_dw_write
  func.func @capacity_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
