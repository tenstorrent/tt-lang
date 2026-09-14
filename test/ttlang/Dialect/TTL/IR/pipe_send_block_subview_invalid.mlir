// RUN: ttlang-opt %s --split-input-file --verify-diagnostics
// Summary: Reject pipe-send block operands without a valid DFB acquisition or
// supported static subview.

module {
  func.func @pipe_send_requires_waited_subview()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %source_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %reserved = ttl.cb_reserve %source_dfb
        : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %reserved, %source_dfb
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 1] [1, 1] [1, 1]
        : tensor<1x2x!ttcore.tile<32x32, f32>>
        to tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe send source DFB view must come from ttl.cb_wait}}
    %send = ttl.copy %view, %pipe
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

module {
  func.func @pipe_send_requires_unit_stride()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %source_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %waited = ttl.cb_wait %source_dfb
        : <[1, 3], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x3x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %waited, %source_dfb
        : (tensor<1x3x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x3x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 0] [1, 2] [1, 2]
        : tensor<1x3x!ttcore.tile<32x32, f32>>
        to tensor<1x2x!ttcore.tile<32x32, f32>>
    // expected-error @below {{pipe send DFB view must have unit strides}}
    %send = ttl.copy %view, %pipe
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

module {
  func.func @pipe_transfer_send_requires_dfb_source(%source: tensor<1x1xf32>)
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    // expected-error @below {{requires a DFB block or block subview source}}
    %send = ttl.pipe_transfer.send %transfer, %source
        : (!ttl.pipe_transfer, tensor<1x1xf32>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

module {
  func.func @pipe_send_requires_acquired_block(
      %source: tensor<1x1x!ttcore.tile<32x32, f32>>)
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %source_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %block = ttl.attach_cb %source, %source_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{pipe send source must come from ttl.cb_reserve or ttl.cb_wait}}
    %send = ttl.copy %block, %pipe
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}
