// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})'

// Verify byte-count agreement, compatibility, and one-block PipeNet transfers.

// Sender and receiver planning requires one exact transfer size.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @sender_receiver_count_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{pipe sender and receiver must use the same byte_count}}
      %token = ttl.pipe_transfer.post %transfer, %recv {
          byte_count = 1792 : i64}
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send is here}}
      %send = ttl.pipe_transfer.send %transfer, %src_dfb {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// PipeNet endpoints must preserve the source DFB data format.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @byte_counted_receiver_data_format_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // expected-error @below {{pipe receiver cannot accept the sender's byte-counted payload of 896 byte}}
      %token = ttl.pipe_transfer.post %transfer, %recv {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-note @below {{corresponding pipe send is here}}
      %send = ttl.pipe_transfer.send %transfer, %src_dfb {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A byte-counted transfer represents one DFB block per execution.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @byte_counted_grouped_transfer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 4} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        block_span = 2 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
      %token = ttl.pipe_transfer.post %transfer, %recv {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             tensor<1x2x!ttcore.tile<32x32, bf16>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{byte-counted pipe transfers require a one-block transfer span}}
      %send = ttl.pipe_transfer.send %transfer, %src_dfb {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
