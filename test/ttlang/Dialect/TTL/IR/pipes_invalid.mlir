// RUN: ttlang-opt %s --split-input-file -verify-diagnostics

// Test: cannot copy directly between two pipes.
func.func @pipe_to_pipe_copy() {
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p2 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  // expected-error @+1 {{'ttl.copy' op cannot copy directly between pipes}}
  %xf = ttl.copy %p1, %p2 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Test: pipe receive without a reserved destination DFB slot.
func.func @pipe_receive_without_reserve(%t: tensor<32x32xf32>) {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.copy' op pipe receive requires a cb_reserve destination}}
  %xf = ttl.copy %p, %t : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<32x32xf32>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Test: internal pipe transfer expected receiver count must be positive.
func.func @pipe_transfer_expected_receiver_count_positive() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op requires positive expectedReceivers}}
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 0 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// Test: internal pipe transfer expected receiver count must match the pipe.
func.func @pipe_transfer_expected_receiver_count_mismatch() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op expectedReceivers must match the pipe receiver count}}
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// Test: point-to-point pipe transfer cannot target multiple receivers.
func.func @pipe_transfer_point_to_point_multi_receiver() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op point_to_point transfer requires one receiver}}
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// Test: pipe transfer post requires a token for the same PipeNet.
func.func @pipe_transfer_post_token_net_mismatch() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @+1 {{'ttl.pipe_transfer.post' op token pipeNetId must match transfer pipeNetId}}
  %token = ttl.pipe_transfer.post %transfer, %dst
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 1>
  func.return
}

// -----

// Test: pipe transfer send result is a write handle.
func.func @pipe_transfer_send_requires_write_handle() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  // expected-error @+1 {{'ttl.pipe_transfer.send' op requires a write transfer handle result}}
  %xf = ttl.pipe_transfer.send %transfer, %cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Test: pipe transfer wait requires a token produced by a post.
func.func @pipe_transfer_wait_requires_post_token() {
  %token = builtin.unrealized_conversion_cast to !ttl.pipe_token<net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.wait' op requires token derived from ttl.pipe_transfer.post}}
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// Test: negative source coordinates.
// expected-error @+1 {{'ttl.create_pipe' op source coordinates must be non-negative}}
%p = ttl.create_pipe src(-1, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(-1, 0) dst(1, 0) to(1, 0) net 0>

// -----

// Test: negative destination coordinates.
// expected-error @+1 {{'ttl.create_pipe' op destination coordinates must be non-negative}}
%p = ttl.create_pipe src(0, 0) dst(-1, 0) to(-1, 0) net 0 : !ttl.pipe<src(0, 0) dst(-1, 0) to(-1, 0) net 0>

// -----

// Test: attributes must match result pipe type.
// expected-error @+1 {{'ttl.create_pipe' op attributes must match result pipe type}}
%p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>

// -----

// Test: dst start must not exceed dst end on x.
// expected-error @+1 {{'ttl.create_pipe' op destination start must not exceed destination end on any axis}}
%p = ttl.create_pipe src(0, 0) dst(3, 0) to(0, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(0, 0) net 0>

// -----

// Test: dst start must not exceed dst end on y.
// expected-error @+1 {{'ttl.create_pipe' op destination start must not exceed destination end on any axis}}
%p = ttl.create_pipe src(0, 0) dst(0, 5) to(0, 2) net 0 : !ttl.pipe<src(0, 0) dst(0, 5) to(0, 2) net 0>

// -----

// Test: explicit point-to-point metadata cannot contradict a multi-receiver pipe.
// expected-error @+1 {{'ttl.create_pipe' op isCollective=false is invalid for a multi-receiver pipe}}
%p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 {isCollective = false} : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
