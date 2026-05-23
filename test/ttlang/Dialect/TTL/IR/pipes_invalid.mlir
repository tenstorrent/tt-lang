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

// Test: internal pipe receive post requires an untyped transfer handle.
func.func @pipe_recv_post_typed_handle_result() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @+1 {{'ttl.pipe_recv_post' op requires an untyped transfer handle result}}
  %xf = ttl.pipe_recv_post %p, %dst
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle<write>
  func.return
}

// -----

// Test: internal pipe receive wait requires an untyped transfer handle.
func.func @pipe_recv_wait_typed_handle_operand() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %cb, %p
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  // expected-error @+1 {{'ttl.pipe_recv_wait' op requires an untyped transfer handle operand}}
  ttl.pipe_recv_wait %xf, %p, %dst
      : !ttl.transfer_handle<write>,
        !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
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
