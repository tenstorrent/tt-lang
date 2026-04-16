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

// Test: pipe copy without CB operand.
func.func @pipe_without_cb(%t: tensor<32x32xf32>) {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.copy' op pipe transfers require one operand to be !ttl.cb}}
  %xf = ttl.copy %p, %t : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<32x32xf32>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
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
