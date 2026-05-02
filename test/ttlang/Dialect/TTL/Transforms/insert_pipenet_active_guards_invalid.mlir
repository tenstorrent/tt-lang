// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-insert-pipenet-active-guards

// Negative tests for ttl-insert-pipenet-active-guards. The pass requires
// kernel-thread functions to be single-block with a func.return terminator;
// violations must produce a clear diagnostic rather than miscompile.

// expected-error @below {{ttl-insert-pipenet-active-guards requires single-block functions}}
func.func @multi_block_kernel_thread() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  cf.br ^bb1
^bb1:
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
  }
  func.return
}
