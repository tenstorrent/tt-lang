// A pipe send after a reserve uses the reserved block even when an earlier
// wait on the same DFB dominates it. The push must follow the send completion.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' | FileCheck %s

// CHECK-LABEL: func.func @wait_pop_reserve_pipe_send
// CHECK: %[[DFB:.*]] = ttl.bind_cb
// CHECK: ttl.cb_wait %[[DFB]]
// CHECK: ttl.cb_pop %[[DFB]]
// CHECK: ttl.cb_reserve %[[DFB]]
// CHECK: %[[SEND:.*]] = ttl.copy %[[DFB]],
// CHECK: ttl.wait %[[SEND]]
// CHECK-NEXT: ttl.cb_push %[[DFB]]
func.func @wait_pop_reserve_pipe_send()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %waited = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %send = ttl.copy %dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
