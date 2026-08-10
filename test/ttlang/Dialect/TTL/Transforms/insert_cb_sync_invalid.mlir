// Verifies diagnostics for DFB publication without a receive completion proof.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))'

// Wait-any proves one candidate complete, not every candidate.
func.func @unguarded_wait_any_push()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %block0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request0 = ttl.copy %pipe0, %block0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %block1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  // expected-error @below {{publishes a wait-any receive reservation without proving that candidate complete}}
  ttl.cb_push %landing0
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
