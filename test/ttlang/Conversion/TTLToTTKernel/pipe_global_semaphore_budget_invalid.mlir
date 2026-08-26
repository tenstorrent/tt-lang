// Verifies exact L1 validation includes target-aligned GlobalSemaphore storage.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-global-semaphores-only=true l1-budget-override=12479})'

// expected-error @below {{combined DFB and runtime resources require 12480 L1 bytes but the budget is 12479 (DFB=12288, scratch=64, global semaphores=128, reconfiguration state=0)}}
module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @global_pipe_counters()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    return
  }
}
