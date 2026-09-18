// Summary: Verifies the external DFB domain override in PipeNet guard analysis.

// RUN: env TTL_RELAX_DFB_SPSC=0 ttlang-opt %s -ttl-verify-pipenet-guards 2>%t.warning | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// A standalone guard pass records the same audit marker and warning.
// CHECK: ttl.relaxed_dfb_protocol_domain_verification
// WARNING-COUNT-1: warning: `TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, consumer, and wait correspondence checks
// WARNING-NOT: warning: `TTL_RELAX_DFB_SPSC`

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @wait_without_represented_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
