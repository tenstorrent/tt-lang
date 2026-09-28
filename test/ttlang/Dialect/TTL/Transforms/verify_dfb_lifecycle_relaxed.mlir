// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle)' 2>%t.warning | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// Relaxed ownership lets several producer or consumer kernels be visible on
// one node. The lifecycle verifier still checks each kernel's own sequence and
// the totals of every kernel that provably acts on the node: two producers
// reserving once each balance two waits.
// CHECK: ttl.relaxed_dfb_protocol_domain_verification
// WARNING-COUNT-1: warning: `TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, consumer, and wait correspondence checks

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer_first() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @producer_second() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
