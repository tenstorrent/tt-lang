// Verifies reset scratch cannot exceed the combined per-core L1 budget.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=2111})'

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset_overflow() {
    // expected-error @below {{total DFB and synchronized-reset allocation (2112 bytes) exceeds L1 budget (2111 bytes)}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>
    return
  }
}
