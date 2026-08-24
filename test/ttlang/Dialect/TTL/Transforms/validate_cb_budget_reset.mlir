// Verifies an allocator-rounded reset record participates in the DFB budget.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=2112})' | FileCheck %s

// CHECK-LABEL: module attributes
// CHECK: ttl.reset_all_dfbs
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @exact_reset_budget() {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>
    return
  }
}
