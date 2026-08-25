// Verifies three reset records round to a 64-byte L1 allocation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=63})'

// expected-error @below {{total DFB and fixed-state allocation (64 bytes) exceeds L1 budget (63 bytes)}}
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @three_reset_records() {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_only">, <kind = data_movement, identity = "reader", operation = "reset_only">, <kind = data_movement, identity = "writer", operation = "reset_only">]>
    ttl.reset_all_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_only">, <kind = data_movement, identity = "reader", operation = "reset_only">, <kind = data_movement, identity = "writer", operation = "reset_only">]>
    ttl.reset_all_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reset_only">, <kind = data_movement, identity = "reader", operation = "reset_only">, <kind = data_movement, identity = "writer", operation = "reset_only">]>
    return
  }
}
