// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-print-dfb-network-plan)' -o /dev/null 2>&1 | FileCheck %s

// Summary: Verifies the printed DFB network plan: every network that cannot be
// expanded yet is listed in module order, including one in a nested module, and
// a module without networks prints a planned result.

// Test that two networks are both listed, in order.
// CHECK-LABEL: DFB network plan: rejected
// CHECK-NEXT:    @plan_net_1: DFB networks are not supported yet
// CHECK-NEXT:    @plan_net_2: DFB networks are not supported yet
module {
  ttl.dfb.network @plan_net_1 {}
  ttl.dfb.network @plan_net_2 {}
}

// -----

// Test that a network in a nested module is listed.
// CHECK-LABEL: DFB network plan: rejected
// CHECK-NEXT:    @plan_net_nested: DFB networks are not supported yet
module {
  module @inner {
    ttl.dfb.network @plan_net_nested {}
  }
}

// -----

// Test that a module without networks prints a planned result.
// CHECK-LABEL: DFB network plan: planned
module {
  func.func @no_network() {
    return
  }
}
