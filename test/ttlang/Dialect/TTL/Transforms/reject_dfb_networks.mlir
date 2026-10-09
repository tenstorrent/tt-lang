// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-reject-dfb-networks | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-to-ttkernel-pipeline -o /dev/null

// Summary: Verifies that every DFB network reaching the pass is rejected, both
// when the pass runs alone and as the first pass of the TTL-to-TTKernel
// pipeline, and that a module without networks passes through unchanged.

// Test that each network in a module is reported, not only the first.
module {
  // expected-error @below {{'ttl.dfb.network' op "reject_net_1": DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_1 {}
  // expected-error @below {{'ttl.dfb.network' op "reject_net_2": DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_2 {}
}

// -----

// Test that a network with records is rejected.
module {
  // expected-error @below {{'ttl.dfb.network' op "reject_net_records": DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_records {
    ttl.dfb.split @s 0 : index -> [1, 2] policy = <round_robin>
  }
}

// -----

// Test that a network in a nested module is rejected.
module {
  module @inner {
    // expected-error @below {{'ttl.dfb.network' op "reject_net_nested": DFB networks are not supported yet}}
    ttl.dfb.network @reject_net_nested {}
  }
}

// -----

// Test that a module without networks passes through unchanged.
// CHECK-LABEL: func.func @no_network
// CHECK-NEXT: return
module {
  func.func @no_network() {
    return
  }
}
