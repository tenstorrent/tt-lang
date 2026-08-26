// Verifies DFB reconfiguration conversion precondition diagnostics.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

module attributes {
  ttl.target_arch = #ttcore.arch<wormhole_b0>,
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @boundary() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    // expected-error @below {{'ttl.dfb_reconfiguration' op is supported only for Blackhole; selected target is #ttcore.arch<wormhole_b0>}}
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}

// -----

// Verifies that conversion requires the finalized reconfiguration plan.
// expected-error @below {{requires finalized DFB reconfiguration metadata}}
module attributes {
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @missing_plan() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}

// -----

// Verifies that conversion rejects a boundary absent from the finalized plan.
// expected-error @below {{boundary ordinal is absent from finalized DFB reconfiguration metadata}}
module attributes {
  ttl.target_arch = #ttcore.arch<blackhole>,
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 1>,
    dfbs = []
  }
} {
  func.func @missing_boundary() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}
