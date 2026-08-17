// Verifies that conversion rejects reconfiguration on unsupported targets.
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
