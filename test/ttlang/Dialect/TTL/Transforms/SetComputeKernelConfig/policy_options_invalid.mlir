// Verify an invalid three-state policy value is rejected.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=invalid}))' --verify-diagnostics

// expected-error @below {{invalid fp32-dest-acc-en value 'invalid'; expected auto, enabled, or disabled}}
func.func @invalid_policy_option() {
  return
}
