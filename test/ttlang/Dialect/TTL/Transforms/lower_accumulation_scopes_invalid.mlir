// Verifies ttl-lower-accumulation-scopes reports invalid options before
// mutating IR.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=invalid}))' --verify-diagnostics --split-input-file

// expected-error @below {{op invalid accumulation strategy `invalid`; expected auto, dst, or l1-pack}}
func.func @invalid_strategy() {
  func.return
}
