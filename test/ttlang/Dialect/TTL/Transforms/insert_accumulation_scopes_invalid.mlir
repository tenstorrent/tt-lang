// Verifies ttl-insert-accumulation-scopes rejects unsupported insertion modes.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-accumulation-scopes{kind=invalid}))' --verify-diagnostics --split-input-file

// expected-error @below {{'func.func' op invalid accumulation scope insertion kind `invalid`; expected `dfb`}}
func.func @unsupported_scope_kind() {
  return
}
