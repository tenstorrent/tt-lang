// Verifies ttl-form-accumulation-scopes rejects unsupported formation modes.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{kind=invalid}))' --verify-diagnostics

// expected-error @below {{'func.func' op invalid accumulation scope formation kind `invalid`; expected `tensor` or `dfb`}}
func.func @unsupported_scope_kind() {
  return
}
