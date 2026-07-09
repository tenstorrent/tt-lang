// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{kind=unknown}))' --verify-diagnostics

// Summary: Verifies accumulation scope formation rejects unknown scope kinds.

func.func @invalid_kind() {
  // expected-error @above {{'func.func' op invalid accumulation scope formation kind `unknown`; expected `tensor` or `dfb`}}
  return
}
