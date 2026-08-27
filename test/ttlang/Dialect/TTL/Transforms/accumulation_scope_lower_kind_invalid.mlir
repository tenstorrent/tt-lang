// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=unknown}))' --verify-diagnostics --split-input-file

// Summary: Verifies accumulation scope lowering rejects unknown scope kinds.

func.func @invalid_kind() {
  // expected-error @above {{'func.func' op invalid accumulation scope lowering kind `unknown`; expected `tensor` or `dfb`}}
  return
}
