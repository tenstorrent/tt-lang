// Verifies accumulation scope formation rejects unknown strategy options.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{strategy=invalid}))' --verify-diagnostics --split-input-file

func.func @invalid_strategy() {
  // expected-error @above {{'func.func' op invalid accumulation strategy `invalid`; expected auto, dst, or l1-pack}}
  return
}
