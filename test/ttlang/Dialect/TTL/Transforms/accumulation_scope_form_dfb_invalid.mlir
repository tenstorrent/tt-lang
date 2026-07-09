// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{kind=dfb}))' --verify-diagnostics

// Summary: Verifies the reserved DFB accumulation scope kind emits an
// unsupported diagnostic.

func.func @dfb_kind_is_reserved() {
  // expected-error @above {{'func.func' op DFB accumulation scopes are not supported yet}}
  return
}
