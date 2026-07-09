// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{kind=dfb}))' --verify-diagnostics

// Summary: Verifies the reserved DFB accumulation scope formation kind emits
// an unsupported diagnostic.

func.func @dfb_kind_is_reserved() {
  // expected-error @above {{'func.func' op DFB accumulation scope formation is not supported yet}}
  return
}
