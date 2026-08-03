// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=dfb}))' --verify-diagnostics

// Summary: Verifies the reserved DFB accumulation scope lowering kind emits an
// unsupported diagnostic.

func.func @dfb_kind_is_reserved() {
  // expected-error @above {{'func.func' op DFB accumulation scope lowering is not supported yet}}
  return
}
