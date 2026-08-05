// Verify that passes consuming selected tile execution semantics reject a
// strategy that is no longer legal for the operation's operands.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst))' --verify-diagnostics
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-schedule-operations))' --verify-diagnostics
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel)' --verify-diagnostics

func.func @stale_fpu_strategy(%lhs: !ttcore.tile<32x32, bf16>,
                              %rhs: !ttcore.tile<32x32, bf16>) {
  %zero = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_add' op explicit ttl.tile_execution_strategy is not legal for its operands}}
  %sum = ttl.tile_add %lhs, %rhs into dst[%zero]
      {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>}
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}
