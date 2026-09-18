// Verify conversion rejects tile operations with alternatives when no
// execution strategy has been selected.
// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --verify-diagnostics

func.func @missing_tile_execution_strategy(
    %lhs: !ttcore.tile<32x32, bf16>,
    %rhs: !ttcore.tile<32x32, bf16>) {
  %zero = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_add' op requires a selected ttl.tile_execution_strategy attribute; run ttl-set-compute-kernel-config before DST assignment, scheduling, or lowering}}
  %sum = ttl.tile_add %lhs, %rhs into dst[%zero]
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}
