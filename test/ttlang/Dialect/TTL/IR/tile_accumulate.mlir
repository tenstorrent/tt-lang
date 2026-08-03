// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies `ttl.tile_accumulate` parses and prints its combiner.

// CHECK-LABEL: func.func @tile_accumulate_add
// CHECK: %[[RES:.*]] = ttl.tile_accumulate %{{.*}}, %{{.*}} add into dst[%{{.*}}]
func.func @tile_accumulate_add(
    %accumulator: !ttcore.tile<32x32, bf16>,
    %contribution: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, bf16> {
  %c0 = arith.constant 0 : index
  %result = ttl.tile_accumulate %accumulator, %contribution add into dst[%c0]
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return %result : !ttcore.tile<32x32, bf16>
}
