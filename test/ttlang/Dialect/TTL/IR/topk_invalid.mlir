// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: ttl.topk rejects shapes and attributes the local bitonic lowering
// cannot implement.

func.func @unsupported_k(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                         %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 3
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

func.func @width_is_not_a_power_of_two(
    %values: tensor<1x3x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x3x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{width in tiles must be a power of two}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x3x!ttcore.tile<32x32, bf16>>, tensor<1x3x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}
