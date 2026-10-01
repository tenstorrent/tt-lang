// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: ttl.topk rejects operand types, shapes, and attributes the local
// bitonic lowering cannot implement.

// Operands must be rank-2 tensors.
func.func @rank_one_operands(%values: tensor<2x!ttcore.tile<32x32, bf16>>,
                             %indices: tensor<2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{values and indices must be static rank-2 tensors}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<2x!ttcore.tile<32x32, bf16>>, tensor<2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x!ttcore.tile<32x32, bf16>>, tensor<1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// Indices must be shaped like values.
func.func @shape_mismatch(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                          %indices: tensor<1x4x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{values and indices must have the same shape}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x4x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// Scalar element types are not tiles.
func.func @scalar_elements(%values: tensor<1x2xbf16>, %indices: tensor<1x2xi16>) {
  // expected-error @below {{values and indices must have tile element types}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2xbf16>, tensor<1x2xi16>) -> (tensor<1x1xbf16>, tensor<1x1xi16>)
  return
}

// -----

// The fused sort key holds a 16-bit value.
func.func @values_are_not_bf16(%values: tensor<1x2x!ttcore.tile<32x32, f32>>,
                               %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{values must be bf16 tiles}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// The fused sort key holds a 16-bit index.
func.func @indices_are_not_u16(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                               %indices: tensor<1x2x!ttcore.tile<32x32, u32>>) {
  // expected-error @below {{indices must be u16 tiles}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u32>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u32>>)
  return
}

// -----

func.func @unsupported_k(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                         %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 3
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// A single tile has no partner for the first merge.
func.func @width_is_one_tile(%values: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                             %indices: tensor<1x1x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{width in tiles must be 2, 4, or 8}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

func.func @width_is_not_a_power_of_two(
    %values: tensor<1x3x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x3x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{width in tiles must be 2, 4, or 8}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x3x!ttcore.tile<32x32, bf16>>, tensor<1x3x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// The unrolled lowering of 16 tiles exceeds the kernel binary budget.
func.func @width_is_too_wide(%values: tensor<1x16x!ttcore.tile<32x32, bf16>>,
                             %indices: tensor<1x16x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{width in tiles must be 2, 4, or 8}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x16x!ttcore.tile<32x32, bf16>>, tensor<1x16x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// Results are whole tiles: one tile for k <= 32, two for k = 64.
func.func @result_width_does_not_match_k(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{results must have shape [height, (k + 31) / 32] and the input element types}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// Result element types follow the inputs.
func.func @result_element_type_differs(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  // expected-error @below {{results must have shape [height, (k + 31) / 32] and the input element types}}
  %values_out, %indices_out = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u32>>)
  return
}
