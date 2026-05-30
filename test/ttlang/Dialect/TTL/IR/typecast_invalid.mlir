// Verifier tests for ttl.tile_typecast: the tile shape must be preserved,
// input/result data types must differ, and both data types must be floating
// point formats supported by the hardware typecast path.
//
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// Result tile shape must equal input tile shape.
func.func @tile_typecast_shape_mismatch(%a: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<16x16, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{input and result tile shapes must match}}
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<16x16, f32>
  return %0 : !ttcore.tile<16x16, f32>
}

// -----

// ttl.tile_typecast expects tile input and tile result types.
func.func @tile_typecast_nontile_input_result(%a: i32) -> i32 {
  %c0 = arith.constant 0 : index
  // expected-error @below {{ttcore.tile type}}
  %0 = ttl.tile_typecast %a into dst[%c0] : i32 -> i32
  return %0 : i32
}

// -----

// Identity tile typecasts are not meaningful and should be rejected.
func.func @tile_typecast_identity(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{input and result tile data types must differ}}
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// Integer inputs are not supported by the SFPU typecast_tile path.
func.func @tile_typecast_int_to_float(%a: !ttcore.tile<32x32, si32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{only supports floating-point tile data types}}
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, si32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// Integer outputs are not supported by the SFPU typecast_tile path.
func.func @tile_typecast_float_to_int(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, si32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{only supports floating-point tile data types}}
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, si32>
  return %0 : !ttcore.tile<32x32, si32>
}
