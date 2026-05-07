// Verifier tests for ttl.tile_typecast: the tile shape must be preserved
// (only the element data type may differ between input and result).
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
