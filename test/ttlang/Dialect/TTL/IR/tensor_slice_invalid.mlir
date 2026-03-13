// Verifier tests for ttl.tensor_slice op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Index count does not match tensor rank.
func.func @index_count_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op index count (1) must match tensor rank (2)}}
  %slice = ttl.tensor_slice %t[%c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  func.return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Result rank does not match tensor rank.
func.func @result_rank_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op result rank (1) must match tensor rank (2)}}
  %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x!ttcore.tile<32x32, f32>, #layout>
  func.return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Result element type does not match tensor element type.
func.func @element_type_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op result element type ('!ttcore.tile<32x32, bf16>') must match tensor element type ('!ttcore.tile<32x32, f32>')}}
  %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
