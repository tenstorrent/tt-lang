// Verifier tests for ttl.tensor_slice op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Index count does not match tensor rank.
func.func @index_count_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op index count (1) must match tensor rank (2)}}
  %slice = ttl.tensor_slice %t[%c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout1x1 = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>,
              memref<1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Result rank does not match tensor rank.
func.func @result_rank_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op result rank (1) must match tensor rank (2)}}
  %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x!ttcore.tile<32x32, f32>, #layout1x1>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Result element type does not match tensor element type.
func.func @element_type_mismatch(%t: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'ttl.tensor_slice' op result element type ('!ttcore.tile<32x32, bf16>') must match tensor element type ('!ttcore.tile<32x32, f32>')}}
  %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
