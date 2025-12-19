// RUN: ttlang-opt %s --split-input-file -verify-diagnostics

// Test: src must be a ttcore.tile type.
func.func @non_tile_src(%t: tensor<1xf32>, %idx: index) {
  // expected-error @+1 {{'ttl.copy_tile' op operand #0 must be ttcore.tile type, but got 'tensor<1xf32>'}}
  %0 = ttl.copy_tile %t, %idx, %idx : tensor<1xf32>, index, index -> !ttl.dst
  func.return
}

// -----

// Test: result must be !ttl.dst type.
func.func @mismatched_result_type(%t: !ttcore.tile<32x32, f32>, %idx: index) {
  // expected-error @+1 {{custom op 'ttl.copy_tile' invalid kind of type specified}}
  %0 = ttl.copy_tile %t, %idx, %idx : !ttcore.tile<32x32, f32>, index, index -> !ttcore.tile<32x32, bf16>
  func.return
}

// -----

// Test: src_index must have index type.
func.func @non_index_src_idx(%t: !ttcore.tile<32x32, f32>, %idx: i32) {
  // expected-error @+1 {{custom op 'ttl.copy_tile' invalid kind of type specified}}
  %0 = ttl.copy_tile %t, %idx, %idx : !ttcore.tile<32x32, f32>, i32, i32 -> !ttl.dst
  func.return
}
