// RUN: ttlang-opt %s --split-input-file -verify-diagnostics

// Test: src must be a ttcore.tile type.
func.func @non_tile_src(%t: tensor<1xf32>, %idx: index) {
  // expected-error @+1 {{'ttl.copy_tile' op operand #0 must be ttcore.tile type, but got 'tensor<1xf32>'}}
  %0, %1 = ttl.copy_tile %t, %idx, %idx : tensor<1xf32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  func.return
}

// -----

// Test: malformed result list should trigger a parse error.
func.func @mismatched_result_type(%t: !ttcore.tile<32x32, f32>, %idx: index) {
  // expected-error @+1 {{expected ','}}
  %0 = ttl.copy_tile %t, %idx, %idx : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst
  func.return
}

// -----

// Test: dst_tile type must match src type.
func.func @dst_tile_type_mismatch(%t: !ttcore.tile<32x32, f32>, %idx: index) {
  // expected-error @+1 {{dst_tile type must match src type, but got dst_tile: '!ttcore.tile<32x32, bf16>', src: '!ttcore.tile<32x32, f32>'}}
  %0, %1 = ttl.copy_tile %t, %idx, %idx : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
  func.return
}
