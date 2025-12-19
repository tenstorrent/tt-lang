// RUN: ttlang-opt %s --split-input-file -verify-diagnostics

// CHECK-LABEL: func.func @mismatched_types
func.func @mismatched_types(%t: !ttcore.tile<32x32, f32>, %dst: index) {
  // expected-error @+1 {{expected ttl.dst}}
  %0 = ttl.copy_tile %t, %dst, %dst : !ttcore.tile<32x32, f32>, index, index -> !ttcore.tile<32x32, bf16>
  func.return
}

// -----

// CHECK-LABEL: func.func @non_tile_src
func.func @non_tile_src(%t: tensor<1xf32>, %dst: index) {
  // expected-error @+1 {{'ttl.copy_tile' op operand #0 must be ttcore.tile type}}
  %0 = ttl.copy_tile %t, %dst, %dst : tensor<1xf32>, index, index -> !ttl.dst
  func.return
}

// -----

// CHECK-LABEL: func.func @non_index_dst
func.func @non_index_dst(%t: !ttcore.tile<32x32, f32>, %dst: i32) {
  // expected-error @+1 {{expected builtin.index}}
  %0 = ttl.copy_tile %t, %dst, %dst : !ttcore.tile<32x32, f32>, i32, i32 -> !ttl.dst
  func.return
}
