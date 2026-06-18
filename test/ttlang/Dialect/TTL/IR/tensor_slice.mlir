// Verifier-positive (round-trip) tests for ttl.tensor_slice op.
// RUN: ttlang-opt %s | FileCheck %s

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Rank-reducing slice: a rank-3 source sliced to a rank-2 result. The leading
// dim is squeezed by a scalar index; the trailing two dims map to the result.
// CHECK-LABEL: func.func @rank_reducing_slice
// CHECK: ttl.tensor_slice
// CHECK-SAME: tensor<2x2x2x!ttcore.tile<32x32, f32>
// CHECK-SAME: -> tensor<2x2x!ttcore.tile<32x32, f32>
func.func @rank_reducing_slice(%t: tensor<2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %slice = ttl.tensor_slice %t[%c1, %c0, %c0]
      : tensor<2x2x2x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  func.return
}
