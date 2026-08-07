// Verifies multi-stage tensor graphs with distinct reduction and parallel
// domains and more than one live pipeline result.
// RUN: ttlang-opt %s | FileCheck %s

// The reduced scalar feeds the row stage and remains a pipeline result. The
// row result and scalar result therefore have distinct types and domains.
// CHECK-LABEL: func.func @reduction_scalar_and_row_live_outs
// CHECK:       %[[PIPELINE_RESULTS:.*]]:2 = ttl.compute_pipeline
// CHECK:         %[[SCALAR:.*]] = ttl.compute_stage
// CHECK-SAME:      iterator_types = ["reduction", "reduction"]
// CHECK:           ttl.compute_stage_yield
// CHECK:         %[[ROW:.*]] = ttl.compute_stage
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK:           ttl.compute_stage_yield
// CHECK:         ttl.compute_pipeline_yield %[[ROW]], %[[SCALAR]]
func.func @reduction_scalar_and_row_live_outs(
    %input: tensor<1x4x!ttcore.tile<32x32, bf16>>)
    -> (tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %normalized, %scalar = ttl.compute_pipeline ins(%input)
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x4x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %reduced = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>]
          iterator_types = ["reduction", "reduction"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
          %product = ttl.mul %stage_input, %stage_input
              : tensor<1x4x!ttcore.tile<32x32, bf16>>,
                tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          %scaler = ttl.fill 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %sum = ttl.reduce %product, %scaler 0 : i32 [0, 1]
              : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %sum
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %row = ttl.compute_stage ins(%pipeline_input, %reduced)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x4x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>,
             %stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %broadcast = ttl.block.broadcast %stage_scalar
              dims = [0, 1], shape = [1, 4]
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          %result = ttl.mul %stage_input, %broadcast
              : tensor<1x4x!ttcore.tile<32x32, bf16>>,
                tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %result
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %row, %reduced
          : tensor<1x4x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return %normalized, %scalar
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
}
