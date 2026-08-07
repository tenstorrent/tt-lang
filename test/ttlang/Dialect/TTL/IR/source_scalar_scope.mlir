// Verifies the structured producer-consumer lifetime for one retained full
// scalar and its storage-neutral lowering to tensor SSA.
// RUN: ttlang-opt %s | FileCheck %s --check-prefix=SCOPE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-source-scalar-scopes))' | FileCheck %s --check-prefix=INLINE

// The producer finalizes a scalar before the consumer broadcasts it across a
// row. The scope boundary carries only that scalar between the two regions.
// SCOPE-LABEL: func.func @source_scalar_scope
// SCOPE:       %[[RESULT:.*]] = ttl.source_scalar_scope
// SCOPE:       producer {
// SCOPE-NEXT:  ^bb0(%[[PRODUCER_INPUT:.*]]: tensor<1x4x!ttcore.tile<32x32, bf16>>):
// SCOPE-NEXT:    %[[REDUCED:.*]] = ttl.compute_stage
// SCOPE:         %[[INVERSE:.*]] = ttl.compute_stage
// SCOPE:         ttl.source_scalar_yield %[[INVERSE]]
// SCOPE-NEXT:  } consumer {
// SCOPE-NEXT:  ^bb0(%[[SOURCE_SCALAR:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>, %[[CONSUMER_INPUT:.*]]: tensor<1x4x!ttcore.tile<32x32, bf16>>):
// SCOPE-NEXT:    %[[ROW:.*]] = ttl.compute_stage ins(%[[CONSUMER_INPUT]], %[[SOURCE_SCALAR]])
// SCOPE:         ttl.source_scalar_yield %[[ROW]]
// SCOPE-NEXT:  }
// SCOPE:       return %[[RESULT]]

// INLINE-LABEL: func.func @source_scalar_scope
// INLINE-NOT:   ttl.source_scalar_scope
// INLINE-NOT:   ttl.compute_stage
// INLINE:       %[[SQUARE:.*]] = ttl.mul %arg0, %arg0
// INLINE-NEXT:  %[[SCALER:.*]] = ttl.fill
// INLINE-NEXT:  %[[REDUCED:.*]] = ttl.reduce %[[SQUARE]], %[[SCALER]]
// INLINE-NEXT:  %[[INVERSE:.*]] = ttl.rsqrt %[[REDUCED]]
// INLINE-NEXT:  %[[BROADCAST:.*]] = ttl.block.broadcast %[[INVERSE]]
// INLINE-NEXT:  %[[RESULT:.*]] = ttl.mul %arg0, %[[BROADCAST]]
// INLINE-SAME:  ttl.selected_compute_pipeline_schedule = #ttl.compute_pipeline_schedule<retained_scalar>
// INLINE-NEXT:  return %[[RESULT]]
func.func @source_scalar_scope(
    %input: tensor<1x4x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
  %result = ttl.source_scalar_scope ins(%input)
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>> producer {
    ^bb0(%producer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %reduced = ttl.compute_stage ins(%producer_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>]
          iterator_types = ["reduction", "reduction"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
          %square = ttl.mul %stage_input, %stage_input
              : tensor<1x4x!ttcore.tile<32x32, bf16>>,
                tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          %scaler = ttl.fill 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %sum = ttl.reduce %square, %scaler 0 : i32 [0, 1]
              : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %sum
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %inverse = ttl.compute_stage ins(%reduced)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.rsqrt %stage_scalar
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %inverse
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } consumer {
    ^bb0(%source_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>,
         %consumer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %row = ttl.compute_stage ins(%consumer_input, %source_scalar)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>,
             %stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %broadcast = ttl.block.broadcast %stage_scalar
              dims = [0, 1], shape = [1, 4]
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          %value = ttl.mul %stage_input, %broadcast
              : tensor<1x4x!ttcore.tile<32x32, bf16>>,
                tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %row
          : tensor<1x4x!ttcore.tile<32x32, bf16>>
  }
  return %result : tensor<1x4x!ttcore.tile<32x32, bf16>>
}
