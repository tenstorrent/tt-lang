// Verifies rejection of malformed source-scalar resource scopes.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// A retained value must represent the complete scalar tensor domain.
func.func @non_scalar_producer(
    %input: tensor<1x4x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{'ttl.source_scalar_scope' op producer region must yield a static full-scalar tensor}}
  %result = ttl.source_scalar_scope ins(%input)
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>> producer {
    ^bb0(%producer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %row = ttl.compute_stage ins(%producer_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
          %value = ttl.rsqrt %stage_input
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %row
          : tensor<1x4x!ttcore.tile<32x32, bf16>>
  } consumer {
    ^bb0(%source_scalar: tensor<1x4x!ttcore.tile<32x32, bf16>>,
         %consumer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %row = ttl.compute_stage ins(%source_scalar)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_scalar: tensor<1x4x!ttcore.tile<32x32, bf16>>):
          %value = ttl.rsqrt %stage_scalar
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
                -> tensor<1x4x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x4x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %row
          : tensor<1x4x!ttcore.tile<32x32, bf16>>
  }
  return %result : tensor<1x4x!ttcore.tile<32x32, bf16>>
}

// -----

// The source-scalar block argument cannot become a tensor result without an
// explicit materialization operation.
func.func @scalar_escapes(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{'ttl.source_scalar_scope' op consumer source scalar may be used only as a compute-stage input}}
  %result = ttl.source_scalar_scope ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>> producer {
    ^bb0(%producer_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %scalar = ttl.compute_stage ins(%producer_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.rsqrt %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %scalar
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } consumer {
    ^bb0(%source_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>,
         %consumer_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %unused = ttl.compute_stage ins(%source_scalar)
          indexing_maps = [affine_map<(dim0, dim1) -> (0, 0)>,
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
      ttl.source_scalar_yield %source_scalar
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// A row consumer must read a retained scalar with a full-broadcast map.
func.func @non_scalar_consumer_map(
    %input: tensor<1x4x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{'ttl.source_scalar_scope' op consumer source-scalar inputs require a zero-indexed full-scalar map}}
  %result = ttl.source_scalar_scope ins(%input)
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>> producer {
    ^bb0(%producer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %scalar = ttl.compute_stage ins(%producer_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>]
          iterator_types = ["reduction", "reduction"]
          : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
          %scaler = ttl.fill 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %sum = ttl.reduce %stage_input, %scaler 0 : i32 [0, 1]
              : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %sum
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.source_scalar_yield %scalar
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } consumer {
    ^bb0(%source_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>,
         %consumer_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      %row = ttl.compute_stage ins(%source_scalar, %consumer_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             tensor<1x4x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x4x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>,
             %stage_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
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
