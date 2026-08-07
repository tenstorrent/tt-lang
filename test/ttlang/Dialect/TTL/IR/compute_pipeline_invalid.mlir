// Verifies structural, domain, purity, and isolation errors for multi-stage
// tensor compute graphs.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// A pipeline block has one argument for each explicit input.
func.func @pipeline_block_argument_count(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op body requires one block argument per input, got 2 block arguments for 1 inputs}}
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%first: tensor<1x1x!ttcore.tile<32x32, bf16>>,
         %second: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      ttl.compute_pipeline_yield %first
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Every pipeline result has one explicit yielded graph value.
func.func @pipeline_yield_count(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op body must yield one value per result, got 1 yielded values for 2 results}}
  %results:2 = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Tensor operations belong to a stage rather than directly to the pipeline.
func.func @pipeline_direct_operation(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op body may contain only ttl.compute_stage operations; found ttl.exp}}
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %value = ttl.exp %pipeline_input
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.compute_pipeline_yield %value
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Pipeline results identify stage results, not uncomputed input values.
func.func @pipeline_passthrough_result(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op yielded value 1 must be a result of a stage in this pipeline}}
  %results:2 = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage, %pipeline_input
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Every planned stage result has a consumer or external result mapping.
func.func @unused_stage_result(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op stage result 0 is unused}}
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %unused = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %used = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %used
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A stage block has one tensor argument for each explicit stage input.
func.func @stage_block_argument_count(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      // expected-error @below {{'ttl.compute_stage' op body requires one block argument per input, got 2 block arguments for 1 inputs}}
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%first: tensor<1x1x!ttcore.tile<32x32, bf16>>,
             %second: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %first
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Every stage result has one explicit yielded tensor value.
func.func @stage_yield_count(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %results:2 = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      // expected-error @below {{'ttl.compute_stage' op body must yield one value per result, got 1 yielded values for 2 results}}
      %stage:2 = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage#0, %stage#1
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A reduced dimension cannot occur in a stage result indexing map.
func.func @result_references_reduction_dimension(
    %input: tensor<1x4x!ttcore.tile<32x32, bf16>>) {
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x4x!ttcore.tile<32x32, bf16>>):
      // expected-error @below {{'ttl.compute_stage' op result 0 indexing map cannot reference reduction dimension 1}}
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, dim1)>]
          iterator_types = ["parallel", "reduction"]
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
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Every reduction dimension needs an input extent source.
func.func @unreferenced_reduction_dimension(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      // expected-error @below {{'ttl.compute_stage' op reduction dimension 1 must be referenced by at least one input indexing map}}
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, 0)>,
                           affine_map<(dim0, dim1) -> (dim0, 0)>]
          iterator_types = ["parallel", "reduction"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Instrumentation cannot be represented as a pure stage operation.
func.func @impure_stage_operation(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      // expected-error @below {{'ttl.compute_stage' op stage operations must be pure and speculatable; found ttl.dprint}}
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          "ttl.dprint"() {fmt = "inside stage", mode = "scalar"} : () -> ()
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A stage receives external values only through explicit operands.
func.func @stage_capture(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %result = ttl.compute_pipeline ins(%input)
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      // expected-note @below {{required by region isolation constraints}}
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          // expected-error @below {{'ttl.add' op using value defined outside the region}}
          %value = ttl.add %stage_input, %pipeline_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A selected schedule applies only to a compiler-recognized semantic graph.
func.func @schedule_without_kind(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op selected_schedule requires a compiler-recognized pipeline_kind}}
  %result = ttl.compute_pipeline ins(%input)
      {selected_schedule = #ttl.compute_pipeline_schedule<materialized>}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A multiply-reduction kind must contain the recognized semantic operations.
func.func @invalid_multiply_reduction_kind(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op multiply_full_scalar_reduction stage requires multiply, fill, and reduction operations}}
  %result = ttl.compute_pipeline ins(%input)
      {pipeline_kind = #ttl.compute_pipeline_kind<multiply_full_scalar_reduction>}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A row-normalization kind must contain its three semantic domains.
func.func @invalid_row_normalization_kind(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{'ttl.compute_pipeline' op row_normalization requires three stages, one result, and one or two inputs}}
  %result = ttl.compute_pipeline ins(%input)
      {pipeline_kind = #ttl.compute_pipeline_kind<row_normalization>}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %stage = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %value = ttl.exp %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %stage
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A recognized row-normalization pipeline requires a positive scale.
func.func @row_normalization_negative_scale(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{'ttl.compute_pipeline' op row_normalization scale must be finite and positive}}
  %result = ttl.compute_pipeline ins(%input)
      {pipeline_kind = #ttl.compute_pipeline_kind<row_normalization>}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %reduced = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>]
          iterator_types = ["reduction", "reduction"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %square = ttl.mul %stage_input, %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %scaler = ttl.fill 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %sum = ttl.reduce %square, %scaler 0 : i32 [0, 1]
              : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
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
          %scaled = ttl.mul_unary_const %stage_scalar, -1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %epsilon = ttl.fill 1.000000e-05
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %biased = ttl.add %scaled, %epsilon
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %value = ttl.rsqrt %biased
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %normalized = ttl.compute_stage ins(%pipeline_input, %inverse)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
             %stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %broadcast = ttl.block.broadcast %stage_scalar
              dims = [0, 1], shape = [1, 1]
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %value = ttl.mul %stage_input, %broadcast
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %normalized
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// A recognized row-normalization pipeline requires a positive epsilon.
func.func @row_normalization_negative_epsilon(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{'ttl.compute_pipeline' op row_normalization epsilon must be finite and positive}}
  %result = ttl.compute_pipeline ins(%input)
      {pipeline_kind = #ttl.compute_pipeline_kind<row_normalization>}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    ^bb0(%pipeline_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %reduced = ttl.compute_stage ins(%pipeline_input)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>]
          iterator_types = ["reduction", "reduction"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %square = ttl.mul %stage_input, %stage_input
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %scaler = ttl.fill 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %sum = ttl.reduce %square, %scaler 0 : i32 [0, 1]
              : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
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
          %scaled = ttl.mul_unary_const %stage_scalar, 1.000000e+00
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %epsilon = ttl.fill -1.000000e-05
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %biased = ttl.add %scaled, %epsilon
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %value = ttl.rsqrt %biased
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %normalized = ttl.compute_stage ins(%pipeline_input, %inverse)
          indexing_maps = [affine_map<(dim0, dim1) -> (dim0, dim1)>,
                           affine_map<(dim0, dim1) -> (0, 0)>,
                           affine_map<(dim0, dim1) -> (dim0, dim1)>]
          iterator_types = ["parallel", "parallel"]
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
        ^bb0(%stage_input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
             %stage_scalar: tensor<1x1x!ttcore.tile<32x32, bf16>>):
          %broadcast = ttl.block.broadcast %stage_scalar
              dims = [0, 1], shape = [1, 1]
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %value = ttl.mul %stage_input, %broadcast
              : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                tensor<1x1x!ttcore.tile<32x32, bf16>>
                -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          ttl.compute_stage_yield %value
              : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.compute_pipeline_yield %normalized
          : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
