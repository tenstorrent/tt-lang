// Verifies mechanical multi-stage inlining and the ordinary materialized
// implementation of an inter-stage scalar edge.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-compute-pipelines))' | FileCheck %s --check-prefix=INLINE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-compute-pipelines,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-insert-cb-sync))' | FileCheck %s --check-prefix=MATERIALIZED

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // A consumer pipeline is analyzed before its producer is rewritten. The
  // resulting tensor operations retain the two-operation SSA chain.
  // INLINE-LABEL: func.func @pipeline_dependency_order
  // INLINE:       %[[FIRST:.*]] = ttl.exp
  // INLINE-NEXT:  %[[SECOND:.*]] = ttl.exp %[[FIRST]]
  // INLINE-NOT:   ttl.compute_pipeline
  // INLINE-NOT:   ttl.compute_stage
  // INLINE:       ttl.store %[[SECOND]]
  func.func @pipeline_dependency_order()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %first = ttl.compute_pipeline ins(%input)
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
            %result = ttl.exp %stage_input
                : tensor<1x1x!ttcore.tile<32x32, bf16>>
                  -> tensor<1x1x!ttcore.tile<32x32, bf16>>
            ttl.compute_stage_yield %result
                : tensor<1x1x!ttcore.tile<32x32, bf16>>
        }
        ttl.compute_pipeline_yield %stage
            : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    %second = ttl.compute_pipeline ins(%first)
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
            %result = ttl.exp %stage_input
                : tensor<1x1x!ttcore.tile<32x32, bf16>>
                  -> tensor<1x1x!ttcore.tile<32x32, bf16>>
            ttl.compute_stage_yield %result
                : tensor<1x1x!ttcore.tile<32x32, bf16>>
        }
        ttl.compute_pipeline_yield %stage
            : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %second, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }

  // The generic implementation retains an externally published scalar while
  // materializing the same scalar for the row consumer.
  // INLINE-LABEL:      func.func @materialized_scalar_edge
  // INLINE:            %[[SUM:.*]] = ttl.reduce
  // INLINE:            %[[BROADCAST:.*]] = ttl.block.broadcast %[[SUM]]
  // INLINE:            %[[ROW:.*]] = ttl.mul {{.*}}, %[[BROADCAST]]
  // INLINE-NOT:        ttl.compute_pipeline
  // INLINE:            ttl.store %[[ROW]]
  // INLINE:            ttl.store %[[SUM]]
  // MATERIALIZED-LABEL: func.func @materialized_scalar_edge
  // MATERIALIZED:       %[[PRODUCT_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 4],
  // MATERIALIZED-NEXT:  %[[SCALER_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 1],
  // MATERIALIZED-NEXT:  %[[SCALAR_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 1],
  // MATERIALIZED:       ttl.compute
  // MATERIALIZED:       %[[SCALAR_WAIT:.*]] = ttl.cb_wait %[[SCALAR_DFB]]
  // MATERIALIZED-NEXT:  %[[SCALAR:.*]] = ttl.attach_cb %[[SCALAR_WAIT]], %[[SCALAR_DFB]]
  // MATERIALIZED:       ttl.compute ins({{.*}}, %[[SCALAR]]
  // MATERIALIZED-NOT:   ttl.compute_pipeline
  // MATERIALIZED-NOT:   ttl.compute_stage
  func.func @materialized_scalar_edge()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %row_output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %scalar_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
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
    %row_output = ttl.cb_reserve %row_output_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.store %normalized, %row_output
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
    %scalar_output = ttl.cb_reserve %scalar_output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %scalar, %scalar_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}
