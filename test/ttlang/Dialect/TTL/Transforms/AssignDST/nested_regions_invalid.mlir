// Verifies ttl-assign-dst diagnoses nested DST operations before allocation.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst))' --verify-diagnostics --split-input-file

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // Reject copy insertion whose consumers span the compute body and a nested
  // region before sorting operations from different blocks.
  func.func @cross_region_copy_insertion(
      %input: tensor<1x1x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>, %condition: i1)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %attached_input = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %attached_output = ttl.attach_cb %empty, %output_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %reserved_output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = ttl.compute
        ins(%attached_input : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%attached_output : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
         %output_tile: !ttcore.tile<32x32, f32>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      ttl.tile_store %output_tile, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      %direct = ttl.tile_exp %input_tile into dst[%c0]
          : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %direct, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, f32>,
            tensor<1x1x!ttcore.tile<32x32, f32>>
      scf.if %condition {
        // expected-error @below {{'ttl.tile_exp' op nested DST consumer requires copy insertion by ttl-assign-dst; expected the operation directly in the ttl.compute body}}
        %nested = ttl.tile_exp %input_tile into dst[%c0]
            : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
        ttl.tile_store %nested, %reserved_output[%row, %column] from dst[%c0]
            : !ttcore.tile<32x32, f32>,
              tensor<1x1x!ttcore.tile<32x32, f32>>
        scf.yield
      }
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // Reject a nested non-in-place operation whose source DST indices are not
  // represented by its operands.
  func.func @nested_unresolved_source_dst_indices(
      %input: tensor<1x1x!ttcore.tile<32x32, f32>>,
      %output: tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %attached_input = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %attached_output = ttl.attach_cb %empty, %output_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %reserved_output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = ttl.compute
        ins(%attached_input : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%attached_output : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
         %output_tile: !ttcore.tile<32x32, f32>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      ttl.tile_store %output_tile, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      scf.for %iteration = %c0 to %c1 step %c1 {
        // expected-error @below {{'ttl.tile_add' op nested DST operation requires source DST index resolution by ttl-assign-dst; expected the operation directly in the ttl.compute body}}
        %nested = ttl.tile_add %input_tile, %output_tile into dst[%c0]
            {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>}
            : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
            -> !ttcore.tile<32x32, f32>
        ttl.tile_store %nested, %reserved_output[%row, %column] from dst[%c0]
            : !ttcore.tile<32x32, f32>,
              tensor<1x1x!ttcore.tile<32x32, f32>>
      }
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
  }
}
