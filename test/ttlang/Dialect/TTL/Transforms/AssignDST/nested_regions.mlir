// Verifies ttl-assign-dst preserves resolved nested DST operations that do not
// require compute-body allocation or materialization.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Preserve resolved nested DFB readers and destination writes in conditionals
// and loops.
// CHECK-LABEL: func.func @resolved_nested_dst_operations
// CHECK: scf.if
// CHECK-NEXT: %[[FILL:.*]] = ttl.tile_fill
// CHECK-NEXT: ttl.tile_store %[[FILL]],
// CHECK-NEXT: %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT: ttl.tile_store %[[BCAST]],
// CHECK-NEXT: %{{.*}}, %[[COPY:.*]] = ttl.copy_tile
// CHECK-NEXT: ttl.tile_store %[[COPY]],
// CHECK-NEXT: %[[REDUCE:.*]] = ttl.tile_reduce
// CHECK-NEXT: ttl.tile_store %[[REDUCE]],
// CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp
// CHECK-NEXT: ttl.tile_store %[[EXP]],
// CHECK-NEXT: }
// CHECK: scf.for
// CHECK-NEXT: %[[LOOP_FILL:.*]] = ttl.tile_fill
// CHECK-NEXT: ttl.tile_store %[[LOOP_FILL]],
// CHECK-NEXT: }
func.func @resolved_nested_dst_operations(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %output: tensor<1x1x!ttcore.tile<32x32, bf16>>, %condition: i1)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %attached_input = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_output = ttl.attach_cb %empty, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserved_output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%attached_input : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%attached_output : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<32x32, bf16>,
       %output_tile: !ttcore.tile<32x32, bf16>):
    %row = ttl.iter_index 0 : index
    %column = ttl.iter_index 1 : index
    %pad = ttl.tile_fill 0.0 into dst[%c0] : !ttcore.tile<32x32, bf16>
    ttl.tile_store %pad, %reserved_output[%row, %column] from dst[%c0]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      %filled = ttl.tile_fill 1.0 into dst[%c0]
          : !ttcore.tile<32x32, bf16>
      ttl.tile_store %filled, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      %broadcast = ttl.tile_bcast %input_tile, %input_tile 1 : i32
          into dst[%c0] : (!ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %broadcast, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      %token, %copied = ttl.copy_tile %input_tile[%row, %column] into dst[%c0]
          : !ttcore.tile<32x32, bf16>
          -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.tile_store %copied, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      %reduced = ttl.tile_reduce %input_tile, %input_tile, %input_tile 0 : i32
          <reduce_dim_row> into dst[%c0] : (!ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %reduced, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      %exponential = ttl.tile_exp %input_tile into dst[%c0]
          : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %exponential, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield
    }
    scf.for %iteration = %c0 to %c1 step %c1 {
      %loop_fill = ttl.tile_fill 2.0 into dst[%c0]
          : !ttcore.tile<32x32, bf16>
      ttl.tile_store %loop_fill, %reserved_output[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
