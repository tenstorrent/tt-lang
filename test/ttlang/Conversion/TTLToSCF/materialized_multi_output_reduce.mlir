// Verifies that materializing a published reduction adds a second compute
// output and that accumulating-loop lowering selects each formal output map.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst,ttl-lower-to-loops{dst-accumulation=true}))' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst,ttl-lower-to-loops{dst-accumulation=true}))' | FileCheck %s --check-prefix=NO-COMPUTE

// NO-COMPUTE: module
// NO-COMPUTE-NOT: ttl.compute

// CHECK-LABEL: func.func @published_reduce_broadcast
// CHECK: %[[PUBLISHED_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
// CHECK: %[[BROADCAST_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
// CHECK: %[[MATERIALIZED_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK: %[[PUBLISHED_VIEW:.*]] = ttl.cb_reserve %[[PUBLISHED_DFB]]
// CHECK: %[[MATERIALIZED_VIEW:.*]] = ttl.cb_reserve %[[MATERIALIZED_DFB]]
// CHECK: scf.for
// CHECK-NEXT: ttl.dst_section {
// CHECK-NEXT: scf.for
// CHECK: %[[REDUCED:.*]] = ttl.tile_reduce
// CHECK: }
// CHECK-NEXT: %[[PUBLISHED_TILE:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: ttl.tile_store %[[PUBLISHED_TILE]], %[[PUBLISHED_VIEW]]
// CHECK-NEXT: %[[MATERIALIZED_TILE:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: ttl.tile_store %[[MATERIALIZED_TILE]], %[[MATERIALIZED_VIEW]]
// CHECK-NEXT: }
// CHECK: ttl.cb_push %[[MATERIALIZED_DFB]]
// CHECK: %[[MATERIALIZED_WAIT:.*]] = ttl.cb_wait %[[MATERIALIZED_DFB]]
// CHECK: ttl.tile_bcast
// CHECK: ttl.tile_store {{.*}}, {{.*}}
func.func @published_reduce_broadcast()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %published_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %broadcast_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %input, %scaler 0 : i32 [1]
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %broadcast = ttl.block.broadcast %reduced dims = [-1], shape = [1, 4]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %broadcast_output = ttl.cb_reserve %broadcast_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  ttl.store %broadcast, %broadcast_output
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<32x32, bf16>>
  return
}

// An accumulating compute may publish to outputs of different rank. Each
// post-reduction store must use the map paired with its own formal output.

// CHECK-LABEL: func.func @different_output_maps
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[ROW:[A-Za-z0-9_]+]] =
// CHECK-NEXT: ttl.dst_section {
// CHECK-NEXT: scf.for
// CHECK: ttl.tile_reduce
// CHECK: }
// CHECK-NEXT: %[[MATRIX_TILE:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: ttl.tile_store %[[MATRIX_TILE]], %{{.*}}[%[[ROW]], %[[ZERO]]]
// CHECK-NEXT: %[[VECTOR_TILE:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: ttl.tile_store %[[VECTOR_TILE]], %{{.*}}[%[[ROW]]]
// CHECK-NEXT: }
func.func @different_output_maps()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %matrix_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %vector_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
  %input_tensor = tensor.empty()
      : tensor<2x3x!ttcore.tile<32x32, f32>>
  %scaler_tensor = tensor.empty()
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %matrix_init = tensor.empty()
      : tensor<2x1x!ttcore.tile<32x32, f32>>
  %vector_init = tensor.empty()
      : tensor<2x!ttcore.tile<32x32, f32>>
  %input = ttl.attach_cb %input_tensor, %input_dfb
      : (tensor<2x3x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %scaler = ttl.attach_cb %scaler_tensor, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %matrix_output = ttl.attach_cb %matrix_init, %matrix_dfb
      : (tensor<2x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %vector_output = ttl.attach_cb %vector_init, %vector_dfb
      : (tensor<2x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x!ttcore.tile<32x32, f32>>
  %matrix_view = ttl.cb_reserve %matrix_dfb
      : <[2, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %vector_view = ttl.cb_reserve %vector_dfb
      : <[2], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x!ttcore.tile<32x32, f32>>
  %matrix_result, %vector_result = ttl.compute
      ins(%input, %scaler
          : tensor<2x3x!ttcore.tile<32x32, f32>>,
            tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%matrix_output, %vector_output
           : tensor<2x1x!ttcore.tile<32x32, f32>>,
             tensor<2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (0, 0)>,
                        affine_map<(d0, d1) -> (d0, 0)>,
                        affine_map<(d0, d1) -> (d0)>],
       iterator_types = ["parallel", "reduction"]} {
  ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
       %scaler_tile: !ttcore.tile<32x32, f32>,
       %matrix_tile: !ttcore.tile<32x32, f32>,
       %vector_tile: !ttcore.tile<32x32, f32>):
    %row = ttl.iter_index 0 : index
    %zero = arith.constant 0 : index
    %reduced = ttl.tile_reduce %input_tile, %scaler_tile, %matrix_tile
        0 : i32 <reduce_dim_col> into dst[%zero]
        : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>,
           !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    ttl.tile_store %reduced, %matrix_view[%row, %zero] from dst[%zero]
        : !ttcore.tile<32x32, f32>,
          tensor<2x1x!ttcore.tile<32x32, f32>>
    ttl.tile_store %reduced, %vector_view[%row] from dst[%zero]
        : !ttcore.tile<32x32, f32>, tensor<2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x1x!ttcore.tile<32x32, f32>>,
        tensor<2x!ttcore.tile<32x32, f32>>)
  func.return
}
