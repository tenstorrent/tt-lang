// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops))' | FileCheck %s --check-prefix=LOOPS
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' | FileCheck %s --check-prefix=TTK

// Summary: Verifies `ttl.compute` reductions that use `ttl.tile_accumulate`
// keep DST live across the reduction loop nest and lower to TTKernel
// binary_dest_reuse_tiles.

#init_map = affine_map<(d0, d1, d2) -> (d0, d1)>
#contrib_map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>
#init_reduction_first_map = affine_map<(d0, d1, d2) -> (d1, d2)>
#contrib_reduction_first_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out_reduction_first_map = affine_map<(d0, d1, d2) -> (d1, d2)>
#init_two_reductions_map = affine_map<(d0, d1, d2) -> (d0)>
#contrib_two_reductions_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out_two_reductions_map = affine_map<(d0, d1, d2) -> (d0)>

// LOOPS-LABEL: func.func @tile_accumulate_reduction
// LOOPS: ttl.dst_section {
// LOOPS: %[[INIT_TILE:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}]
// LOOPS: %{{.*}}, %[[INIT_COPY:.*]] = ttl.copy_tile %[[INIT_TILE]]{{.*}} into dst[%[[ACC:.*]]]
// LOOPS: scf.for %[[K:.*]] =
// LOOPS: %[[CONTRIB_TILE:.*]] = tensor.extract %{{.*}}[%[[K]], %{{.*}}, %{{.*}}]
// LOOPS-NOT: ttl.copy_tile %[[CONTRIB_TILE]]
// LOOPS: ttl.tile_accumulate %[[INIT_COPY]], %[[CONTRIB_TILE]] add into dst[%[[ACC]]]
// LOOPS: ttl.tile_store %{{.*}} from dst[%[[ACC]]]

// TTK-LABEL: func.func @tile_accumulate_reduction
// TTK: ttkernel.binary_dest_reuse_tiles_init
// TTK: ttkernel.binary_dest_reuse_tiles(%{{.*}}, %{{.*}}, %{{.*}}, <add>, <dest_to_srca>)
// TTK-NOT: ttkernel.add_binary_tile
func.func @tile_accumulate_reduction()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<3x1x1x!ttcore.tile<32x32, bf16>>
  %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<3x1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<3x1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %empty, %out_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%init, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                             tensor<3x1x1x!ttcore.tile<32x32, bf16>>)
      outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#init_map, #contrib_map, #out_map],
       iterator_types = ["parallel", "parallel", "reduction"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, bf16>,
       %contrib_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %acc = ttl.tile_accumulate %init_tile, %contrib_tile add into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %acc, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  ttl.cb_pop %contrib_cb {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  func.return
}

// Reduction dimensions may appear before parallel dimensions in the compute
// domain. Lowering still emits parallel loops outside the DST section and maps
// the reordered IVs back to the original indexing maps.

// LOOPS-LABEL: func.func @tile_accumulate_reduction_first
// LOOPS: scf.for %[[J:.*]] =
// LOOPS: scf.for %[[K:.*]] =
// LOOPS: ttl.dst_section {
// LOOPS: tensor.extract %{{.*}}[%[[J]], %[[K]]]
// LOOPS: scf.for %[[I:.*]] =
// LOOPS: tensor.extract %{{.*}}[%[[I]], %[[J]], %[[K]]]
// LOOPS: ttl.tile_accumulate
// LOOPS: ttl.tile_store %{{.*}}, %{{.*}}[%[[J]], %[[K]]]{{.*}}from dst
func.func @tile_accumulate_reduction_first()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 4>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 12} : !ttl.cb<[3, 2, 2], !ttcore.tile<32x32, bf16>, 12>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 4} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 4>

  %init_wait = ttl.cb_wait %init_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 4> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 4>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 12 : i64} : <[3, 2, 2], !ttcore.tile<32x32, bf16>, 12> -> tensor<3x2x2x!ttcore.tile<32x32, bf16>>
  %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<3x2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 2, 2], !ttcore.tile<32x32, bf16>, 12>) -> tensor<3x2x2x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %out_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 4> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %empty, %out_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 4>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%init, %contrib : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                             tensor<3x2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#init_reduction_first_map,
                        #contrib_reduction_first_map,
                        #out_reduction_first_map],
       iterator_types = ["reduction", "parallel", "parallel"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, bf16>,
       %contrib_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %j = ttl.iter_index 1 : index
    %k = ttl.iter_index 2 : index
    %acc = ttl.tile_accumulate %init_tile, %contrib_tile add into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %acc, %out_view[%j, %k] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  ttl.cb_pop %contrib_cb {num_tiles = 12 : i64} : <[3, 2, 2], !ttcore.tile<32x32, bf16>, 12>
  func.return
}

// Multiple reduction dimensions form one nested reduction loop inside the DST
// section; the final store executes after both reductions complete.

// LOOPS-LABEL: func.func @tile_accumulate_two_reduction_dims
// LOOPS: scf.for %[[I:.*]] =
// LOOPS: ttl.dst_section {
// LOOPS: tensor.extract %{{.*}}[%[[I]]]
// LOOPS: scf.for %[[J:.*]] =
// LOOPS: scf.for %[[K:.*]] =
// LOOPS: tensor.extract %{{.*}}[%[[I]], %[[J]], %[[K]]]
// LOOPS: ttl.tile_accumulate
// LOOPS: ttl.tile_store %{{.*}}, %{{.*}}[%[[I]]]{{.*}}from dst
func.func @tile_accumulate_two_reduction_dims()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 24} : !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, bf16>, 24>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x!ttcore.tile<32x32, bf16>>
  %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 24 : i64} : <[2, 3, 4], !ttcore.tile<32x32, bf16>, 24> -> tensor<2x3x4x!ttcore.tile<32x32, bf16>>
  %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<2x3x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, bf16>, 24>) -> tensor<2x3x4x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %out_cb : <[2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<2x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %empty, %out_cb : (tensor<2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%init, %contrib : tensor<2x!ttcore.tile<32x32, bf16>>,
                             tensor<2x3x4x!ttcore.tile<32x32, bf16>>)
      outs(%out : tensor<2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#init_two_reductions_map,
                        #contrib_two_reductions_map,
                        #out_two_reductions_map],
       iterator_types = ["parallel", "reduction", "reduction"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, bf16>,
       %contrib_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %acc = ttl.tile_accumulate %init_tile, %contrib_tile add into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %acc, %out_view[%i] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x!ttcore.tile<32x32, bf16>>

  ttl.cb_pop %contrib_cb {num_tiles = 24 : i64} : <[2, 3, 4], !ttcore.tile<32x32, bf16>, 24>
  func.return
}
