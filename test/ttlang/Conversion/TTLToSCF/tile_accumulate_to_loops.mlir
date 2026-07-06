// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops))' | FileCheck %s --check-prefix=LOOPS
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' | FileCheck %s --check-prefix=TTK

// Summary: Verifies `ttl.tile_accumulate` initializes DST before the reduction
// loop and lowers to TTKernel binary_dest_reuse_tiles.

#init_map = affine_map<(d0, d1, d2) -> (d0, d1)>
#contrib_map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

// LOOPS-LABEL: func.func @tile_accumulate_reduction
// LOOPS: ttl.dst_section {
// LOOPS: %[[INIT_TILE:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}]
// LOOPS: %{{.*}}, %[[INIT_COPY:.*]] = ttl.copy_tile %[[INIT_TILE]]{{.*}} into dst[%[[ACC:.*]]]
// LOOPS: scf.for %[[K:.*]] =
// LOOPS: %[[CONTRIB_TILE:.*]] = tensor.extract %{{.*}}[%[[K]], %{{.*}}, %{{.*}}]
// LOOPS: %{{.*}}, %[[CONTRIB_COPY:.*]] = ttl.copy_tile %[[CONTRIB_TILE]]
// LOOPS: ttl.tile_accumulate %[[INIT_COPY]], %[[CONTRIB_COPY]] add into dst[%[[ACC]]]
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
