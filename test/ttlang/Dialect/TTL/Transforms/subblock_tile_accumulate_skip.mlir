// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-subblock-compute-for-dst))' | FileCheck %s

// Summary: Verifies subblocking skips non-matmul accumulating computes.

#init_map = affine_map<(d0, d1, d2) -> (d0, d1)>
#contrib_map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @skip_tile_accumulate
// CHECK-NOT: scf.for
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 2
// CHECK: ttl.tile_accumulate
func.func @skip_tile_accumulate() {
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
       iterator_types = ["parallel", "parallel", "reduction"],
       ttl.unroll_factor = 2 : i64} {
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
  func.return
}
