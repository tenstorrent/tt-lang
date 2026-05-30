// Summary: Mixed-dtype tile arguments are accepted when the compute body
// performs an explicit dtype conversion via ttl.tile_typecast. The companion
// negative test (invalid/dst_typecast_stray_mixed_dtypes_invalid.mlir)
// covers the case where an unrelated mixed-dtype input is used directly.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst),canonicalize,cse)' | FileCheck %s

#idx_map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: typecast bf16 -> f32 is allowed even though the compute op has
// both bf16 and f32 tile block-arguments. DST capacity falls back to the
// conservative f32 size (4 tiles in default double-buffered mode).
// CHECK-LABEL: func.func @typecast_bf16_to_f32
// CHECK: ttl.compute
// CHECK: ttl.tile_typecast {{.*}} into dst[%c0]
// CHECK: ttl.tile_store
func.func @typecast_bf16_to_f32(%in: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cbin = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %in_cb = ttl.attach_cb %in, %cbin
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cbout
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%in_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#idx_map, #idx_map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in_arg: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %c0 = arith.constant 0 : index
      %dtok, %dtile = ttl.copy_tile %in_arg[%c0] into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %cast = ttl.tile_typecast %dtile into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %cast, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
