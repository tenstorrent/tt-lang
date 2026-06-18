// Summary: Mixed input tile dtypes are rejected when only part of the mix is an explicit typecast.
// RUN: ttlang-opt %s --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{enable-fpu-binary-ops=1 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst))'

#idx_map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: ttl.tile_typecast permits the specific input it converts, but an
// unrelated f32 input in the same compute body must not make the mixed-dtype
// DST capacity decision look safe.
func.func @typecast_with_stray_mixed_input(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cba
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cbout
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{mixed f32 and non-f32 tile arguments}}
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#idx_map, #idx_map, #idx_map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %c0 = arith.constant 0 : index
      %cast = ttl.tile_typecast %b_arg into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
      %use_stray = ttl.tile_add %a_arg, %a_arg into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %sum = ttl.tile_add %cast, %use_stray into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %sum, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
