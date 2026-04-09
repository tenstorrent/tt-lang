// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}))' \
// RUN:   | FileCheck %s
//
// Verify that f32 tiles get the correct DST capacity (4, not the bf16
// default of 8). With dstPerIteration = 2 (SFPU binary: copy lhs + copy rhs),
// unroll_factor should be floor(4 / 2) = 2.
//
// Before the fix, computeDSTCapacity fell through to the bf16 default
// capacity of 8 when no system descriptor was present, giving
// unroll_factor=4 and causing the subblock pass to over-provision DST.

// CHECK-LABEL: func.func @f32_capacity
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 2

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_capacity()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs_ready = ttl.cb_wait %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<2x3x!ttcore.tile<32x32, f32>>,
                        tensor<2x3x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<2x3x!ttcore.tile<32x32, f32>>)
      {fp32_dest_acc_en = true,
       indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, f32>,
       %rhs_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %tanh = ttl.tile_tanh %sum into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %tanh, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  return
}
