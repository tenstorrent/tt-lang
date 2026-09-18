// Summary: Verify compute configuration captures finalized physical DFB indices.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false},ttl-set-compute-kernel-config)' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// A user DFB in another kernel sets the end of the user index range.
func.func @global_user_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 5 : i32,
                ttl.crta_indices = []} {
  %user1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %user2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %user3 = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %user4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// The compiler DFB enters finalization with kernel-local provisional index
// 1. Finalization moves it to physical index 5 before compute configuration
// records the SFPU f32 input index.

// CHECK: module attributes {ttl.dfb_allocations = {{.*}}}
// CHECK-LABEL: func.func @compiler_f32_sfpu
// CHECK-SAME: ttl.base_cta_index = 6 : i32
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 5>
// CHECK: %[[COMPILER_DFB:.*]] = ttl.bind_cb{cb_index = 5, {{.*}}} {dfb_id = 5 : index, ttl.compiler_allocated}
func.func @compiler_f32_sfpu(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %output_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %compiler_reserve = ttl.cb_reserve %compiler_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %input, %compiler_reserve
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %compiler_wait = ttl.cb_wait %compiler_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %input_attached = ttl.attach_cb %compiler_wait, %compiler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_attached = ttl.attach_cb %init, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%input_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
         %output_tile: !ttcore.tile<32x32, f32>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      %exp = ttl.tile_exp %input_tile into dst[%c0]
          : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %exp, %output_view[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, f32>,
            tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  ttl.cb_pop %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
