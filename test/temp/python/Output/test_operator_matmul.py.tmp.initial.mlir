#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @test_matmul(%arg0: tensor<32x32xf32> {d2m.stream = false}, %arg1: tensor<32x32xf32> {d2m.stream = false}, %arg2: tensor<32x32xf32> {d2m.stream = false}) -> tensor<32x32xf32> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = d2m.to_layout %arg1, %2 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %5 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%1, %3 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%4 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %7 = d2m.wait %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %8 = d2m.wait %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %9 = d2m.reserve %cb2 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %10 = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
      %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %8 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) outs(%10 : tensor<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_0: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %13 = "d2m.tile_matmul"(%in, %in_0, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %13 : !ttcore.tile<32x32, f32>
      } -> tensor<1x1x!ttcore.tile<32x32, f32>>
      d2m.store %9, %11 : tensor<1x1x!ttcore.tile<32x32, f32>>
      %12 = d2m.wait %cb2 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %6 = d2m.to_layout %5, %arg2 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> into tensor<32x32xf32> -> tensor<32x32xf32>
    return %6 : tensor<32x32xf32>
  }
}

