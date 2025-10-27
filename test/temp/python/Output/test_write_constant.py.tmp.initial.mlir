#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
module {
  ttcore.global @inp = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> [0]
  func.func @test_write_constant(%arg0: tensor<32x32xf32> {d2m.stream = true}, %arg1: tensor<32x32xf32> {d2m.stream = false}) -> tensor<32x32xf32> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%1, %2) : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%3 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %6 = ttcore.get_global @inp : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
      %7 = d2m.reserve %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %tx = d2m.dma %6 [%c0, %c0_0], %7 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %6 = d2m.wait %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %7 = d2m.reserve %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      d2m.store %7, %6 : tensor<1x1x!ttcore.tile<32x32, f32>>
      %8 = d2m.wait %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %5 = d2m.to_layout %4, %arg1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> into tensor<32x32xf32> -> tensor<32x32xf32>
    return %5 : tensor<32x32xf32>
  }
}

