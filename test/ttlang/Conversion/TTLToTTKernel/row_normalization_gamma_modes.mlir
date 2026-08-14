// Verifies TTKernel lowering when gamma multiplication is disabled.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,func.func(ttkernel-combine-pack-tiles),canonicalize,cse,lower-affine)' | FileCheck %s

#identity = affine_map<(row, column) -> (row, column)>

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {

// Gamma-disabled lowering reuses the input DFB operand without reading gamma.
// CHECK-LABEL: func.func @row_normalization_no_gamma
// CHECK:       ttkernel.init_sfpu(%[[NO_GAMMA_INPUT:[a-zA-Z0-9_]+]], %[[NO_GAMMA_OUTPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT:  ttkernel.tile_regs_acquire
// CHECK-NEXT:  ttkernel.experimental_row_normalization_block(%[[NO_GAMMA_INPUT]], %[[NO_GAMMA_INPUT]], %[[NO_GAMMA_OUTPUT]]) num_tiles = 3
// CHECK-SAME:  has_gamma = false dtype = <bf16>
// CHECK-NEXT:  ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.tile_regs_wait
// CHECK-NEXT:  ttkernel.pack_tile_block(%{{.*}}, %[[NO_GAMMA_OUTPUT]], %{{.*}})
// CHECK-NEXT:  ttkernel.tile_regs_release
func.func @row_normalization_no_gamma()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c-1 = arith.constant -1 : index
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x3x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %reserved = ttl.cb_reserve %output_dfb
      : <[1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x3x!ttcore.tile<32x32, bf16>>
  %output = ttl.attach_cb %empty, %output_dfb
      : (tensor<1x3x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%input : tensor<1x3x!ttcore.tile<32x32, bf16>>)
      outs(%output : tensor<1x3x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#identity, #identity],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<32x32, bf16>,
       %output_tile: !ttcore.tile<32x32, bf16>):
    %row = ttl.iter_index 0 : index
    %column = ttl.iter_index 1 : index
    %normalized = ttl.tile_row_normalization_block
        %input_tile, %input_tile, %output_tile
        scale = 3.255208e-04 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 3 into dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %normalized, %reserved[%row, %column] from dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  return
}
}
