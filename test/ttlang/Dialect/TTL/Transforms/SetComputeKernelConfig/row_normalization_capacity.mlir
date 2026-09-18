// Verifies that fixed-block destination residency participates in kernel
// configuration selection.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-set-compute-kernel-config{fp32-dest-acc-en=enabled dst-full-sync-en=auto reduce-full-fp32=0 matmul-full-fp32=0 enable-fpu-binary-ops=0})' | FileCheck %s

#identity = affine_map<(row, column) -> (row, column)>

// Five 32-bit destination tiles require full synchronization.
// CHECK-LABEL: func.func @five_tile_row
// CHECK-SAME:  dst_full_sync_en = true
// CHECK-SAME:  fp32_dest_acc_en = true
// CHECK:       ttl.tile_row_normalization_block
// CHECK-SAME:  num_tiles = 5
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @five_tile_row(
      %input: tensor<1x5x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x5x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x5x!ttcore.tile<32x32, bf16>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c-1 = arith.constant -1 : index
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>
    %input_view = ttl.attach_cb %input, %input_dfb
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %output_view = ttl.attach_cb %output, %output_dfb
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %result = ttl.compute
        ins(%input_view : tensor<1x5x!ttcore.tile<32x32, bf16>>)
        outs(%output_view : tensor<1x5x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#identity, #identity],
         iterator_types = ["parallel", "parallel"]} {
      ^bb0(%input_tile: !ttcore.tile<32x32, bf16>,
           %output_tile: !ttcore.tile<32x32, bf16>):
        %row = ttl.iter_index 0 : index
        %column = ttl.iter_index 1 : index
        %normalized = ttl.tile_row_normalization_block
            %input_tile, %input_tile, %output_tile
            scale = 1.953125e-04 epsilon = 1.000000e-05
            has_gamma = false num_tiles = 5 into dst[%c-1]
            {ttl.dst_placeholder}
            : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
              !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
        ttl.tile_store %normalized, %output_view[%row, %column] from dst[%c-1]
            {ttl.dst_placeholder}
            : !ttcore.tile<32x32, bf16>,
              tensor<1x5x!ttcore.tile<32x32, bf16>>
        ttl.yield
    } -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    return %result : tensor<1x5x!ttcore.tile<32x32, bf16>>
  }
}
