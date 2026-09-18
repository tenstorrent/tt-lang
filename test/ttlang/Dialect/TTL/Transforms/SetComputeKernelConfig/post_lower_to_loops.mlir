// Verify configuration analysis accepts tensor-of-tile operands produced by
// ttl-lower-to-loops when a pipeline re-runs configuration analysis.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-lower-to-loops),ttl-set-compute-kernel-config{matmul-full-fp32=0 reduce-full-fp32=0})' | FileCheck %s

#identity = affine_map<(tileRow, tileColumn) -> (tileRow, tileColumn)>

// CHECK-LABEL: func.func @matmul_tensor_operands_after_loop_lowering
// CHECK-SAME: fp32_dest_acc_en = false
// CHECK: ttl.tile_matmul_block %{{.*}}, %{{.*}} into dst[%{{.*}}]
// CHECK-SAME: tensor<1x1x!ttcore.tile<32x32, bf16>>,
// CHECK-SAME: tensor<1x1x!ttcore.tile<32x32, bf16>>
func.func @matmul_tensor_operands_after_loop_lowering(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhsDfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhsDfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %outputDfb = ttl.bind_cb {cb_index = 16, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhsAttached = ttl.attach_cb %lhs, %lhsDfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhsAttached = ttl.attach_cb %rhs, %rhsDfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %outputAttached = ttl.attach_cb %output, %outputDfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %outputReserve = ttl.cb_reserve %outputDfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%lhsAttached, %rhsAttached
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%outputAttached : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#identity, #identity, #identity],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhsTile: !ttcore.tile<32x32, bf16>,
       %rhsTile: !ttcore.tile<32x32, bf16>,
       %outputTile: !ttcore.tile<32x32, bf16>):
    %zero = arith.constant 0 : index
    %product = ttl.tile_matmul_block %lhsTile, %rhsTile into dst[%zero]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    %tileRow = ttl.iter_index 0 : index
    %tileColumn = ttl.iter_index 1 : index
    ttl.tile_store %product, %outputReserve[%tileRow, %tileColumn]
        from dst[%zero]
        : !ttcore.tile<32x32, bf16>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
