// Tests that ordinary compute lowering does not require target-specific
// schedule capabilities.

// RUN: ttlang-opt %s -ttl-lower-to-loops | FileCheck %s

#identity = affine_map<(dim0, dim1) -> (dim0, dim1)>

module attributes {ttl.target_arch = #ttcore.arch<quasar>} {
  // An unsupported schedule target does not affect ordinary tile addition.
  // CHECK-LABEL: func.func @ordinary_compute
  // CHECK: scf.for
  // CHECK: ttl.tile_add
  // CHECK-NOT: ttl.compute
  func.func @ordinary_compute(
      %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %lhs_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %rhs_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_cb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lhs_attached = ttl.attach_cb %lhs, %lhs_cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_attached = ttl.attach_cb %rhs, %rhs_cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_attached = ttl.attach_cb %output, %output_cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_view = ttl.cb_reserve %output_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.compute
        ins(%lhs_attached, %rhs_attached
            : tensor<1x1x!ttcore.tile<32x32, bf16>>,
              tensor<1x1x!ttcore.tile<32x32, bf16>>)
        outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#identity, #identity, #identity],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
         %rhs_tile: !ttcore.tile<32x32, bf16>,
         %output_tile: !ttcore.tile<32x32, bf16>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      %zero = arith.constant 0 : index
      %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
          : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
            -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %sum, %output_view[%row, %column] from dst[%zero]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}
