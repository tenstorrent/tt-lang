// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// Summary: ttl.kernel operation parsing and round-trip tests.

// CHECK-LABEL: func.func @kernel_single_dm_region
// CHECK: ttl.kernel
// CHECK-SAME: grid = #ttcore.grid<1x1>
// CHECK-SAME: threads = [#ttl.thread<datamovement>]
// CHECK: ins()
// CHECK-SAME: outs({{.*}} : tensor<32x32xf32>)
module {
  func.func @kernel_single_dm_region(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// CHECK-LABEL: func.func @kernel_single_compute_region
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<compute>]
module {
  func.func @kernel_single_compute_region(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Two thread regions: datamovement + compute.
// CHECK-LABEL: func.func @kernel_dm_and_compute
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]
// CHECK-SAME: ins({{.*}} : tensor<32x32xf32>) outs({{.*}} : tensor<32x32xf32>)
// CHECK: ^bb0({{.*}}: !ttl.cb<[1, 1], f32, 2>, {{.*}}: !ttl.cb<[1, 1], f32, 2>):
// CHECK: }, {
// CHECK: ^bb0({{.*}}: !ttl.cb<[1, 1], f32, 2>, {{.*}}: !ttl.cb<[1, 1], f32, 2>):
module {
  func.func @kernel_dm_and_compute(%in: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins(%in : tensor<32x32xf32>)
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%in_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%in_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Three thread regions: dm_read + dm_write + compute (common pattern).
// CHECK-LABEL: func.func @kernel_three_threads
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<datamovement>, #ttl.thread<datamovement>, #ttl.thread<compute>]
module {
  func.func @kernel_three_threads(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%lhs_cb: !ttl.cb<[1, 1], f32, 2>, %rhs_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%lhs_cb: !ttl.cb<[1, 1], f32, 2>, %rhs_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%lhs_cb: !ttl.cb<[1, 1], f32, 2>, %rhs_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Kernel with tile element type.
// CHECK-LABEL: func.func @kernel_tile_element
// CHECK: ttl.kernel
// CHECK: ^bb0({{.*}}: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
module {
  func.func @kernel_tile_element(%out: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins()
        outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>):
    } : tensor<1x1x!ttcore.tile<32x32, bf16>>
    return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// Kernel with 2x2 grid.
// CHECK-LABEL: func.func @kernel_2x2_grid
// CHECK: ttl.kernel
// CHECK-SAME: grid = #ttcore.grid<2x2>
module {
  func.func @kernel_2x2_grid(%out: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<2x2>, threads = [#ttl.thread<datamovement>]}
        ins()
        outs(%out : tensor<64x64xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}

// -----

// Multiple inputs and outputs.
// CHECK-LABEL: func.func @kernel_multi_io
// CHECK: ttl.kernel
// CHECK-SAME: ins({{.*}}, {{.*}} : tensor<32x32xf32>, tensor<32x32xf32>) outs({{.*}}, {{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
module {
  func.func @kernel_multi_io(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>,
                             %c: tensor<32x32xf32>, %d: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %r0, %r1 = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%c, %d : tensor<32x32xf32>, tensor<32x32xf32>) {
    ^bb0(%a_cb: !ttl.cb<[1, 1], f32, 2>, %b_cb: !ttl.cb<[1, 1], f32, 2>,
         %c_cb: !ttl.cb<[1, 1], f32, 2>, %d_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>, tensor<32x32xf32>
    return %r0, %r1 : tensor<32x32xf32>, tensor<32x32xf32>
  }
}
