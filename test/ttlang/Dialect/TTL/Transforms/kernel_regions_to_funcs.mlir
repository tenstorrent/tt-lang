// RUN: ttlang-opt --ttl-kernel-regions-to-funcs --split-input-file %s | FileCheck %s
// Summary: Test ttl-kernel-regions-to-funcs pass extracts regions to functions.

// Single datamovement thread extraction.
// CHECK-LABEL: func.func private @single_dm_dm_0
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK: func.func @single_dm(
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<datamovement, @single_dm_dm_0>]
module {
  func.func @single_dm(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Single compute thread extraction.
// CHECK-LABEL: func.func private @single_compute_compute_0
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
// CHECK: func.func @single_compute(
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<compute, @single_compute_compute_0>]
module {
  func.func @single_compute(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Two threads: datamovement + compute.
// CHECK-LABEL: func.func private @two_threads_dm_0
// CHECK-SAME: ({{.*}}: !ttl.cb<[1, 1], f32, 2>, {{.*}}: !ttl.cb<[1, 1], f32, 2>)
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK-LABEL: func.func private @two_threads_compute_1
// CHECK-SAME: ({{.*}}: !ttl.cb<[1, 1], f32, 2>, {{.*}}: !ttl.cb<[1, 1], f32, 2>)
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
// CHECK: func.func @two_threads(
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<datamovement, @two_threads_dm_0>, #ttl.thread<compute, @two_threads_compute_1>]
module {
  func.func @two_threads(%in: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
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

// Three threads: dm_read + dm_write + compute (typical add kernel pattern).
// CHECK-LABEL: func.func private @three_threads_dm_0
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK-LABEL: func.func private @three_threads_dm_1
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK-LABEL: func.func private @three_threads_compute_2
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
// CHECK: func.func @three_threads(
// CHECK: ttl.kernel
// CHECK-SAME: threads = [#ttl.thread<datamovement, @three_threads_dm_0>, #ttl.thread<datamovement, @three_threads_dm_1>, #ttl.thread<compute, @three_threads_compute_2>]
module {
  func.func @three_threads(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
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

// Region with CB operations preserved in extracted function.
// CHECK-LABEL: func.func private @with_cb_ops_dm_0
// CHECK-SAME: (%[[CB:.*]]: !ttl.cb<[1, 1], f32, 2>)
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: ttl.cb_reserve %[[CB]], %[[C1]]
// CHECK: ttl.cb_push %[[CB]], %[[C1]]
// CHECK: return
module {
  func.func @with_cb_ops(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
      %c1 = arith.constant 1 : i32
      %view = ttl.cb_reserve %out_cb, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
      ttl.cb_push %out_cb, %c1 : <[1, 1], f32, 2>
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Kernel with tile element type CB arguments.
// CHECK-LABEL: func.func private @tile_cb_compute_0
// CHECK-SAME: ({{.*}}: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
module {
  func.func @tile_cb(%out: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins()
        outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>):
    } : tensor<1x1x!ttcore.tile<32x32, bf16>>
    return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}
