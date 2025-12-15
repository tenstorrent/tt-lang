// RUN: ttlang-opt --verify-diagnostics --split-input-file %s
// Summary: Invalid ttl.kernel cases rejected by verifiers.

// -----

// Thread count mismatch: 2 thread attrs but 1 region.
module {
  func.func @threads_count_mismatch(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @below {{threads attribute count (2) must match number of regions (1)}}
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Thread count mismatch: 1 thread attr but 2 regions.
module {
  func.func @threads_count_mismatch_fewer(%out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @below {{threads attribute count (1) must match number of regions (2)}}
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>]}
        ins()
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Block argument count mismatch between regions.
module {
  func.func @block_arg_count_mismatch(%in: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @below {{thread region 1 has 1 block arguments, expected 2 (must match first region)}}
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins(%in : tensor<32x32xf32>)
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%in_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Block argument type mismatch between regions.
module {
  func.func @block_arg_type_mismatch(%in: tensor<32x32xf32>, %out: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @below {{thread region 1 block argument 0 has type '!ttl.cb<[2, 2], f32, 2>', expected '!ttl.cb<[1, 1], f32, 2>' (must match first region)}}
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins(%in : tensor<32x32xf32>)
        outs(%out : tensor<32x32xf32>) {
    ^bb0(%in_cb: !ttl.cb<[1, 1], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    }, {
    ^bb0(%in_cb: !ttl.cb<[2, 2], f32, 2>, %out_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

// Results count does not match outputs count.
module {
  func.func @results_outputs_mismatch(%out1: tensor<32x32xf32>, %out2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // expected-error @below {{number of results (1) must match number of outputs (2)}}
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<compute>]}
        ins()
        outs(%out1, %out2 : tensor<32x32xf32>, tensor<32x32xf32>) {
    ^bb0(%out1_cb: !ttl.cb<[1, 1], f32, 2>, %out2_cb: !ttl.cb<[1, 1], f32, 2>):
    } : tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}
