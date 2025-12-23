// RUN: ttlang-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cb_reserve_single(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], f32, 2>
// CHECK: %[[VIEW:.*]] = ttl.cb_reserve %[[CB]] : <[1, 1], f32, 2> -> tensor<1x1xf32>
// CHECK: return %[[VIEW]]
module {
  func.func @cb_reserve_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_push_single(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], f32, 2>
// CHECK: ttl.cb_push %[[CB]] : <[1, 1], f32, 2>
// CHECK: return
module {
  func.func @cb_push_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cb_wait_single(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], f32, 2>
// CHECK: %[[VIEW:.*]] = ttl.cb_wait %[[CB]] : <[1, 1], f32, 2> -> tensor<1x1xf32>
// CHECK: return %[[VIEW]]
module {
  func.func @cb_wait_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_pop_single(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], f32, 2>
// CHECK: ttl.cb_pop %[[CB]] : <[1, 1], f32, 2>
// CHECK: return
module {
  func.func @cb_pop_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB with tile element type.
// CHECK-LABEL: func.func @cb_tile_element(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
// CHECK: ttl.cb_reserve %[[CB]] : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
module {
  func.func @cb_tile_element() -> tensor<1x1x!ttcore.tile<32x32, bf16>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// CB with 2D block shape.
// CHECK-LABEL: func.func @cb_2d_shape(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[2, 2], f32, 2>
// CHECK: ttl.cb_reserve %[[CB]] : <[2, 2], f32, 2> -> tensor<2x2xf32>
module {
  func.func @cb_2d_shape() -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// Store a tile into a CB view.
// CHECK-LABEL: func.func @store_single(
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = {{.*}}, buffer_factor = {{.*}}} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
// CHECK: %[[VIEW:.*]] = ttl.cb_reserve %[[CB]] : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: ttl.store %{{.*}}, %[[VIEW]] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
module {
  func.func @store_single(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
