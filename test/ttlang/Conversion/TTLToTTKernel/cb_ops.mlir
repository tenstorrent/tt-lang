// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cb_reserve_single(
// CHECK: ttkernel.cb_reserve_back
// CHECK: return
module {
  func.func @cb_reserve_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_push_single(
// CHECK: ttkernel.cb_push_back
// CHECK: return
module {
  func.func @cb_push_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cb_wait_single(
// CHECK: ttkernel.cb_wait_front
// CHECK: return
module {
  func.func @cb_wait_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_pop_single(
// CHECK: ttkernel.cb_pop_front
// CHECK: return
module {
  func.func @cb_pop_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Producer pattern: reserve space, write data, push to consumer.
// CHECK-LABEL: func.func @cb_producer_pattern(
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.cb_push_back
module {
  func.func @cb_producer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Consumer pattern: wait for data, read data, pop to free space.
// CHECK-LABEL: func.func @cb_consumer_pattern(
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.cb_pop_front
module {
  func.func @cb_consumer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB with tile element type.
// CHECK-LABEL: func.func @cb_tile_element(
// CHECK: ttkernel.cb_reserve_back
module {
  func.func @cb_tile_element() -> tensor<1x1x!ttcore.tile<32x32, bf16>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = !ttcore.tile<32x32, bf16>, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// CB with 2D block shape.
// CHECK-LABEL: func.func @cb_2d_shape(
// CHECK: ttkernel.cb_reserve_back
module {
  func.func @cb_2d_shape() -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [2, 2], element_type = f32, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// Multiple CBs in same kernel.
// CHECK-LABEL: func.func @cb_multiple(
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.cb_push_back
module {
  func.func @cb_multiple() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view0 = ttl.cb_reserve %cb0 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb0 : <[1, 1], f32, 2>
    %view1 = ttl.cb_reserve %cb1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB operations in a loop.
// CHECK-LABEL: func.func @cb_in_loop(
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.cb_push_back
module {
  func.func @cb_in_loop() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
      ttl.cb_push %cb : <[1, 1], f32, 2>
    }
    func.return
  }
}
