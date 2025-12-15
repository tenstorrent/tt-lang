// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=LOWERED
// Summary: CB synchronization operation tests (cb_reserve, cb_push, cb_wait, cb_pop).

// CHECK-LABEL: func.func @cb_reserve_single(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[VIEW:.*]] = ttl.cb_reserve %[[CB]], %[[C1]] : <[1, 1], f32, 2> -> tensor<1x1xf32>
// CHECK: return %[[VIEW]]

// LOWERED-LABEL: func.func @cb_reserve_single(
// LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : i32
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: return
module {
  func.func @cb_reserve_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_push_single(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: ttl.cb_push %[[CB]], %[[C1]] : <[1, 1], f32, 2>
// CHECK: return

// LOWERED-LABEL: func.func @cb_push_single(
// LOWERED: ttkernel.cb_push_back
// LOWERED: return
module {
  func.func @cb_push_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    ttl.cb_push %cb, %c1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cb_wait_single(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[VIEW:.*]] = ttl.cb_wait %[[CB]], %[[C1]] : <[1, 1], f32, 2> -> tensor<1x1xf32>
// CHECK: return %[[VIEW]]

// LOWERED-LABEL: func.func @cb_wait_single(
// LOWERED: ttkernel.cb_wait_front
// LOWERED: return
module {
  func.func @cb_wait_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_wait %cb, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_pop_single(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: ttl.cb_pop %[[CB]], %[[C1]] : <[1, 1], f32, 2>
// CHECK: return

// LOWERED-LABEL: func.func @cb_pop_single(
// LOWERED: ttkernel.cb_pop_front
// LOWERED: return
module {
  func.func @cb_pop_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    ttl.cb_pop %cb, %c1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Producer pattern: reserve space, write data, push to consumer.
// CHECK-LABEL: func.func @cb_producer_pattern(
// CHECK: ttl.cb_reserve
// CHECK: ttl.cb_push

// LOWERED-LABEL: func.func @cb_producer_pattern(
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: ttkernel.cb_push_back
module {
  func.func @cb_producer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb, %c1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Consumer pattern: wait for data, read data, pop to free space.
// CHECK-LABEL: func.func @cb_consumer_pattern(
// CHECK: ttl.cb_wait
// CHECK: ttl.cb_pop

// LOWERED-LABEL: func.func @cb_consumer_pattern(
// LOWERED: ttkernel.cb_wait_front
// LOWERED: ttkernel.cb_pop_front
module {
  func.func @cb_consumer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_wait %cb, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_pop %cb, %c1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Multiple pages pattern: reserve and push multiple pages at once.
// CHECK-LABEL: func.func @cb_multi_page(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 4>
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: ttl.cb_reserve %[[CB]], %[[C2]] : <[1, 1], f32, 4>
// CHECK: ttl.cb_push %[[CB]], %[[C2]] : <[1, 1], f32, 4>

// LOWERED-LABEL: func.func @cb_multi_page(
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: ttkernel.cb_push_back
module {
  func.func @cb_multi_page() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 4} : !ttl.cb<[1, 1], f32, 4>
    %c2 = arith.constant 2 : i32
    %view = ttl.cb_reserve %cb, %c2 : <[1, 1], f32, 4> -> tensor<1x1xf32>
    ttl.cb_push %cb, %c2 : <[1, 1], f32, 4>
    func.return
  }
}

// -----

// CB with tile element type (common hardware pattern).
// CHECK-LABEL: func.func @cb_tile_element(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
// CHECK: ttl.cb_reserve %[[CB]], {{.*}} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

// LOWERED-LABEL: func.func @cb_tile_element(
// LOWERED: ttkernel.cb_reserve_back
module {
  func.func @cb_tile_element() -> tensor<1x1x!ttcore.tile<32x32, bf16>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = !ttcore.tile<32x32, bf16>, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// CB with 2D block shape.
// CHECK-LABEL: func.func @cb_2d_shape(
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[2, 2], f32, 2>
// CHECK: ttl.cb_reserve %[[CB]], {{.*}} : <[2, 2], f32, 2> -> tensor<2x2xf32>

// LOWERED-LABEL: func.func @cb_2d_shape(
// LOWERED: ttkernel.cb_reserve_back
module {
  func.func @cb_2d_shape() -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [2, 2], element_type = f32, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// Multiple CBs in same kernel (common data movement pattern).
// CHECK-LABEL: func.func @cb_multiple(
// CHECK: %[[CB0:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[CB1:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: ttl.cb_reserve %[[CB0]]
// CHECK: ttl.cb_push %[[CB0]]
// CHECK: ttl.cb_reserve %[[CB1]]
// CHECK: ttl.cb_push %[[CB1]]

// LOWERED-LABEL: func.func @cb_multiple(
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: ttkernel.cb_push_back
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: ttkernel.cb_push_back
module {
  func.func @cb_multiple() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : i32
    %view0 = ttl.cb_reserve %cb0, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb0, %c1 : <[1, 1], f32, 2>
    %view1 = ttl.cb_reserve %cb1, %c1 : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb1, %c1 : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB operations in a loop (common streaming pattern).
// CHECK-LABEL: func.func @cb_in_loop(
// CHECK: scf.for
// CHECK: ttl.cb_reserve
// CHECK: ttl.cb_push

// LOWERED-LABEL: func.func @cb_in_loop(
// LOWERED: scf.for
// LOWERED: ttkernel.cb_reserve_back
// LOWERED: ttkernel.cb_push_back
module {
  func.func @cb_in_loop() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1 {
      %view = ttl.cb_reserve %cb, %c1_i32 : <[1, 1], f32, 2> -> tensor<1x1xf32>
      ttl.cb_push %cb, %c1_i32 : <[1, 1], f32, 2>
    }
    func.return
  }
}
