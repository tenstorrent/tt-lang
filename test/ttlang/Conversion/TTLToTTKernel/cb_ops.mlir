// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cb_reserve_single(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[CB_TTL:.*]] = builtin.unrealized_conversion_cast %[[C0]] : i32 to !ttl.cb<[1, 1], f32, 2>
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast %[[CB_TTL]] : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
module {
  func.func @cb_reserve_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_push_single(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[CB_TTL:.*]] = builtin.unrealized_conversion_cast %[[C0]] : i32 to !ttl.cb<[1, 1], f32, 2>
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast %[[CB_TTL]] : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_push_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
module {
  func.func @cb_push_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cb_wait_single(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[CB_TTL:.*]] = builtin.unrealized_conversion_cast %[[C0]] : i32 to !ttl.cb<[1, 1], f32, 2>
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast %[[CB_TTL]] : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_wait_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
module {
  func.func @cb_wait_single() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @cb_pop_single(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[CB_TTL:.*]] = builtin.unrealized_conversion_cast %[[C0]] : i32 to !ttl.cb<[1, 1], f32, 2>
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast %[[CB_TTL]] : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_pop_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
module {
  func.func @cb_pop_single() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Producer pattern: reserve space, write data, push to consumer.
// CHECK-LABEL: func.func @cb_producer_pattern(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast {{.*}} : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_producer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Consumer pattern: wait for data, read data, pop to free space.
// CHECK-LABEL: func.func @cb_consumer_pattern(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast {{.*}} : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_wait_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_pop_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_consumer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB with tile element type.
// CHECK-LABEL: func.func @cb_tile_element(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast {{.*}} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> to !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
module {
  func.func @cb_tile_element() -> tensor<1x1x!ttcore.tile<32x32, bf16>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = !ttcore.tile<32x32, bf16>, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// CB with 2D block shape: num_pages = 2*2 = 4.
// CHECK-LABEL: func.func @cb_2d_shape(
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[CB:.*]] = builtin.unrealized_conversion_cast {{.*}} : !ttl.cb<[2, 2], f32, 2> to !ttkernel.cb<8, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C4]]) : (!ttkernel.cb<8, f32>, i32) -> ()
module {
  func.func @cb_2d_shape() -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [2, 2], element_type = f32, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// Multiple CBs in same kernel: each gets its own conversion.
// CHECK-LABEL: func.func @cb_multiple(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.cb_reserve_back({{.*}}, %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back({{.*}}, %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_reserve_back({{.*}}, %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back({{.*}}, %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_multiple() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
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
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[CB_TTL:.*]] = builtin.unrealized_conversion_cast {{.*}} : i32 to !ttl.cb<[1, 1], f32, 2>
// CHECK: scf.for {{.*}} {
// CHECK:   %[[CB:.*]] = builtin.unrealized_conversion_cast %[[CB_TTL]] : !ttl.cb<[1, 1], f32, 2> to !ttkernel.cb<2, f32>
// CHECK:   ttkernel.cb_reserve_back(%[[CB]], %[[C1_I32]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK:   ttkernel.cb_push_back(%[[CB]], %[[C1_I32]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: }
module {
  func.func @cb_in_loop() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
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
