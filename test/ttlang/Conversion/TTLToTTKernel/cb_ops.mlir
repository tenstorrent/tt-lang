// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cb_reserve_single(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
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
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_push_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
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
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_wait_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
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
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_pop_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
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

// Producer pattern: reserve space, write data, push to consumer.
// CHECK-LABEL: func.func @cb_producer_pattern(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_producer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Consumer pattern: wait for data, read data, pop to free space.
// CHECK-LABEL: func.func @cb_consumer_pattern(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_wait_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_pop_front(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_consumer_pattern() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    ttl.cb_pop %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// CB with tile element type.
// CHECK-LABEL: func.func @cb_tile_element(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
module {
  func.func @cb_tile_element() -> tensor<1x1x!ttcore.tile<32x32, bf16>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

// CB with 2D block shape: num_pages = 2*2 = 4.
// CHECK-LABEL: func.func @cb_2d_shape(
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C4]]) : (!ttkernel.cb<8, f32>, i32) -> ()
module {
  func.func @cb_2d_shape() -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// Multiple CBs in same kernel: each gets its own index in get_compile_time_arg_val.
// CHECK-LABEL: func.func @cb_multiple(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB0]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back(%[[CB0]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_reserve_back(%[[CB1]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: ttkernel.cb_push_back(%[[CB1]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
module {
  func.func @cb_multiple() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
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
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.cb_reserve_back(%[[CB]], %[[C1_I32]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK:   ttkernel.cb_push_back(%[[CB]], %[[C1_I32]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: }
module {
  func.func @cb_in_loop() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
      ttl.cb_push %cb : <[1, 1], f32, 2>
    }
    func.return
  }
}

// -----

// Non-zero CB index: verifies cb_index attribute is used correctly.
// CHECK-LABEL: func.func @cb_nonzero_index(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(7) : () -> !ttkernel.cb<2, f32>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, f32>, i32) -> ()
// CHECK: return
module {
  func.func @cb_nonzero_index() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return %view : tensor<1x1xf32>
  }
}

// -----

// CB index at upper boundary (31 is max valid index).
// CHECK-LABEL: func.func @cb_max_index(
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(31) : () -> !ttkernel.cb<2, f32>
module {
  func.func @cb_max_index() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 31, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ttl.cb_push %cb : <[1, 1], f32, 2>
    func.return
  }
}

// -----

// Store tile into CB: lowers to pack_tile.
// CHECK-LABEL: func.func @store_single(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
// CHECK: ttkernel.pack_tile(%[[C0]], %[[CB]], %[[C0]], {{.*}}) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
module {
  func.func @store_single(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Store followed by push: typical producer pattern.
// CHECK-LABEL: func.func @store_and_push(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
// CHECK: ttkernel.pack_tile(%[[C0]], %[[CB]], %[[C0]], {{.*}}) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
// CHECK: ttkernel.cb_push_back(%[[CB]], %[[C1]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
module {
  func.func @store_and_push(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
