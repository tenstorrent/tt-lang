// Tests for unsafe_element_read/unsafe_element_write lowering through
// convert-ttl-to-ttkernel. Verifies that element access ops are lowered to
// structured arith + TTKernel ops (get_read_ptr, reinterpret_cast,
// load_from_l1, store_to_l1) with the face-based tile layout offset formula.

// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(convert-ttl-to-ttkernel)' \
// RUN:   | FileCheck %s

// Test: f32 unsafe_element_read from a cb_wait block produces get_read_ptr +
// reinterpret_cast + load_from_l1.

// CHECK-LABEL: func.func @read_f32_from_wait
// CHECK: ttkernel.get_read_ptr
// CHECK: ttkernel.reinterpret_cast
// CHECK: ttkernel.load_from_l1
// CHECK-NOT: ttl.unsafe_element_read
func.func @read_f32_from_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %block = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %val = ttl.unsafe_element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Test: f32 unsafe_element_write to a cb_reserve block produces get_write_ptr +
// reinterpret_cast + store_to_l1.

// CHECK-LABEL: func.func @write_f32_to_reserve
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.reinterpret_cast
// CHECK: ttkernel.store_to_l1
// CHECK-NOT: ttl.unsafe_element_write
func.func @write_f32_to_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %block = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  ttl.unsafe_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, i32
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Test: read + write in same function both lower to arith + TTKernel ops.

// CHECK-LABEL: func.func @read_write_same_function
// CHECK: ttkernel.get_read_ptr
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.store_to_l1
// CHECK-NOT: ttl.unsafe_element_read
// CHECK-NOT: ttl.unsafe_element_write
func.func @read_write_same_function()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rblk = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %wblk = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %val = ttl.unsafe_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  ttl.unsafe_element_write %wblk[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Test: function without element ops is not modified.

// CHECK-LABEL: func.func @no_element_ops
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: ttkernel.store_to_l1
func.func @no_element_ops()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Test: f32 lowering with dynamic row/col produces arith face layout ops.
// Uses function arguments to prevent constant folding.

// CHECK-LABEL: func.func @read_f32_dynamic_indices
// CHECK: ttkernel.get_read_ptr
// CHECK: ttkernel.reinterpret_cast
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.addi
// CHECK: arith.muli
// CHECK: arith.remui
// CHECK: arith.muli
// CHECK: arith.remui
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK: ttkernel.load_from_l1
func.func @read_f32_dynamic_indices(%row: index, %col: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %block = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %val = ttl.unsafe_element_read %block[%row, %col] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}
