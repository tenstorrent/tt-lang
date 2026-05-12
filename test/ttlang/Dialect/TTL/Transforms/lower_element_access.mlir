// Tests for ttl-lower-element-access-to-emitc pass. Verifies that
// element_read/element_write ops are lowered to EmitC calls with the correct
// helper functions, pointer resolution, and face layout formula.

// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(ttl-lower-element-access-to-emitc)' \
// RUN:   | FileCheck %s

// Test: bf16 element_read from a cb_wait block produces get_read_ptr + _ttl_elem_read_bf16.

// CHECK-LABEL: func.func @read_bf16_from_wait
// CHECK: emitc.verbatim "auto _ttl_elem_read_bf16{{.*}}ASSERT(row < 32 && col < 32)
// CHECK: emitc.literal "get_compile_time_arg_val(0)"
// CHECK: emitc.call_opaque "get_read_ptr"
// CHECK: emitc.call_opaque "_ttl_elem_read_bf16"
// CHECK-NOT: ttl.element_read
func.func @read_bf16_from_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %val = ttl.element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test: bf16 element_write to a cb_reserve block produces get_write_ptr + _ttl_elem_write_bf16.

// CHECK-LABEL: func.func @write_bf16_to_reserve
// CHECK: emitc.literal "get_compile_time_arg_val(1)"
// CHECK: emitc.call_opaque "get_write_ptr"
// CHECK: emitc.call_opaque "_ttl_elem_write_bf16"
// CHECK-NOT: ttl.element_write
func.func @write_bf16_to_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  ttl.element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test: f32 element_read emits _ttl_elem_read_f32 with uint32_t pointer type.

// CHECK-LABEL: func.func @read_f32_from_wait
// CHECK: emitc.verbatim "auto _ttl_elem_read_f32{{.*}}uint32_t*{{.*}}"
// CHECK: emitc.call_opaque "_ttl_elem_read_f32"
// CHECK-NOT: ttl.element_read
func.func @read_f32_from_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %block = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %val = ttl.element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Test: read + write in same function emits both helper lambdas.

// CHECK-LABEL: func.func @read_write_same_function
// CHECK: emitc.verbatim "auto _ttl_elem_read_bf16{{.*}}"
// CHECK: emitc.verbatim "auto _ttl_elem_write_bf16{{.*}}"
// CHECK: emitc.call_opaque "_ttl_elem_read_bf16"
// CHECK: emitc.call_opaque "_ttl_elem_write_bf16"
// CHECK-NOT: ttl.element_read
// CHECK-NOT: ttl.element_write
func.func @read_write_same_function()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rblk = ttl.attach_cb %wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %wblk = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %val = ttl.element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> i32
  ttl.element_write %wblk[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test: function without element ops is not modified.

// CHECK-LABEL: func.func @no_element_ops
// CHECK-NOT: emitc.verbatim
// CHECK-NOT: _ttl_elem
func.func @no_element_ops()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
