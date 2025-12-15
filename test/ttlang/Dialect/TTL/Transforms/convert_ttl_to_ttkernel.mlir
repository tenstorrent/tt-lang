// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s
// Summary: Test convert-ttl-to-ttkernel pass patterns.

// -----

// FuncCBArgsToGetArgVal: Single CB argument converted to get_compile_time_arg_val.
// Total elements = shape * buffer_factor = 1*1*2 = 2
// CHECK-LABEL: func.func @single_cb_arg()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK: ttkernel.get_compile_time_arg_val(0) {{.*}} -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: return
module {
  func.func @single_cb_arg(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// FuncCBArgsToGetArgVal: Multiple CB arguments get sequential indices.
// CHECK-LABEL: func.func @multi_cb_args()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<noc>}
// CHECK-DAG: ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: ttkernel.get_compile_time_arg_val(2)
// CHECK: return
module {
  func.func @multi_cb_args(
      %lhs: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %rhs: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %out: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// FuncCBArgsToGetArgVal: CB with f32 element type.
// Total = 1*1*2 = 2
// CHECK-LABEL: func.func @cb_f32_element()
// CHECK: ttkernel.get_compile_time_arg_val(0) {{.*}} -> !ttkernel.cb<2, f32>
module {
  func.func @cb_f32_element(%cb: !ttl.cb<[1, 1], f32, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// FuncCBArgsToGetArgVal: CB with 2D block shape.
// Total = 2*2*2 = 8
// CHECK-LABEL: func.func @cb_2d_shape()
// CHECK: ttkernel.get_compile_time_arg_val(0) {{.*}} -> !ttkernel.cb<8, f32>
module {
  func.func @cb_2d_shape(%cb: !ttl.cb<[2, 2], f32, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// FuncCBArgsToGetArgVal: Compute thread with CB args.
// CHECK-LABEL: func.func @compute_thread_cb_args()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
// CHECK: ttkernel.get_compile_time_arg_val(0) {{.*}} -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
module {
  func.func @compute_thread_cb_args(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

// FuncCBArgsToGetArgVal: Function without ttkernel.thread attr is unchanged.
// CHECK-LABEL: func.func @no_thread_attr
// CHECK-SAME: (%{{.*}}: !ttl.cb<[1, 1], f32, 2>)
module {
  func.func @no_thread_attr(%cb: !ttl.cb<[1, 1], f32, 2>) {
    return
  }
}

// -----

// CB sync ops with CB from get_compile_time_arg_val chain.
// CHECK-LABEL: func.func @cb_sync_with_args()
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.cb_reserve_back(%[[CB]]
// CHECK: ttkernel.cb_push_back(%[[CB]]
module {
  func.func @cb_sync_with_args(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// CB wait and pop with CB from get_compile_time_arg_val chain.
// CHECK-LABEL: func.func @cb_wait_pop_with_args()
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.cb_wait_front(%[[CB]]
// CHECK: ttkernel.cb_pop_front(%[[CB]]
module {
  func.func @cb_wait_pop_with_args(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_wait %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// CreateCBLowering: CB creation lowered.
// CHECK-LABEL: func.func @create_cb_with_index()
// CHECK-NOT: ttl.create_cb
module {
  func.func @create_cb_with_index() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2, buffer_index = 0 : i32} : !ttl.cb<[1, 1], f32, 2>
    return
  }
}
