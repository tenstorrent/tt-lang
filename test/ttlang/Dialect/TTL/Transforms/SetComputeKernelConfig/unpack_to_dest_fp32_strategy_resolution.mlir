// Verify that execution strategy selection and f32 unpack configuration are
// resolved together across the complete compute kernel.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file | FileCheck %s

// A binary operation with FPU/SFPU alternatives selects SFPU when a fixed
// SFPU consumer requires unpack-to-DST mode for the same dataflow buffer.
// CHECK-LABEL: func.func @shared_dfb_selects_compatible_strategy
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_exp
// CHECK-NEXT: ttl.tile_add
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @shared_dfb_selects_compatible_strategy(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// A flexible operation visited before the fixed SFPU consumer still selects
// the strategy compatible with the complete kernel requirements.
// CHECK-LABEL: func.func @reverse_order_sub_selects_compatible_strategy
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_sub
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
// CHECK-NEXT: ttl.tile_exp
func.func @reverse_order_sub_selects_compatible_strategy(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %difference = ttl.tile_sub %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Multiply selects SFPU when a fixed SFPU consumer requires unpack-to-DST mode
// for the same dataflow buffer.
// CHECK-LABEL: func.func @shared_dfb_mul_selects_compatible_strategy
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_exp
// CHECK-NEXT: ttl.tile_mul
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @shared_dfb_mul_selects_compatible_strategy(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %product = ttl.tile_mul %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// A fixed requirement constrains strategy-dependent binary operations both
// before and after it in operation order.
// CHECK-LABEL: func.func @shared_dfb_selection_is_order_independent
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_sub
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
// CHECK-NEXT: ttl.tile_exp
// CHECK-NEXT: ttl.tile_mul
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @shared_dfb_selection_is_order_independent(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %difference = ttl.tile_sub %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %product = ttl.tile_mul %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Requirements in nested regions constrain strategy decisions elsewhere in
// the same compute kernel.
// CHECK-LABEL: func.func @strategy_resolution_spans_regions
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: scf.if
// CHECK: ttl.tile_exp
// CHECK: ttl.tile_add
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @strategy_resolution_spans_regions(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>, %condition: i1)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  scf.if %condition {
    %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    scf.yield
  }
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Binary operations with execution alternatives are selected as one compatible
// assignment rather than independently by operation order.
// CHECK-LABEL: func.func @dependent_strategy_decisions_are_resolved_together
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_add
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
// CHECK-NEXT: ttl.tile_mul
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @dependent_strategy_decisions_are_resolved_together(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  %product = ttl.tile_mul %sum, %rhs_tile into dst[%zero]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Physical subtile dimensions do not change the shared configuration or
// strategy constraints.
// CHECK-LABEL: func.func @subtile_shared_dfb_selects_compatible_strategy
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0, 1>
// CHECK: ttl.tile_exp
// CHECK-NEXT: ttl.tile_add
// CHECK-SAME: ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @subtile_shared_dfb_selects_compatible_strategy(
    %lhs: tensor<1x1x!ttcore.tile<16x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<16x32, f32>>)
    attributes {ttl.enable_fpu_binary_ops = true} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, f32>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<16x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<16x32, f32>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<16x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<16x32, f32>>
  %exponential = ttl.tile_exp %rhs_tile into dst[%zero]
      : !ttcore.tile<16x32, f32> -> !ttcore.tile<16x32, f32>
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      : !ttcore.tile<16x32, f32>, !ttcore.tile<16x32, f32>
        -> !ttcore.tile<16x32, f32>
  return
}
