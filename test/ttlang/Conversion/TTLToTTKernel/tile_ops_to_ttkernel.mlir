// RUN: ttlang-opt --convert-ttl-to-ttkernel %s | FileCheck %s
// Summary: Tests for ttl.tile_* op lowering to TTKernel.
//
// This tests the tile op patterns only. The ttl.compute op is NOT lowered by
// this pass - it will be lowered to scf.for loops by ttl-lower-to-loops first,
// then this pass converts the remaining ttl.tile_* ops to ttkernel ops.
//
// TODO(#124): Add DST lifecycle wrapper tests once full pipeline is integrated.

// CHECK-LABEL: func.func @tile_exp
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
func.func @tile_exp(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %exp = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %exp : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_log
// CHECK: ttkernel.log_tile_init
// CHECK: ttkernel.log_tile
func.func @tile_log(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %log = ttl.tile_log %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %log : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_sqrt
// CHECK: ttkernel.sqrt_tile_init
// CHECK: ttkernel.sqrt_tile
func.func @tile_sqrt(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %sqrt = ttl.tile_sqrt %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %sqrt : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_add
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
func.func @tile_add(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %sum = ttl.tile_add %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %sum : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_sub
// CHECK: ttkernel.sub_binary_tile_init
// CHECK: ttkernel.sub_binary_tile
func.func @tile_sub(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %diff = ttl.tile_sub %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %diff : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_mul
// CHECK: ttkernel.mul_binary_tile_init
// CHECK: ttkernel.mul_binary_tile
func.func @tile_mul(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %prod = ttl.tile_mul %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_max
// CHECK: ttkernel.max_tile_init
// CHECK: ttkernel.max_tile
func.func @tile_max(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %max = ttl.tile_max %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %max : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_chain
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
func.func @tile_chain(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %sum = ttl.tile_add %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %exp = ttl.tile_exp %sum {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %exp : !ttcore.tile<32x32, f32>
}

//===----------------------------------------------------------------------===//
// Tests for copy_tile emission with ttl.compute context
//===----------------------------------------------------------------------===//
// When tile ops are inside ttl.compute with attached CBs, copy_tile_init and
// copy_tile should be emitted before the compute op.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @unary_with_compute
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CB_IN:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
// CHECK: ttl.compute
// CHECK:   ttkernel.copy_tile_init(%[[CB_IN]])
// CHECK:   ttkernel.copy_tile(%[[CB_IN]], %[[C0]], %[[C0]])
// CHECK:   ttkernel.exp_tile_init
// CHECK:   ttkernel.exp_tile(%[[C0]])
func.func @unary_with_compute(
    %input_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %output_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb_in = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb_out = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %attached_in = ttl.attach_cb %input_tensor, %cb_in
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %attached_out = ttl.attach_cb %output_tensor, %cb_out
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute ins(%attached_in : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%attached_out : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %in {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @binary_with_compute
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[CB_LHS:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
// CHECK: %[[CB_RHS:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
// CHECK: ttl.compute
// CHECK:   ttkernel.copy_tile_init(%[[CB_LHS]])
// CHECK:   ttkernel.copy_tile(%[[CB_LHS]], %[[C0]], %[[C0]])
// CHECK:   ttkernel.copy_tile_init(%[[CB_RHS]])
// CHECK:   ttkernel.copy_tile(%[[CB_RHS]], %[[C0]], %[[C1]])
// CHECK:   ttkernel.add_binary_tile_init
// CHECK:   ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
func.func @binary_with_compute(
    %lhs_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %output_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb_lhs = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb_rhs = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb_out = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %attached_lhs = ttl.attach_cb %lhs_tensor, %cb_lhs
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %attached_rhs = ttl.attach_cb %rhs_tensor, %cb_rhs
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %attached_out = ttl.attach_cb %output_tensor, %cb_out
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%attached_lhs, %attached_rhs : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%attached_out : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs: !ttcore.tile<32x32, f32>, %rhs: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %lhs, %rhs {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
