// Verify direct tile operations in nested regions use the same strategy and
// dataflow-buffer resolution as operations in ttl.compute.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{matmul-full-fp32=0 reduce-full-fp32=0})' | FileCheck %s

// CHECK-LABEL: func.func @nested_direct_tile_ops
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// CHECK: scf.if
// CHECK: ttl.tile_abs
// CHECK: scf.for
// CHECK: ttl.tile_add {{.*}}ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>
func.func @nested_direct_tile_ops(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>, %condition: i1) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %input_tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  scf.if %condition {
    %absolute = ttl.tile_abs %input_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    scf.yield
  }
  scf.for %iteration = %zero to %one step %one {
    %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
        : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
          -> !ttcore.tile<32x32, f32>
    scf.yield
  }
  return
}

// -----

// A resultless DST consumer contributes both DST and unpack requirements for a
// DFB-backed compute input.
// CHECK-LABEL: func.func @f32_store_of_compute_input
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 3>
func.func @f32_store_of_compute_input(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %output: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %input_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_attached = ttl.attach_cb %output, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %result = ttl.compute
      ins(%input_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
       %output_tile: !ttcore.tile<32x32, f32>):
    ttl.tile_store %input_tile, %output_view[%zero, %zero] from dst[%zero]
        : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return
}

// -----

// A compute output block argument retains the DFB identity of its output
// operand when consumed through DST.
// CHECK-LABEL: func.func @f32_compute_output_argument
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 5>
func.func @f32_compute_output_argument(
    %output: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %output_dfb = ttl.bind_cb {cb_index = 5, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %output_attached = ttl.attach_cb %output, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %result = ttl.compute
      ins()
      outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%output_tile: !ttcore.tile<32x32, f32>):
    %absolute = ttl.tile_abs %output_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %absolute, %output_view[%zero, %zero] from dst[%zero]
        : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// An explicit DFB-to-DST copy requires f32 unpack mode for its source DFB.
// CHECK-LABEL: func.func @explicit_f32_copy
// CHECK-SAME: fp32_dest_acc_en = true
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32: 4>
func.func @explicit_f32_copy(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %input_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_attached = ttl.attach_cb %output, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %result = ttl.compute
      ins(%input_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<32x32, f32>,
       %output_tile: !ttcore.tile<32x32, f32>):
    %token, %tile = ttl.copy_tile %input_tile[%zero, %zero] into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttl.dst, !ttcore.tile<32x32, f32>
    ttl.tile_store %tile, %output_view[%zero, %zero] from dst[%zero]
        : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
