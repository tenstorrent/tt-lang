// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),canonicalize)' | FileCheck %s

// Verifies that conversion preserves physical tile dimensions for direct,
// fused, and passthrough compute operations.

// Direct short-height elementwise lowering uses the tensor element tile type
// for every compute block argument and tile result.
// CHECK-LABEL: func.func @direct_short_height_subtile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<1x32, bf16>, %[[OUTPUT:.*]]: !ttcore.tile<1x32, bf16>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[INPUT]]{{.*}} -> !ttcore.tile<1x32, bf16>
// CHECK-NEXT:    ttl.tile_store %[[EXP]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK:       } -> tensor<1x1x!ttcore.tile<1x32, bf16>>
func.func @direct_short_height_subtile(
    %arg: tensor<1x1x!ttcore.tile<1x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<1x32, bf16>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<1x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<1x32, bf16>>
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<1x32, bf16>>,
        tensor<1x1x!ttcore.tile<1x32, bf16>>
  return %result : tensor<1x1x!ttcore.tile<1x32, bf16>>
}

// -----

// Fused lowering preserves physical tile dimensions through intermediate tile
// operations and the output store.
// CHECK-LABEL: func.func @fused_subtile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[LHS:.*]]: !ttcore.tile<32x16, f32>, %[[RHS:.*]]: !ttcore.tile<32x16, f32>, %[[OUTPUT:.*]]: !ttcore.tile<32x16, f32>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[LHS]]{{.*}} -> !ttcore.tile<32x16, f32>
// CHECK-NEXT:    %[[SUM:.*]] = ttl.tile_add %[[EXP]], %[[RHS]]{{.*}} -> !ttcore.tile<32x16, f32>
// CHECK-NEXT:    ttl.tile_store %[[SUM]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK:       } -> tensor<1x1x!ttcore.tile<32x16, f32>>
func.func @fused_subtile(
    %lhs_arg: tensor<1x1x!ttcore.tile<32x16, f32>>,
    %rhs_arg: tensor<1x1x!ttcore.tile<32x16, f32>>)
    -> tensor<1x1x!ttcore.tile<32x16, f32>> {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>
  %lhs = ttl.attach_cb %lhs_arg, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x16, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %rhs = ttl.attach_cb %rhs_arg, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x16, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %exp = ttl.exp %lhs
      : tensor<1x1x!ttcore.tile<32x16, f32>>
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %result = ttl.add %exp, %rhs
      : tensor<1x1x!ttcore.tile<32x16, f32>>,
        tensor<1x1x!ttcore.tile<32x16, f32>>
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x16, f32>>,
        tensor<1x1x!ttcore.tile<32x16, f32>>
  return %result : tensor<1x1x!ttcore.tile<32x16, f32>>
}

// -----

// Passthrough-store lowering preserves physical tile dimensions when it
// constructs the compute block arguments.
// CHECK-LABEL: func.func @passthrough_subtile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<16x16, bf16>, %[[OUTPUT:.*]]: !ttcore.tile<16x16, bf16>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    ttl.tile_store %[[INPUT]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT:    ttl.yield
// CHECK-NEXT:  } -> tensor<1x1x!ttcore.tile<16x16, bf16>>
func.func @passthrough_subtile(%arg: tensor<1x1x!ttcore.tile<16x16, bf16>>)
    -> tensor<1x1x!ttcore.tile<16x16, bf16>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<16x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<16x16, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<16x16, bf16>>
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<16x16, bf16>>,
        tensor<1x1x!ttcore.tile<16x16, bf16>>
  return %input : tensor<1x1x!ttcore.tile<16x16, bf16>>
}

// -----

// Passthrough uses only unpack and pack, so it accepts storage-valid short and
// narrow physical tile dimensions without requiring a matmul in the kernel.
// CHECK-LABEL: func.func @passthrough_short_narrow_tile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<4x16, bf16>, %[[OUTPUT:.*]]: !ttcore.tile<4x16, bf16>):
// CHECK:         ttl.tile_store %[[INPUT]],
// CHECK:       } -> tensor<1x1x!ttcore.tile<4x16, bf16>>
func.func @passthrough_short_narrow_tile(
    %arg: tensor<1x1x!ttcore.tile<4x16, bf16>>)
    -> tensor<1x1x!ttcore.tile<4x16, bf16>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<4x16, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<4x16, bf16>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<4x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<4x16, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<4x16, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<4x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<4x16, bf16>>
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<4x16, bf16>>,
        tensor<1x1x!ttcore.tile<4x16, bf16>>
  return %input : tensor<1x1x!ttcore.tile<4x16, bf16>>
}

// -----

// A short-height passthrough remains valid without an enclosing matmul.
// CHECK-LABEL: func.func @passthrough_short_tile
// CHECK:       ^bb0(%[[INPUT:.*]]: !ttcore.tile<8x32, f32>, %[[OUTPUT:.*]]: !ttcore.tile<8x32, f32>):
// CHECK:         ttl.tile_store %[[INPUT]],
func.func @passthrough_short_tile(
    %arg: tensor<1x1x!ttcore.tile<8x32, f32>>)
    -> tensor<1x1x!ttcore.tile<8x32, f32>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, f32>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<8x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<8x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<8x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<8x32, f32>>
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<8x32, f32>>,
        tensor<1x1x!ttcore.tile<8x32, f32>>
  return %input : tensor<1x1x!ttcore.tile<8x32, f32>>
}

// -----

// Passthrough retains the existing f16 compiler support.
// CHECK-LABEL: func.func @passthrough_f16
// CHECK:       ^bb0(%[[INPUT:.*]]: !ttcore.tile<32x32, f16>, %[[OUTPUT:.*]]: !ttcore.tile<32x32, f16>):
// CHECK:         ttl.tile_store %[[INPUT]],
func.func @passthrough_f16(
    %arg: tensor<1x1x!ttcore.tile<32x32, f16>>)
    -> tensor<1x1x!ttcore.tile<32x32, f16>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<32x32, f16>>,
        tensor<1x1x!ttcore.tile<32x32, f16>>
  return %input : tensor<1x1x!ttcore.tile<32x32, f16>>
}

// -----

// Direct fill lowering preserves the result's physical tile dimensions.
// CHECK-LABEL: func.func @direct_fill_subtile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[OUTPUT:.*]]: !ttcore.tile<16x32, bf16>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[FILL:.*]] = ttl.tile_fill 1.250000e+00{{.*}} : !ttcore.tile<16x32, bf16>
// CHECK-NEXT:    ttl.tile_store %[[FILL]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT:    ttl.yield
// CHECK-NEXT:  } -> tensor<1x1x!ttcore.tile<16x32, bf16>>
func.func @direct_fill_subtile()
    -> tensor<1x1x!ttcore.tile<16x32, bf16>> {
  %output_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %result = ttl.fill 1.250000e+00
      : tensor<1x1x!ttcore.tile<16x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, bf16>>,
        tensor<1x1x!ttcore.tile<16x32, bf16>>
  return %result : tensor<1x1x!ttcore.tile<16x32, bf16>>
}

// -----

// Fused fill lowering preserves physical tile dimensions through its consumer.
// CHECK-LABEL: func.func @fused_fill_subtile
// CHECK:       %[[RESULT:.*]] = ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<32x16, f32>, %[[OUTPUT:.*]]: !ttcore.tile<32x16, f32>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[FILL:.*]] = ttl.tile_fill 1.250000e+00{{.*}} : !ttcore.tile<32x16, f32>
// CHECK-NEXT:    %[[SUM:.*]] = ttl.tile_add %[[INPUT]], %[[FILL]]{{.*}} -> !ttcore.tile<32x16, f32>
// CHECK-NEXT:    ttl.tile_store %[[SUM]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT:    ttl.yield
// CHECK-NEXT:  } -> tensor<1x1x!ttcore.tile<32x16, f32>>
func.func @fused_fill_subtile(
    %arg: tensor<1x1x!ttcore.tile<32x16, f32>>)
    -> tensor<1x1x!ttcore.tile<32x16, f32>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x16, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  %fill = ttl.fill 1.250000e+00
      : tensor<1x1x!ttcore.tile<32x16, f32>>
  %result = ttl.add %input, %fill
      : tensor<1x1x!ttcore.tile<32x16, f32>>,
        tensor<1x1x!ttcore.tile<32x16, f32>>
        -> tensor<1x1x!ttcore.tile<32x16, f32>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x16, f32>>,
        tensor<1x1x!ttcore.tile<32x16, f32>>
  return %result : tensor<1x1x!ttcore.tile<32x16, f32>>
}

// -----

// Integer tile types pass through compute creation without scalar-type
// reconstruction.
// CHECK-LABEL: func.func @integer_passthrough_subtile
// CHECK:       ^bb0(%[[INPUT:.*]]: !ttcore.tile<32x16, u32>, %[[OUTPUT:.*]]: !ttcore.tile<32x16, u32>):
// CHECK:         ttl.tile_store %[[INPUT]],
func.func @integer_passthrough_subtile(
    %arg: tensor<1x1x!ttcore.tile<32x16, u32>>)
    -> tensor<1x1x!ttcore.tile<32x16, u32>> {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, u32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, u32>, 2>
  %input = ttl.attach_cb %arg, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x16, u32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, u32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x16, u32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x16, u32>, 2>
        -> tensor<1x1x!ttcore.tile<32x16, u32>>
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<32x16, u32>>,
        tensor<1x1x!ttcore.tile<32x16, u32>>
  return %input : tensor<1x1x!ttcore.tile<32x16, u32>>
}
