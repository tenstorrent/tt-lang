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
// CHECK: ttkernel.binary_max_tile_init
// CHECK: ttkernel.binary_max_tile
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

// Test that block arguments (function parameters) use their argument number as dst_idx.
// This supports testing tile ops in isolation without copy_tile operations.
// Also tests reusing a block argument in a second operation.
// CHECK-LABEL: func.func @tile_add_block_args
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// First add: a (arg0) + b (arg1) -> DST[2]
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C2]])
// Second add: sum (DST[2]) + a (arg0, reused) -> DST[3]
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C2]], %[[C0]], %[[C3]])
func.func @tile_add_block_args(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %sum = ttl.tile_add %a, %b {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  %result = ttl.tile_add %sum, %a {dst_idx = 3 : i32} : !ttcore.tile<32x32, f32>
  func.return %result : !ttcore.tile<32x32, f32>
}

// Test axby pattern: a*x + b*y with 4 block arguments and 2 binary ops.
// This validates the fix for the bug where adaptor operands were used instead of
// original operands, causing incorrect register indices for the second multiply.
// The bug would have generated mul_binary_tile(0, 1, 1) for the second multiply
// instead of the correct mul_binary_tile(2, 3, 1).
//
// Block arguments %a, %x, %b, %y map to DST indices 0, 1, 2, 3 respectively.
// CHECK-LABEL: func.func @tile_axby_pattern
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// First multiply: a (arg0=DST[0]) * x (arg1=DST[1]) -> DST[0]
// CHECK: ttkernel.mul_binary_tile_init
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// Second multiply: b (arg2=DST[2]) * y (arg3=DST[3]) -> DST[1]
// CRITICAL: This checks that we use indices 2 and 3, not 0 and 1.
// Before the fix, the adaptor bug caused this to generate (0, 1, 1).
// CHECK: ttkernel.mul_binary_tile_init
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C2]], %[[C3]], %[[C1]])
// Add: term1 (DST[0]) + term2 (DST[1]) -> DST[2]
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C2]])
func.func @tile_axby_pattern(%a: !ttcore.tile<32x32, f32>, %x: !ttcore.tile<32x32, f32>,
                             %b: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %term1 = ttl.tile_mul %a, %x {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %term2 = ttl.tile_mul %b, %y {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %result = ttl.tile_add %term1, %term2 {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  func.return %result : !ttcore.tile<32x32, f32>
}
