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
