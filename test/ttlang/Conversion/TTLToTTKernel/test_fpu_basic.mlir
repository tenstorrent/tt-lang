// RUN: ttlang-opt --convert-ttl-to-ttkernel %s | FileCheck %s
// Summary: Test FPU binary operation lowering for add, sub, mul when both
// operands are block arguments (from circular buffers).
//
// This verifies Phase 2 of the FPU optimization:
// - FPU pattern matches when both operands are block arguments
// - Generates ttkernel.add_tiles (FPU) instead of copy_tile + add_binary_tile (SFPU)
// - Fallback to SFPU still works when operands are DST intermediates

// Test 1: FPU Add - both operands from CB (block arguments)
// CHECK-LABEL: func.func @test_fpu_add
// CHECK: ttkernel.add_tiles_init
// CHECK-NEXT: ttkernel.add_tiles
// CHECK-NOT: ttkernel.copy_tile
// CHECK-NOT: ttkernel.add_binary_tile
func.func @test_fpu_add(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %sum = ttl.tile_add %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %sum : !ttcore.tile<32x32, f32>
}

// Test 2: FPU Sub - both operands from CB
// CHECK-LABEL: func.func @test_fpu_sub
// CHECK: ttkernel.sub_tiles_init
// CHECK-NEXT: ttkernel.sub_tiles
// CHECK-NOT: ttkernel.copy_tile
// CHECK-NOT: ttkernel.sub_binary_tile
func.func @test_fpu_sub(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %diff = ttl.tile_sub %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %diff : !ttcore.tile<32x32, f32>
}

// Test 3: FPU Mul - both operands from CB
// CHECK-LABEL: func.func @test_fpu_mul
// CHECK: ttkernel.mul_tiles_init
// CHECK-NEXT: ttkernel.mul_tiles
// CHECK-NOT: ttkernel.copy_tile
// CHECK-NOT: ttkernel.mul_binary_tile
func.func @test_fpu_mul(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %prod = ttl.tile_mul %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// Test 4: SFPU Fallback - both operands are DST intermediates
// When operands come from previous tile operations (DST intermediates),
// the FPU pattern should fail and fall back to SFPU operations.
// CHECK-LABEL: func.func @test_sfpu_fallback
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile
// CHECK-NOT: ttkernel.add_tiles
func.func @test_sfpu_fallback(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %c = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %d = ttl.tile_exp %b {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %sum = ttl.tile_add %c, %d {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  func.return %sum : !ttcore.tile<32x32, f32>
}

// Test 5: Mixed operands - one CB, one DST
// Currently falls back to SFPU (Phase 3 would add dest-reuse pattern for this)
// CHECK-LABEL: func.func @test_mixed_operands
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile
// CHECK-NOT: ttkernel.add_tiles
func.func @test_mixed_operands(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %c = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %sum = ttl.tile_add %c, %b {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  func.return %sum : !ttcore.tile<32x32, f32>
}
