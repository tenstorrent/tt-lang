// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s | FileCheck %s
// Summary: Tests for ttl.tile_* op lowering to TTKernel with init consolidation.
//
// This tests the tile op patterns followed by init consolidation. The
// convert-ttl-to-ttkernel pass emits compute ops without inits, then
// ttkernel-insert-inits inserts the minimal set of init ops.
// All compute ops are inside tile_regs_acquire/release sync regions as
// required by the hardware.


// CHECK-LABEL: func.func @tile_exp
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_exp(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %exp = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %exp : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_log
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.log_tile_init
// CHECK: ttkernel.log_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_log(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %log = ttl.tile_log %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %log : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_sqrt
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.sqrt_tile_init
// CHECK: ttkernel.sqrt_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_sqrt(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %sqrt = ttl.tile_sqrt %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %sqrt : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_add
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_add(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %sum = ttl.tile_add %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %sum : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_sub
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.sub_binary_tile_init
// CHECK: ttkernel.sub_binary_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_sub(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %diff = ttl.tile_sub %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %diff : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_mul
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.mul_binary_tile_init
// CHECK: ttkernel.mul_binary_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_mul(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %prod = ttl.tile_mul %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %prod : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_max
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.binary_max_tile_init
// CHECK: ttkernel.binary_max_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_max(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %max = ttl.tile_max %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %max : !ttcore.tile<32x32, f32>
}

// CHECK-LABEL: func.func @tile_chain
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_chain(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %sum = ttl.tile_add %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %exp = ttl.tile_exp %sum {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %exp : !ttcore.tile<32x32, f32>
}

// Test that block arguments (function parameters) use their argument number as dst_idx.
// Also tests reusing a block argument in a second operation.
// CHECK-LABEL: func.func @tile_add_block_args
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: ttkernel.tile_regs_acquire
// First add: a (arg0) + b (arg1) -> DST[2]
// Both adds are the same op type so only ONE init should be emitted.
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C2]])
// Second add: same type, no re-init needed.
// CHECK-NOT: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile(%[[C2]], %[[C0]], %[[C3]])
// CHECK: ttkernel.tile_regs_release
func.func @tile_add_block_args(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %sum = ttl.tile_add %a, %b {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  %result = ttl.tile_add %sum, %a {dst_idx = 3 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %result : !ttcore.tile<32x32, f32>
}

// Test axby pattern: a*x + b*y with 4 block arguments and 2 binary ops.
// This validates the fix for the bug where adaptor operands were used instead of
// original operands, causing incorrect register indices for the second multiply.
//
// Block arguments %a, %x, %b, %y map to DST indices 0, 1, 2, 3 respectively.
// Consolidation: one init for the two muls, then one init for the add.
// CHECK-LABEL: func.func @tile_axby_pattern
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: ttkernel.tile_regs_acquire
// First multiply: a (arg0=DST[0]) * x (arg1=DST[1]) -> DST[0]
// CHECK: ttkernel.mul_binary_tile_init
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// Second multiply: same type, no re-init needed.
// CHECK-NOT: ttkernel.mul_binary_tile_init
// CHECK: ttkernel.mul_binary_tile(%[[C2]], %[[C3]], %[[C1]])
// Add: different type, new init.
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C2]])
// CHECK: ttkernel.tile_regs_release
func.func @tile_axby_pattern(%a: !ttcore.tile<32x32, f32>, %x: !ttcore.tile<32x32, f32>,
                             %b: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  ttkernel.tile_regs_acquire() : () -> ()
  %term1 = ttl.tile_mul %a, %x {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %term2 = ttl.tile_mul %b, %y {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %result = ttl.tile_add %term1, %term2 {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %result : !ttcore.tile<32x32, f32>
}
