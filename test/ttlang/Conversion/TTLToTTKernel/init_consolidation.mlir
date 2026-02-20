// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-consolidate-inits %s | FileCheck %s
// Summary: Tests for ttkernel-consolidate-inits pass in isolation.
//
// Verifies that consecutive same-type compute ops share a single init op,
// while type switches get separate inits.

// Test 1: 4 consecutive exp ops -> only 1 init
// CHECK-LABEL: func.func @four_consecutive_exp
// CHECK: ttkernel.exp_tile_init
// CHECK-NEXT: ttkernel.exp_tile
// CHECK-NOT: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK-NOT: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK-NOT: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
func.func @four_consecutive_exp(
    %a: !ttcore.tile<32x32, f32>,
    %b: !ttcore.tile<32x32, f32>,
    %c: !ttcore.tile<32x32, f32>,
    %d: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %e0 = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %e1 = ttl.tile_exp %b {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %e2 = ttl.tile_exp %c {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  %e3 = ttl.tile_exp %d {dst_idx = 3 : i32} : !ttcore.tile<32x32, f32>
  func.return %e3 : !ttcore.tile<32x32, f32>
}

// Test 2: grouped different types -> one init per type
// CHECK-LABEL: func.func @exp_then_log
// CHECK: ttkernel.exp_tile_init
// CHECK-NEXT: ttkernel.exp_tile
// CHECK-NOT: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.log_tile_init
// CHECK-NEXT: ttkernel.log_tile
// CHECK-NOT: ttkernel.log_tile_init
// CHECK: ttkernel.log_tile
func.func @exp_then_log(
    %a: !ttcore.tile<32x32, f32>,
    %b: !ttcore.tile<32x32, f32>,
    %c: !ttcore.tile<32x32, f32>,
    %d: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %e0 = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %e1 = ttl.tile_exp %b {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %l0 = ttl.tile_log %c {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  %l1 = ttl.tile_log %d {dst_idx = 3 : i32} : !ttcore.tile<32x32, f32>
  func.return %l1 : !ttcore.tile<32x32, f32>
}

// Test 3: interleaved ops without scheduling -> init for every type switch
// exp, log, exp, log -> 4 inits (2 per type)
// CHECK-LABEL: func.func @interleaved_no_scheduling
// CHECK: ttkernel.exp_tile_init
// CHECK-NEXT: ttkernel.exp_tile
// CHECK: ttkernel.log_tile_init
// CHECK-NEXT: ttkernel.log_tile
// CHECK: ttkernel.exp_tile_init
// CHECK-NEXT: ttkernel.exp_tile
// CHECK: ttkernel.log_tile_init
// CHECK-NEXT: ttkernel.log_tile
func.func @interleaved_no_scheduling(
    %a: !ttcore.tile<32x32, f32>,
    %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %e0 = ttl.tile_exp %a {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %l0 = ttl.tile_log %e0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %e1 = ttl.tile_exp %l0 {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %l1 = ttl.tile_log %e1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  func.return %l1 : !ttcore.tile<32x32, f32>
}

// Test 4: mixed binary ops -> consolidation respects type identity
// 2 mul then 1 add -> 2 inits total (1 for mul group, 1 for add)
// CHECK-LABEL: func.func @mixed_binary
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: ttkernel.mul_binary_tile_init
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// CHECK-NOT: ttkernel.mul_binary_tile_init
// CHECK: ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C2]])
func.func @mixed_binary(
    %a: !ttcore.tile<32x32, f32>,
    %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %m0 = ttl.tile_mul %a, %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %m1 = ttl.tile_mul %a, %b {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %s0 = ttl.tile_add %m0, %m1 {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  func.return %s0 : !ttcore.tile<32x32, f32>
}
