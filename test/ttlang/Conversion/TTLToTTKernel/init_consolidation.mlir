// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s | FileCheck %s
// RUN: ttlang-opt --ttkernel-insert-inits %s | FileCheck %s --check-prefix=FPU
// RUN: ttlang-opt --ttkernel-insert-inits %s | FileCheck %s --check-prefix=COMMON
// Summary: Tests for ttkernel-insert-inits pass.
//
// Phase 1 (common init): Inserts init_sfpu or binary_op_init_common before
// each sync region (tile_regs_acquire ... tile_regs_release).
// Phase 2 (per-op init): Consecutive same-type compute ops share a single
// init op, while type switches get separate inits.

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

// Test 5: FPU binary ops (add_tiles, mul_tiles) -> one init per group
// Uses TTKernel ops directly (second RUN line with --ttkernel-insert-inits only).
// 2 add_tiles then 2 mul_tiles -> 1 add_tiles_init + 1 mul_tiles_init
// FPU-LABEL: func.func @fpu_binary_consolidation
// FPU-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG: %[[C3:.*]] = arith.constant 3 : index
// FPU: ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU-NEXT: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]])
// FPU-NOT: ttkernel.add_tiles_init
// FPU: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C1]], %[[C1]], %[[C1]])
// FPU: ttkernel.mul_tiles_init(%[[CB0]], %[[CB1]])
// FPU-NEXT: ttkernel.mul_tiles(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C2]])
// FPU-NOT: ttkernel.mul_tiles_init
// FPU: ttkernel.mul_tiles(%[[CB0]], %[[CB1]], %[[C1]], %[[C1]], %[[C3]])
func.func @fpu_binary_consolidation() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  ttkernel.add_tiles(%cb0, %cb1, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.add_tiles(%cb0, %cb1, %c1, %c1, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.mul_tiles(%cb0, %cb1, %c0, %c0, %c2) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.mul_tiles(%cb0, %cb1, %c1, %c1, %c3) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  func.return
}

// =============================================================================
// Phase 1 tests: common init insertion before sync regions
// =============================================================================

// Test 6: SFPU sync region -> init_sfpu inserted before acquire
// COMMON-LABEL: func.func @common_init_sfpu
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.copy_tile(%[[CB0]],
// COMMON: ttkernel.exp_tile(
// COMMON: ttkernel.pack_tile({{.*}}, %[[CB2]],
// COMMON: ttkernel.tile_regs_release
// No duplicate init_sfpu
// COMMON-NOT: ttkernel.init_sfpu
func.func @common_init_sfpu() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.pack_tile(%c0, %cb2, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Test 7: FPU binary sync region -> binary_op_init_common inserted before acquire
// COMMON-LABEL: func.func @common_init_fpu_binary
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// COMMON: ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB2]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.add_tiles(%[[CB0]], %[[CB1]],
// COMMON: ttkernel.pack_tile({{.*}}, %[[CB2]],
// COMMON: ttkernel.tile_regs_release
// No duplicate binary_op_init_common
// COMMON-NOT: ttkernel.binary_op_init_common
func.func @common_init_fpu_binary() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.add_tiles(%cb0, %cb1, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.pack_tile(%c0, %cb2, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Test 8: Two sync regions -> each gets its own common init
// COMMON-LABEL: func.func @two_sync_regions_two_inits
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// First region
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.tile_regs_release
// Second region
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.tile_regs_release
func.func @two_sync_regions_two_inits() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // First sync region
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.pack_tile(%c0, %cb2, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  // Second sync region
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.log_tile(%c0) : (index) -> ()
  ttkernel.pack_tile(%c0, %cb2, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Test 9: Compiler-generated loop (ttl.tile_loop) -> common init hoisted above
// COMMON-LABEL: func.func @common_init_hoisted_above_compiler_loop
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: scf.for
// COMMON: ttkernel.tile_regs_acquire
// COMMON: ttkernel.exp_tile(
// COMMON: ttkernel.tile_regs_release
func.func @common_init_hoisted_above_compiler_loop() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%cb0, %i, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.pack_tile(%c0, %cb2, %i, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop = 1 : index}
  func.return
}

// Test 10: Unmarked loop -> init NOT hoisted (stays inside loop).
// Only compiler-marked loops are safe to hoist through.
// COMMON-LABEL: func.func @common_init_not_hoisted_past_unmarked_loop
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// COMMON: scf.for
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.tile_regs_release
func.func @common_init_not_hoisted_past_unmarked_loop() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%cb0, %i, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.pack_tile(%c0, %cb2, %i, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  func.return
}

// Test 11: Nested compiler loops (tile_loop wrapping subblock_stride) -> common
// init hoisted above both loops.
// COMMON-LABEL: func.func @common_init_hoisted_above_nested_compiler_loops
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// COMMON-NEXT: scf.for
// COMMON: scf.for
// COMMON: ttkernel.tile_regs_acquire
// COMMON: ttkernel.exp_tile(
// COMMON: ttkernel.tile_regs_release
func.func @common_init_hoisted_above_nested_compiler_loops() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.copy_tile(%cb0, %j, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
      ttkernel.exp_tile(%c0) : (index) -> ()
      ttkernel.pack_tile(%c0, %cb2, %j, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.subblock_stride = 1 : index}
  } {ttl.tile_loop = 1 : index}
  func.return
}

// Test 12: CopyTile from different CBs -> separate copy_tile_init per CB group.
// After scheduling (or manual pre-grouping), copies from cb0 are consecutive
// and copies from cb1 are consecutive. Each group gets its own init.
// FPU-LABEL: func.func @copy_tile_init_per_cb
// FPU-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU: ttkernel.copy_tile_init(%[[CB0]])
// FPU-NEXT: ttkernel.copy_tile(%[[CB0]],
// FPU-NOT: ttkernel.copy_tile_init
// FPU: ttkernel.copy_tile(%[[CB0]],
// FPU: ttkernel.copy_tile_init(%[[CB1]])
// FPU-NEXT: ttkernel.copy_tile(%[[CB1]],
// FPU-NOT: ttkernel.copy_tile_init
// FPU: ttkernel.copy_tile(%[[CB1]],
func.func @copy_tile_init_per_cb() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Copies from cb0 (grouped)
  ttkernel.copy_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.copy_tile(%cb0, %c1, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  // Copies from cb1 (grouped) -> new init
  ttkernel.copy_tile(%cb1, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.copy_tile(%cb1, %c1, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  func.return
}

// Test 13: Multiple output CBs with same data format -> accepted, one common init.
// When two pack ops target different CBs that share the same element type,
// PACK data format routing is identical and one common init suffices.
// COMMON-LABEL: func.func @multi_output_cb_same_format
// COMMON-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// COMMON-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// COMMON: ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// COMMON-NEXT: ttkernel.tile_regs_acquire
// COMMON: ttkernel.pack_tile({{.*}}, %[[CB1]],
// COMMON: ttkernel.pack_tile({{.*}}, %[[CB2:.*]],
// COMMON: ttkernel.tile_regs_release
func.func @multi_output_cb_same_format() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.copy_tile(%cb0, %c1, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.exp_tile(%c1) : (index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb1, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb2, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
