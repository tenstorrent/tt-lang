// Summary: ttl.tile_regs_acquire lowers to ttkernel.tile_regs_acquire.
// RUN: ttlang-opt %s --convert-ttl-to-ttkernel | FileCheck %s

// CHECK-LABEL: func.func @acquire_lower
// Purpose: confirm tile_regs_* ops lower to TTKernel equivalents.
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
// CHECK-NOT: ttl.acquire_dst
// CHECK-NOT: ttl.tile_regs_wait
func.func @acquire_lower() {
  ttl.tile_regs_acquire
  ttl.tile_regs_commit
  ttl.tile_regs_wait
  ttl.tile_regs_release
  func.return
}

// -----

// CHECK-LABEL: func.func @acquire_two_compute_lowers
// Purpose: ensure multiple compute regions each lower their reg ops to TTKernel, preserving order.
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
func.func @acquire_two_compute_lowers(%t0: !ttcore.tile<32x32, f32>, %t1: !ttcore.tile<32x32, f32>) {
  // First region
  ttl.tile_regs_acquire
  %a = ttl.tile_add %t0, %t1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttl.tile_regs_commit
  ttl.tile_regs_wait
  ttl.tile_regs_release

  // Second region
  ttl.tile_regs_acquire
  %b = ttl.tile_mul %a, %t1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttl.tile_regs_commit
  ttl.tile_regs_wait
  ttl.tile_regs_release
  func.return
}

// -----

// CHECK-LABEL: func.func @acquire_chain_lowers
// Purpose: chain add->mul->exp with reg ops lowers fully to TTKernel.
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
func.func @acquire_chain_lowers(%t0: !ttcore.tile<32x32, f32>,
                                %t1: !ttcore.tile<32x32, f32>,
                                %t2: !ttcore.tile<32x32, f32>) {
  ttl.tile_regs_acquire
  %a = ttl.tile_add %t0, %t1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  %b = ttl.tile_mul %a, %t2 {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  %c = ttl.tile_exp %b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  ttl.tile_regs_commit
  ttl.tile_regs_wait
  ttl.tile_regs_release
  func.return
}
