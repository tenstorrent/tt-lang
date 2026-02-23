// RUN: ttlang-opt %s --split-input-file --ttl-batch-dst-sync | FileCheck %s

// Test: Unary op over 2 tiles (dstPerIter=1, totalTrip=2, 2<=8).
// The loop should be fully unrolled with batched sync.

// CHECK-LABEL: func.func @unary_2_tiles
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @unary_2_tiles(%a: tensor<2x!ttcore.tile<32x32, bf16>>,
                          %view: tensor<2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %ext = tensor.extract %a[%i] : tensor<2x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok, %tile = ttl.copy_tile %ext, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<2x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: Binary add over 2 tiles (dstPerIter=3, totalTrip=2, 2*3=6<=8).

// CHECK-LABEL: func.func @binary_add_2_tiles
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 4 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 5 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @binary_add_2_tiles(%a: tensor<2x!ttcore.tile<32x32, bf16>>,
                               %b: tensor<2x!ttcore.tile<32x32, bf16>>,
                               %view: tensor<2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %ext_a = tensor.extract %a[%i] : tensor<2x!ttcore.tile<32x32, bf16>>
    %ext_b = tensor.extract %b[%i] : tensor<2x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok_a, %tile_a = ttl.copy_tile %ext_a, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %tok_b, %tile_b = ttl.copy_tile %ext_b, %i, %c1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %tile_a, %tile_b {dst_idx = 2 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %sum, %view : !ttcore.tile<32x32, bf16>, tensor<2x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: 2x2 binary add does NOT batch (dstPerIter=3, totalTrip=4, 4*3=12>8).
// Loops should remain.

// CHECK-LABEL: func.func @binary_add_2x2_skip
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_regs_release
func.func @binary_add_2x2_skip(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                %view: tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %ext_a = tensor.extract %a[%i, %j] : tensor<2x2x!ttcore.tile<32x32, bf16>>
      %ext_b = tensor.extract %b[%i, %j] : tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %tok_a, %tile_a = ttl.copy_tile %ext_a, %j, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %tok_b, %tile_b = ttl.copy_tile %ext_b, %j, %c1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %sum = ttl.tile_add %tile_a, %tile_b {dst_idx = 2 : i32} : !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %sum, %view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    }
  }
  return
}

// -----

// Test: 2x2 unary DOES batch (dstPerIter=1, totalTrip=4, 4*1=4<=8).

// CHECK-LABEL: func.func @unary_2x2_batch
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 2 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 3 : i64}
// CHECK: ttl.tile_regs_release
func.func @unary_2x2_batch(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                            %view: tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %ext = tensor.extract %a[%i, %j] : tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %tok, %tile = ttl.copy_tile %ext, %j, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    }
  }
  return
}

// -----

// Test: f32 tiles with capacity=4 (dstPerIter=1, totalTrip=4, 4<=4).
// Exactly fills DST; should batch.

// CHECK-LABEL: func.func @f32_2x2_batch
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_regs_release
func.func @f32_2x2_batch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                           %view: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %ext = tensor.extract %a[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
      ttl.tile_regs_acquire
      %tok, %tile = ttl.copy_tile %ext, %j, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %exp, %view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
      ttl.tile_regs_release
    }
  }
  return
}

// -----

// Test: f32 3x2 exceeds capacity (dstPerIter=1, totalTrip=6, 6>4). Skips.

// CHECK-LABEL: func.func @f32_3x2_skip
// CHECK: scf.for
func.func @f32_3x2_skip(%a: tensor<3x2x!ttcore.tile<32x32, f32>>,
                          %view: tensor<3x2x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %ext = tensor.extract %a[%i, %j] : tensor<3x2x!ttcore.tile<32x32, f32>>
      ttl.tile_regs_acquire
      %tok, %tile = ttl.copy_tile %ext, %j, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %exp, %view : !ttcore.tile<32x32, f32>, tensor<3x2x!ttcore.tile<32x32, f32>>
      ttl.tile_regs_release
    }
  }
  return
}

// -----

// Test: f32 with only 2 tiles DOES batch (capacity=4, 2*1=2<=4).

// CHECK-LABEL: func.func @f32_2_tiles_batch
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @f32_2_tiles_batch(%a: tensor<2x!ttcore.tile<32x32, f32>>,
                              %view: tensor<2x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %ext = tensor.extract %a[%i] : tensor<2x!ttcore.tile<32x32, f32>>
    ttl.tile_regs_acquire
    %tok, %tile = ttl.copy_tile %ext, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %exp, %view : !ttcore.tile<32x32, f32>, tensor<2x!ttcore.tile<32x32, f32>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: Single tile (totalTrip=1) does NOT batch (no benefit).

// CHECK-LABEL: func.func @single_tile_skip
// CHECK-NOT: ttl.tile_offset
func.func @single_tile_skip(%a: tensor<1x!ttcore.tile<32x32, bf16>>,
                             %view: tensor<1x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %ext = tensor.extract %a[%i] : tensor<1x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok, %tile = ttl.copy_tile %ext, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<1x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}
