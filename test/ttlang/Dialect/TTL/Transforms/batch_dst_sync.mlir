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

// Test: 2x2 binary add subblocks inner dim (dstPerIter=3, maxBatch=8/3=2).
// Inner dim (2) fully unrolled, outer loop remains.

// CHECK-LABEL: func.func @binary_add_2x2_subblock
// CHECK: scf.for
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
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @binary_add_2x2_subblock(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
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

// Test: f32 3x2 subblocks inner dim (dstPerIter=1, capacity=4, maxBatch=4).
// Inner dim (2) fully unrolled, outer loop (3) remains.

// CHECK-LABEL: func.func @f32_3x2_subblock
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @f32_3x2_subblock(%a: tensor<3x2x!ttcore.tile<32x32, f32>>,
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

// -----

// Test: 6x4 unary with partial outer unrolling (dstPerIter=1, capacity=8).
// Inner dim (4) fully unrolled, outer dim (6) partially unrolled by 2.
// Subblock [4, 2] = 8 tiles. Outer loop: for i = 0 to 6 step 2 (3 iterations).

// CHECK-LABEL: func.func @unary_6x4_subblock
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 4 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 5 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 6 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 7 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 2 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 3 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 4 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 5 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 6 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 8 : i64, ttl.tile_offset = 7 : i64}
// CHECK: ttl.tile_regs_release
func.func @unary_6x4_subblock(%a: tensor<6x4x!ttcore.tile<32x32, bf16>>,
                                %view: tensor<6x4x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %ext = tensor.extract %a[%i, %j] : tensor<6x4x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %tok, %tile = ttl.copy_tile %ext, %j, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<6x4x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    }
  }
  return
}

// -----

// Test: 1D partial unroll (9 tiles, capacity=8). largestDivisor(9,8)=3.
// Loop partially unrolled by 3 (step=3). 3 subblocks of 3 tiles.

// CHECK-LABEL: func.func @unary_9_partial
// CHECK: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_exp {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 3 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 3 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 3 : i64, ttl.tile_offset = 2 : i64}
// CHECK: ttl.tile_regs_release
func.func @unary_9_partial(%a: tensor<9x!ttcore.tile<32x32, bf16>>,
                            %view: tensor<9x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  scf.for %i = %c0 to %c9 step %c1 {
    %ext = tensor.extract %a[%i] : tensor<9x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok, %tile = ttl.copy_tile %ext, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<9x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: Binary 4-tile with partial unroll (dstPerIter=3, maxBatch=8/3=2).
// largestDivisor(4,2)=2. Loop step=2. 2 tiles batched per sync cycle.

// CHECK-LABEL: func.func @binary_4_partial
// CHECK: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 4 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 5 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @binary_4_partial(%a: tensor<4x!ttcore.tile<32x32, bf16>>,
                              %b: tensor<4x!ttcore.tile<32x32, bf16>>,
                              %view: tensor<4x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %ext_a = tensor.extract %a[%i] : tensor<4x!ttcore.tile<32x32, bf16>>
    %ext_b = tensor.extract %b[%i] : tensor<4x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok_a, %tile_a = ttl.copy_tile %ext_a, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %tok_b, %tile_b = ttl.copy_tile %ext_b, %i, %c1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %tile_a, %tile_b {dst_idx = 2 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %sum, %view : !ttcore.tile<32x32, bf16>, tensor<4x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: Prime tile count (11 bf16, capacity=8). No divisor > 1 fits.
// Loop remains unchanged.

// CHECK-LABEL: func.func @unary_prime_skip
// CHECK: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_regs_release
// CHECK-NOT: ttl.tile_offset
func.func @unary_prime_skip(%a: tensor<11x!ttcore.tile<32x32, bf16>>,
                             %view: tensor<11x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  scf.for %i = %c0 to %c11 step %c1 {
    %ext = tensor.extract %a[%i] : tensor<11x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok, %tile = ttl.copy_tile %ext, %i, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %exp, %view : !ttcore.tile<32x32, bf16>, tensor<11x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}

// -----

// Test: Hypothetical post-DSE user-defined accumulation loop.
// Represents a user loop over M=4 independent output tiles, each accumulating
// K=2 inputs (K unrolled into the body). CB lifecycle ops are outside the loop
// (in a 'with' scope). The pass operates at SCF level and is agnostic to
// whether the loop came from tiling or user code.
//
// dstPerIter=3 (binary add), capacity=8, maxBatch=8/3=2.
// largestDivisor(4,2)=2. Partial unroll by 2. 2 sync cycles.

// CHECK-LABEL: func.func @user_loop_accum_subblock
// CHECK: scf.for
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.copy_tile {{.*}} {dst_idx = 4 : i32}
// CHECK: ttl.tile_add {{.*}} {dst_idx = 5 : i32}
// CHECK: ttl.tile_regs_commit
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 0 : i64}
// CHECK: ttl.tile_store {{.*}} {ttl.subblock_stride = 2 : i64, ttl.tile_offset = 1 : i64}
// CHECK: ttl.tile_regs_release
func.func @user_loop_accum_subblock(%a: tensor<4x2x!ttcore.tile<32x32, bf16>>,
                                     %view: tensor<4x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %m = %c0 to %c4 step %c1 {
    %ext0 = tensor.extract %a[%m, %c0] : tensor<4x2x!ttcore.tile<32x32, bf16>>
    %ext1 = tensor.extract %a[%m, %c1] : tensor<4x2x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_acquire
    %tok0, %t0 = ttl.copy_tile %ext0, %m, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %tok1, %t1 = ttl.copy_tile %ext1, %m, %c1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %t0, %t1 {dst_idx = 2 : i32} : !ttcore.tile<32x32, bf16>
    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.tile_store %sum, %view : !ttcore.tile<32x32, bf16>, tensor<4x!ttcore.tile<32x32, bf16>>
    ttl.tile_regs_release
  }
  return
}
