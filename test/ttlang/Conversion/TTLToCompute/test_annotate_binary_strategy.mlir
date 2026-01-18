// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-annotate-binary-op-strategy))' --split-input-file | FileCheck %s
// Summary: Test TTLAnnotateBinaryOpStrategy analysis pass.
//
// Verifies that binary tile operations are correctly annotated with execution
// strategy based on operand provenance within ttl.compute regions:
// - "fpu": Both operands are block arguments (from CB)
// - "dest_reuse": Exactly one operand is block argument
// - "sfpu": Neither operand is block argument (both from DST)

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test 1: FPU - both operands are block arguments
// This is the most common case for simple binary operations.
// The ttl.compute region receives two inputs, and the tile operation uses both.
// CHECK-LABEL: func.func @fpu_both_block_args
func.func @fpu_both_block_args(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                                %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // After convert-ttl-to-compute, this creates a compute region with 2 block args
  // Inside the region: ttl.tile_add %arg2, %arg3 where both are block args
  // CHECK: ttl.tile_add{{.*}}execution_target = "fpu"
  %result = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 2: SFPU - both operands from DST (fused operations)
// When operations are fused in the same compute region, intermediates are DST values.
// CHECK-LABEL: func.func @sfpu_both_dst
func.func @sfpu_both_dst(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                          %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Fuse exp(a) + exp(b) into one operation
  // This will create a single compute with 2 exp ops followed by 1 add
  %exp_a = ttl.exp %a : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %exp_b = ttl.exp %b : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Verify compute region contains all three tile operations
  // CHECK: ttl.compute
  // CHECK: ^bb0
  // First exp: should NOT be annotated (unary operations don't get execution_target)
  // CHECK: ttl.tile_exp
  // CHECK-NOT: execution_target
  // Second exp: should NOT be annotated
  // CHECK: ttl.tile_exp
  // CHECK-NOT: execution_target
  // Add: SHOULD be annotated with "sfpu" (both operands from tile ops)
  // CHECK: ttl.tile_add{{.*}}execution_target = "sfpu"
  // CHECK: ttl.yield
  %result = ttl.add %exp_a, %exp_b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 3: Dest-reuse - LHS is DST intermediate, RHS is block arg
// One operand comes from a tile operation (DST), the other is a block arg (CB).
// CHECK-LABEL: func.func @dest_reuse_lhs_dst
func.func @dest_reuse_lhs_dst(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                               %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Fuse exp(a) + b
  %exp_a = ttl.exp %a : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // Verify tile operations in compute region
  // CHECK: ttl.compute
  // Exp should NOT be annotated
  // CHECK: ttl.tile_exp
  // CHECK-NOT: execution_target
  // Add SHOULD be annotated with "dest_reuse" (LHS=DST, RHS=block arg)
  // CHECK: ttl.tile_add{{.*}}execution_target = "dest_reuse"
  %result = ttl.add %exp_a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 4: Dest-reuse - RHS is DST intermediate, LHS is block arg
// Same as Test 3 but with operands swapped.
// CHECK-LABEL: func.func @dest_reuse_rhs_dst
func.func @dest_reuse_rhs_dst(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                               %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Fuse a + exp(b) (operands in opposite order from Test 3)
  // Inside compute region:
  // ^bb0(%arg2, %arg3, %out):
  //   %exp = ttl.tile_exp %arg3  // tile op result (DST)
  //   %sum = ttl.tile_add %arg2, %exp  // LHS=block arg, RHS=DST -> dest_reuse!
  %exp_b = ttl.exp %b : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // CHECK: ttl.tile_add{{.*}}execution_target = "dest_reuse"
  %result = ttl.add %a, %exp_b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 5: All three operations (add, sub, mul) get annotated correctly
// CHECK-LABEL: func.func @all_operations_fpu
func.func @all_operations_fpu(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                               %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Test add with FPU
  // CHECK: ttl.tile_add{{.*}}execution_target = "fpu"
  %sum = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %sum : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 6: Sub operation with FPU
// CHECK-LABEL: func.func @sub_fpu
func.func @sub_fpu(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                    %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sub{{.*}}execution_target = "fpu"
  %diff = ttl.sub %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %diff : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 7: Mul operation with FPU
// CHECK-LABEL: func.func @mul_fpu
func.func @mul_fpu(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                    %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_mul{{.*}}execution_target = "fpu"
  %prod = ttl.mul %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %prod : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 8: Complex fused operation - multiple strategies in same function
// Demonstrates all three strategies can coexist in fused operations.
// CHECK-LABEL: func.func @mixed_strategies
func.func @mixed_strategies(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %arg2: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c = ttl.attach_cb %arg2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Compute: (a + b) * exp(c)
  %sum = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %exp_c = ttl.exp %c : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Verify all tile operations
  // CHECK: ttl.compute
  // First add: SHOULD be annotated "fpu" (both block args)
  // CHECK: ttl.tile_add{{.*}}execution_target = "fpu"
  // Exp: should NOT be annotated (unary)
  // CHECK: ttl.tile_exp
  // CHECK-NOT: execution_target
  // Second mul: SHOULD be annotated "sfpu" (both from tile ops)
  // CHECK: ttl.tile_mul{{.*}}execution_target = "sfpu"
  %result = ttl.mul %sum, %exp_c : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test 9: Chained FPU operations via dest-reuse
// Demonstrates that chained binary ops use FPU hardware throughout (FPU + dest_reuse).
// This is a key optimization - multiple binary operations can chain via FPU!
// CHECK-LABEL: func.func @chained_fpu_via_dest_reuse
func.func @chained_fpu_via_dest_reuse(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                                       %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>,
                                       %arg2: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c = ttl.attach_cb %arg2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Compute: (a + b) + c
  // If fused, this demonstrates FPU chaining:
  // ^bb0(%arg3, %arg4, %arg5, %out):
  //   %sum1 = ttl.tile_add %arg3, %arg4  // Both block args -> "fpu" (FPU hardware)
  //   %sum2 = ttl.tile_add %sum1, %arg5  // LHS=DST, RHS=block arg -> "dest_reuse" (FPU hardware!)
  //
  // Both operations use FPU hardware! This enables efficient operation chaining.
  %sum1 = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // First add: both block args -> FPU
  // CHECK: ttl.tile_add{{.*}}execution_target = "fpu"
  // Second add: result of first add (DST) + block arg -> dest_reuse (also FPU!)
  // CHECK: ttl.tile_add{{.*}}execution_target = "dest_reuse"
  %sum2 = ttl.add %sum1, %c : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %sum2 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
