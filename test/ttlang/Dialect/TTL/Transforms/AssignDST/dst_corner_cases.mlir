// Summary: Corner case tests for DST allocation edge cases not covered elsewhere.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}))' --split-input-file | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder
// SEPARATE-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Block arg with 3 consumers - 2 unary, 1 binary
// =============================================================================
// Purpose: Verify copy insertion handles more than 2 consumers correctly.
// Pattern:
//   sigmoid(a)  - unary consumer #1
//   exp(a)      - unary consumer #2
//   add(a, b)   - binary consumer #3
//
// Expected: Each unary consumer gets its own copy_tile, binary gets a copy too.

// CHECK-LABEL: func.func @block_arg_three_consumers
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// Two copies of A are scheduled first, then sigmoid uses one of them
// CHECK:       %{{.*}}, %[[ACOPY0:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NEXT:  %{{.*}}, %[[ACOPY1:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[A]]
// CHECK-NEXT:  %[[SIG:.*]] = ttl.tile_sigmoid %[[ACOPY1]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// Third copy of A for exp
// CHECK:       %{{.*}}, %[[ACOPY2:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[A]]
// CHECK-NEXT:  %[[EXP:.*]] = ttl.tile_exp %[[ACOPY2]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// Copy B for add
// CHECK:       %{{.*}}, %[[BCOPY:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[ADD:.*]] = ttl.tile_add %[[ACOPY0]], %[[BCOPY]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[SIG]], %[[EXP]], %[[ADD]]

func.func @block_arg_three_consumers(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 18, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result:3 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb, %init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                             tensor<2x2x!ttcore.tile<32x32, f32>>,
                                             tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>,
       %out2_tile: !ttcore.tile<32x32, f32>):
    // Block arg 'a' has 3 consumers: sigmoid, exp, add
    %sig = ttl.tile_sigmoid %a_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sig, %exp, %add : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1, %result#2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Block arg yielded directly without transformation
// =============================================================================
// Purpose: Verify pass-through of block arg to yield works correctly.
// The block arg still needs to be copied to DST before being yielded.

// CHECK-LABEL: func.func @block_arg_passthrough
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// Single copy_tile for the passthrough block arg
// CHECK:       %{{.*}}, %[[TILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  ttl.yield %[[TILE]]

func.func @block_arg_passthrough(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Just yield the input directly - no transformation
    ttl.yield %a_tile : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 3: Single input producing multiple distinct outputs
// =============================================================================
// Purpose: Verify one input can produce multiple outputs through different ops.

// CHECK-LABEL: func.func @single_input_multiple_outputs
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>,
// Two copies of A for two unary consumers
// CHECK:       %{{.*}}, %[[TILE0:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NEXT:  %{{.*}}, %[[TILE1:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[EXP:.*]] = ttl.tile_exp %[[TILE1]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[SIG:.*]] = ttl.tile_sigmoid %[[TILE0]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[EXP]], %[[SIG]]

func.func @single_input_multiple_outputs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result:2 = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>):
    // Single input produces two different outputs
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    %sig = ttl.tile_sigmoid %a_tile : !ttcore.tile<32x32, f32>
    ttl.yield %exp, %sig : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 4: Unary chain that branches mid-way
// =============================================================================
// Purpose: Verify correct handling when a unary chain result is used by
// both another unary AND a binary operation.
// Pattern:
//   %0 = abs(a)       - unary
//   %1 = exp(%0)      - unary, %0 used here
//   %2 = add(%0, b)   - binary, %0 also used here
//   yield %1, %2
//
// %0 has two consumers: one unary (exp) and one binary (add).
// This tests the interaction between unary merging and multi-consumer handling.

// CHECK-LABEL: func.func @unary_chain_with_branch
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// Copy A for abs
// CHECK:       %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[A]]
// CHECK-NEXT:  %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// ABS result has unary consumer (exp), so copy_dst is inserted
// CHECK-NEXT:  %[[ABSCOPY:.*]] = ttl.copy_dst %[[ABS]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[EXP:.*]] = ttl.tile_exp %[[ABSCOPY]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// Copy B for add
// CHECK:       %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[ADD:.*]] = ttl.tile_add %[[ABS]], %[[BTILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[EXP]], %[[ADD]]

func.func @unary_chain_with_branch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result:2 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>):
    // Unary chain that branches
    %abs = ttl.tile_abs %a_tile : !ttcore.tile<32x32, f32>
    // %abs has two consumers: exp (unary) and add (binary)
    %exp = ttl.tile_exp %abs : !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %abs, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %exp, %add : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 5: Deep unary chain followed by binary (tests merging limits)
// =============================================================================
// Purpose: Verify that a long unary chain all merges correctly, then
// the final binary op gets its own register.

// CHECK-LABEL: func.func @deep_unary_then_binary
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// Copy A for the unary chain
// CHECK:       %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[A]]
// All unary ops share DST register 0 (merged interval)
// CHECK-NEXT:  %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[EXP:.*]] = ttl.tile_exp %[[ABS]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[RELU:.*]] = ttl.tile_relu %[[EXP]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[SIG:.*]] = ttl.tile_sigmoid %[[RELU]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// Copy B for binary op
// CHECK:       %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[ADD:.*]] = ttl.tile_add %[[SIG]], %[[BTILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[ADD]]

func.func @deep_unary_then_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Deep unary chain - all should merge
    %abs = ttl.tile_abs %a_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %abs : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp : !ttcore.tile<32x32, f32>
    %sig = ttl.tile_sigmoid %relu : !ttcore.tile<32x32, f32>
    // Then binary op at the end
    %add = ttl.tile_add %sig, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 6: Accumulation pattern (value used as both operands of binary op)
// =============================================================================
// Purpose: Test when a value is used as BOTH operands of a binary operation.
// Pattern: x * x (square)

// CHECK-LABEL: func.func @square_pattern
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>,
// Single copy_tile for A which is used as both operands
// CHECK:       %{{.*}}, %[[TILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// Same tile used as both operands to mul
// CHECK-NEXT:  %[[SQ:.*]] = ttl.tile_mul %[[TILE]], %[[TILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[SQ]]

func.func @square_pattern(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // x * x pattern (square) - same value used as both operands
    %sq = ttl.tile_mul %a_tile, %a_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sq : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 7: Mixed chain with intermediate value reused much later
// =============================================================================
// Purpose: Test register pressure when an early value is needed much later.
// Pattern:
//   %0 = add(a, b)
//   %1 = mul(%0, c)
//   %2 = exp(%1)
//   %3 = add(%2, %0)  <- %0 reused here, must still be live
//
// This tests that the live interval for %0 extends to its last use.

// CHECK-LABEL: func.func @intermediate_reuse_late
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[C:[^:]*]]: !ttcore.tile<32x32, f32>,
// Copy A
// CHECK:       %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} :
// Copy B
// CHECK:       %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[A]]
// CHECK-NOT:   ttl.copy_tile %[[B]]
// CHECK-NEXT:  %[[ADD0:.*]] = ttl.tile_add %[[ATILE]], %[[BTILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// Copy C
// CHECK:       %{{.*}}, %[[CTILE:.*]] = ttl.copy_tile %[[C]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[MUL:.*]] = ttl.tile_mul %[[ADD0]], %[[CTILE]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[EXP:.*]] = ttl.tile_exp %[[MUL]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// ADD0 is reused here - it was kept live across mul and exp
// CHECK-NEXT:  %[[ADD1:.*]] = ttl.tile_add %[[EXP]], %[[ADD0]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[ADD1]]

func.func @intermediate_reuse_late(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %add0 = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %add0, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    // Reuse add0 much later - it must be kept live through mul and exp
    %add1 = ttl.tile_add %exp, %add0 : !ttcore.tile<32x32, f32>
    ttl.yield %add1 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 8: Block arg used by one unary op and two binary ops
// =============================================================================
// Pattern:
//   %abs = tile_abs %x    // Unary - overwrites x in-place
//   %add = tile_add %x, %y // Binary uses same x
//   %mul = tile_mul %x, %y // Binary uses same x
//
// The unary op (abs) would clobber x before the binary ops can use it.
// Phase 1 inserts copy_tile only for abs (last unary consumer).
// The binary consumers (add, mul) share a single copy created later.
// Result: 2 copies of x total (one for abs, one shared by add/mul).

// CHECK-LABEL: func.func @unary_and_binary_consumers
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[X:[^:]*]]: !ttcore.tile<32x32, f32>, %[[Y:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// Three copy_tile ops for X (for mul, abs, and add)
// CHECK:       %{{.*}}, %[[XCOPY0:.*]] = ttl.copy_tile %[[X]], %{{.*}}, %{{.*}} :
// CHECK-NEXT:  %{{.*}}, %[[XCOPY1:.*]] = ttl.copy_tile %[[X]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile %[[X]]
// CHECK-NEXT:  %[[ABS:.*]] = ttl.tile_abs %[[XCOPY1]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// Third copy of X for add
// CHECK:       %{{.*}}, %[[XCOPY2:.*]] = ttl.copy_tile %[[X]], %{{.*}}, %{{.*}} :
// Copy Y for binary ops
// CHECK:       %{{.*}}, %[[YCOPY:.*]] = ttl.copy_tile %[[Y]], %{{.*}}, %{{.*}} :
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT:  %[[ADD:.*]] = ttl.tile_add %[[XCOPY2]], %[[YCOPY]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[MUL:.*]] = ttl.tile_mul %[[XCOPY0]], %[[YCOPY]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// Combine results
// CHECK-NEXT:  %[[TMP:.*]] = ttl.tile_add %[[ABS]], %[[ADD]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  %[[RESULT:.*]] = ttl.tile_add %[[TMP]], %[[MUL]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:  ttl.yield %[[RESULT]]

func.func @unary_and_binary_consumers(%i0: tensor<32x32xf32>,
                                       %i1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb_out = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb0 : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb1 : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb_out : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):
    // x is used by abs (unary), add (binary), and mul (binary)
    // Phase 1 inserts copy_tile for abs and add; mul uses original
    %abs = ttl.tile_abs %x : !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %x, %y : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %x, %y : !ttcore.tile<32x32, f32>
    // Combine results
    %tmp = ttl.tile_add %abs, %add : !ttcore.tile<32x32, f32>
    %result = ttl.tile_add %tmp, %mul : !ttcore.tile<32x32, f32>
    ttl.yield %result : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}
