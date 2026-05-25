// Summary: Corner case tests for DST allocation edge cases not covered elsewhere.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{enable-fpu-binary-ops=1 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst{dst-capacity=8}), canonicalize, cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{enable-fpu-binary-ops=1 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst{dst-capacity=8 separate-output-region=1}), canonicalize, cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder
// SEPARATE-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Block arg with multiple consumers split across FPU and SFPU paths
// =============================================================================
// Purpose: Verify copy insertion handles more than 2 SFPU consumers correctly
// when the same source tensor also feeds an FPU binary op. Because the
// SetComputeKernelConfig pass rejects an f32 CB consumed by both FPU and SFPU
// strategies, the input `a` is attached to two CBs: CB0 for the FPU consumer
// and CB2 for the SFPU consumers. The SFPU view (`a_sfpu`) still has 2
// consumers in the body, so DST allocation must give each its own copy_tile.
// Pattern:
//   sigmoid(a_sfpu)  - SFPU unary consumer #1
//   exp(a_sfpu)      - SFPU unary consumer #2
//   add(a_fpu, b)    - FPU binary consumer (reads from CB, no copy)

// CHECK-LABEL: func.func @block_arg_three_consumers
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A_FPU:[^:]*]]: !ttcore.tile<32x32, f32>, %[[A_SFPU:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// iter_index ops provide iteration coordinates for CB indexing.
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// Copy a_sfpu for sigmoid (SFPU unary needs DST). Exp also gets its own copy.
// Add is FPU binary (a_fpu and b are block args read from CB, no copies).
// CHECK:           %{{.*}}, %[[ACOPY1:.*]] = ttl.copy_tile %[[A_SFPU]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[SIG:.*]] = ttl.tile_sigmoid %[[ACOPY1]] into dst[%[[C0]]]
// CHECK:           %{{.*}}, %[[ACOPY2:.*]] = ttl.copy_tile %[[A_SFPU]][%[[I0]], %[[I1]]] into dst[%[[C1]]]
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[ACOPY2]] into dst[%[[C1]]]
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[A_FPU]], %[[B]] into dst[%[[C2]]]
// CHECK:           ttl.tile_store %[[SIG]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @block_arg_three_consumers(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  // CB0 feeds the FPU consumer; CB2 mirrors `a` for the SFPU consumers so the
  // f32 unpack mode never has to satisfy both strategies on the same CB.
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb_a_sfpu = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 18, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb_fpu = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_cb_sfpu = ttl.attach_cb %a, %cb_a_sfpu : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_2 = ttl.cb_reserve %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result:3 = ttl.compute
      ins(%a_cb_fpu, %a_cb_sfpu, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                         tensor<2x2x!ttcore.tile<32x32, f32>>,
                                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb, %init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                             tensor<2x2x!ttcore.tile<32x32, f32>>,
                                             tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_fpu_tile: !ttcore.tile<32x32, f32>,
       %a_sfpu_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>,
       %out2_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // %a_sfpu_tile has two SFPU consumers (sigmoid, exp); each needs its own
    // copy_tile. %a_fpu_tile feeds the FPU add directly from CB0.
    %c0 = arith.constant 0 : index
    %sig = ttl.tile_sigmoid %a_sfpu_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %a_sfpu_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %a_fpu_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sig, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %exp, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %add, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// Single copy_tile for the passthrough block arg
// CHECK:           %{{.*}}, %[[TILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%c0]
// CHECK-NOT:       ttl.copy_tile
// CHECK:           ttl.tile_store %[[TILE]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @block_arg_passthrough(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_0 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Just yield the input directly - no transformation
    %c0 = arith.constant 0 : index
    ttl.tile_store %a_tile, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// Both exp and sigmoid get their own copy_tile
// CHECK:           %{{.*}}, %[[TILE_EXP:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[TILE_EXP]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           %{{.*}}, %[[TILE_SIG:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C1]]]
// CHECK:      %[[SIG:.*]] = ttl.tile_sigmoid %[[TILE_SIG]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[SIG]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @single_input_multiple_outputs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 17, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_1 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_2 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result:2 = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Single input produces two different outputs
    %c0 = arith.constant 0 : index
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sig = ttl.tile_sigmoid %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %sig, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// Copy A for abs (at first use)
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// ABS result has unary consumer (exp), so copy_dst is inserted
// CHECK:      %[[ABSCOPY:.*]] = ttl.copy_dst %[[ABS]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[ABSCOPY]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// B copied at first use (tile_add)
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c2]
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[ABS]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @unary_chain_with_branch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_2 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_3 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Unary chain that branches
    %c0 = arith.constant 0 : index
    %abs = ttl.tile_abs %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // %abs has two consumers: exp (unary) and add (binary)
    %exp = ttl.tile_exp %abs into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %abs, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %add, %out_view_3[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// Copy A for the unary chain (at first use)
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// All unary ops share DST register 0 (merged interval)
// CHECK:      %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[ABS]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[RELU:.*]] = ttl.tile_relu %[[EXP]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[SIG:.*]] = ttl.tile_sigmoid %[[RELU]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// B copied at first use (tile_add)
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c1]
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[SIG]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @deep_unary_then_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_3 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Deep unary chain - all should merge
    %c0 = arith.constant 0 : index
    %abs = ttl.tile_abs %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %abs into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sig = ttl.tile_sigmoid %relu into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // Then binary op at the end
    %add = ttl.tile_add %sig, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_3[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary: both operands are the same block arg, no copy needed.
// CHECK-NOT:       ttl.copy_tile
// CHECK:           %[[SQ:.*]] = ttl.tile_mul %[[A]], %[[A]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[SQ]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @square_pattern(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_4 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // x * x pattern (square) - same value used as both operands
    %c0 = arith.constant 0 : index
    %sq = ttl.tile_mul %a_tile, %a_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sq, %out_view_4[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[C:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// First add is FPU binary (no copies for A, B). C copied for SFPU mul.
// CHECK:           %[[ADD0:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           %{{.*}}, %[[CTILE:.*]] = ttl.copy_tile %[[C]][%[[I0]], %[[I1]]] into dst[%[[C1]]]
// CHECK:      %[[MUL:.*]] = ttl.tile_mul %[[ADD0]], %[[CTILE]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[MUL]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// ADD0 is reused here - it was kept live across mul and exp
// CHECK:      %[[ADD1:.*]] = ttl.tile_add %[[EXP]], %[[ADD0]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD1]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @intermediate_reuse_late(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view_5 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %add0 = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %add0, %c_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // Reuse add0 much later - it must be kept live through mul and exp
    %add1 = ttl.tile_add %exp, %add0 into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add1, %out_view_5[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 8: Tensor used by one unary op and two binary ops via split CBs
// =============================================================================
// Pattern:
//   %abs = tile_abs %x_sfpu      // SFPU unary, reads from the SFPU-side CB
//   %add = tile_add %x_fpu, %y   // FPU binary, reads from the FPU-side CB
//   %mul = tile_mul %x_fpu, %y   // FPU binary, reads from the FPU-side CB
//
// The same source tensor %i0 feeds both an SFPU unary and two FPU binaries.
// SetComputeKernelConfig forbids an f32 CB from being consumed by both
// strategies, so %i0 is attached to two CBs: CB0 for the FPU consumers and
// CB2 for the SFPU consumer. Phase 1 still inserts copy_tile for abs because
// the unary op overwrites its DST register, while add and mul stay FPU
// binary (both operands are block args) and read directly from their CBs.
// Result: 1 copy of x_sfpu total (for abs only).

// CHECK-LABEL: func.func @unary_and_binary_consumers
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[X_FPU:[^:]*]]: !ttcore.tile<32x32, f32>, %[[X_SFPU:[^:]*]]: !ttcore.tile<32x32, f32>, %[[Y:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:      %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// Copy x_sfpu for abs (SFPU unary consumer needs DST)
// CHECK:           %{{.*}}, %[[XCOPY_ABS:.*]] = ttl.copy_tile %[[X_SFPU]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[ABS:.*]] = ttl.tile_abs %[[XCOPY_ABS]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// Both add and mul are FPU binary on x_fpu/y (both block args)
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[X_FPU]], %[[Y]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[MUL:.*]] = ttl.tile_mul %[[X_FPU]], %[[Y]] into dst[%[[C2]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// Combine results (SFPU, operands from DST)
// CHECK:      %[[TMP:.*]] = ttl.tile_add %[[ABS]], %[[ADD]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[RESULT:.*]] = ttl.tile_add %[[TMP]], %[[MUL]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[RESULT]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

func.func @unary_and_binary_consumers(%i0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                                       %i1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  // CB0 feeds the FPU consumers; CB2 mirrors %i0 for the SFPU consumer.
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb_x_sfpu = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb_out = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %t0_fpu = ttl.attach_cb %i0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %t0_sfpu = ttl.attach_cb %i0, %cb_x_sfpu : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %t1 = ttl.attach_cb %i1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %t_init = ttl.attach_cb %init, %cb_out : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view_6 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
    ins(%t0_fpu, %t0_sfpu, %t1 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
    outs(%t_init : tensor<1x1x!ttcore.tile<32x32, f32>>)
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x_fpu: !ttcore.tile<32x32, f32>, %x_sfpu: !ttcore.tile<32x32, f32>,
       %y: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // x_sfpu feeds abs (SFPU), x_fpu feeds add and mul (FPU). Phase 1 inserts
    // copy_tile only for abs; add and mul read from CB.
    %c0 = arith.constant 0 : index
    %abs = ttl.tile_abs %x_sfpu into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %x_fpu, %y into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %x_fpu, %y into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // Combine results
    %tmp = ttl.tile_add %abs, %add into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %result = ttl.tile_add %tmp, %mul into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %result, %out_view_6[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
