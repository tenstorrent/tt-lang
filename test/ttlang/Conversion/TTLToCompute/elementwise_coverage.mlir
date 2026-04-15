// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst{enable-fpu-binary-ops=0}),cse,canonicalize)' | FileCheck %s

// Note: enable-fpu-binary-ops=0 keeps SFPU lowering path (not testing FPU detection).
// Test: Binary elementwise operations lower to ttl.compute with tile ops
// Input provides explicit bind_cb and attach_cb ops.
// This test verifies the full CB attachment pattern for all arguments.

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @binary_add(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-DAG: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-DAG: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB2]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK:        %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[IN0]]
  // CHECK:        %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[IN1]]
  // CHECK-NEXT:   %[[ADD:.+]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary elementwise operations (SFPU)

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[EXP:.+]] = ttl.tile_exp %[[DTILE]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary exp2 (base-2 exponential) lowers to ttl.compute with tile_exp2

// CHECK-LABEL: func.func @unary_exp2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_exp2(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_exp2 %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp2 %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary ceil lowers to ttl.compute with tile_ceil

// CHECK-LABEL: func.func @unary_ceil
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_ceil(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_ceil %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.ceil %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary sign lowers to ttl.compute with tile_sign

// CHECK-LABEL: func.func @unary_sign
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_sign(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_sign %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.sign %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary gelu lowers to ttl.compute with tile_gelu

// CHECK-LABEL: func.func @unary_gelu
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_gelu(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_gelu %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.gelu %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary silu lowers to ttl.compute with tile_silu

// CHECK-LABEL: func.func @unary_silu
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_silu(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_silu %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.silu %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary hardsigmoid lowers to ttl.compute with tile_hardsigmoid

// CHECK-LABEL: func.func @unary_hardsigmoid
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_hardsigmoid(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_hardsigmoid %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.hardsigmoid %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary expm1 lowers to ttl.compute with tile_expm1

// CHECK-LABEL: func.func @unary_expm1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_expm1(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_expm1 %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.expm1 %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary square lowers to ttl.compute with tile_square

// CHECK-LABEL: func.func @unary_square
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_square(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_square %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.square %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary softsign lowers to ttl.compute with tile_softsign

// CHECK-LABEL: func.func @unary_softsign
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_softsign(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_softsign %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.softsign %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary signbit lowers to ttl.compute with tile_signbit

// CHECK-LABEL: func.func @unary_signbit
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_signbit(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_signbit %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.signbit %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary frac lowers to ttl.compute with tile_frac (shared rounding init)

// CHECK-LABEL: func.func @unary_frac
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_frac(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_frac %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.frac %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary trunc lowers to ttl.compute with tile_trunc (shared rounding init)

// CHECK-LABEL: func.func @unary_trunc
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_trunc(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK:      %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[RES:.+]] = ttl.tile_trunc %[[DTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[RES]], %{{.*}}[%[[I0]], %[[I1]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.trunc %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Chained elementwise operations produce multiple ttl.compute ops

// CHECK-LABEL: func.func @chained_ops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @chained_ops(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[CB3:.+]] = ttl.bind_cb{cb_index = 3
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EMPTY_CB:.+]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: binary add
  // CHECK:      %[[ADD:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : {{.*}}) outs(%[[EMPTY_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT0:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK:        %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[IN0]]
  // CHECK:        %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[IN1]]
  // CHECK-NEXT:   %[[TILE_ADD:.+]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %r0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %r0 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: %[[ADD_CB:.+]] = ttl.attach_cb %[[ADD]], %[[CB2]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary relu
  // CHECK:      %[[EMPTY_CB2:.+]] = ttl.attach_cb %[[EMPTY]], %[[CB3]]
  // CHECK:      %[[RESULT:.+]] = ttl.compute ins(%[[ADD_CB]] : {{.*}}) outs(%[[EMPTY_CB2]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN2:.+]]: !ttcore.tile<32x32, f32>, %[[OUT1:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[I0B:.*]] = ttl.iter_index 0 : index
  // CHECK-NEXT:   %[[I1B:.*]] = ttl.iter_index 1 : index
  // CHECK-NEXT:   %[[DTOK2:.*]], %[[DTILE2:.*]] = ttl.copy_tile %[[IN2]][%[[I0B]], %[[I1B]]] into dst[%[[C0]]]
  // CHECK-NEXT:   %[[TILE_RELU:.+]] = ttl.tile_relu %[[DTILE2]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.tile_store %[[TILE_RELU]], %{{.*}}[%[[I0B]], %[[I1B]]] from dst[%[[C0]]]
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %r1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %1 = ttl.relu %add_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %1, %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All binary operations

// CHECK-LABEL: func.func @all_binary_ops
func.func @all_binary_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_store
  %r0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %r0 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %add_cb = ttl.attach_cb %add, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb2 = ttl.attach_cb %b, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sub
  // CHECK: ttl.tile_store
  %r1 = ttl.cb_reserve %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sub = ttl.sub %add_cb, %b_cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sub, %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %sub_cb = ttl.attach_cb %sub, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_cb2 = ttl.attach_cb %a, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_mul
  // CHECK: ttl.tile_store
  %r2 = ttl.cb_reserve %cb6 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %mul = ttl.mul %sub_cb, %a_cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %mul, %r2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %mul_cb = ttl.attach_cb %mul, %cb6 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb3 = ttl.attach_cb %b, %cb7 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_max
  // CHECK: ttl.tile_store
  %r3 = ttl.cb_reserve %cb8 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %max = ttl.max %mul_cb, %b_cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %max, %r3 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %max : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All unary operations

// CHECK-LABEL: func.func @all_unary_ops
func.func @all_unary_ops(%x: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb9 = ttl.bind_cb {cb_index = 9, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb10 = ttl.bind_cb {cb_index = 10, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb11 = ttl.bind_cb {cb_index = 11, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb12 = ttl.bind_cb {cb_index = 12, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb13 = ttl.bind_cb {cb_index = 13, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb14 = ttl.bind_cb {cb_index = 14, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb15 = ttl.bind_cb {cb_index = 15, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb16 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb17 = ttl.bind_cb {cb_index = 17, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb18 = ttl.bind_cb {cb_index = 18, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb19 = ttl.bind_cb {cb_index = 19, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb20 = ttl.bind_cb {cb_index = 20, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb21 = ttl.bind_cb {cb_index = 21, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %x_cb = ttl.attach_cb %x, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_exp %
  // CHECK: ttl.tile_store
  %r0 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %exp = ttl.exp %x_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %exp, %r0 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %exp_cb = ttl.attach_cb %exp, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_log
  // CHECK: ttl.tile_store
  %r1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %log = ttl.log %exp_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %log, %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %log_cb = ttl.attach_cb %log, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sqrt
  // CHECK: ttl.tile_store
  %r2 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sqrt = ttl.sqrt %log_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sqrt, %r2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %sqrt_cb = ttl.attach_cb %sqrt, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_rsqrt
  // CHECK: ttl.tile_store
  %r3 = ttl.cb_reserve %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %rsqrt = ttl.rsqrt %sqrt_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %rsqrt, %r3 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %rsqrt_cb = ttl.attach_cb %rsqrt, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_tanh
  // CHECK: ttl.tile_store
  %r4 = ttl.cb_reserve %cb5 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %tanh = ttl.tanh %rsqrt_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %tanh, %r4 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %tanh_cb = ttl.attach_cb %tanh, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sigmoid
  // CHECK: ttl.tile_store
  %r5 = ttl.cb_reserve %cb6 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sigmoid = ttl.sigmoid %tanh_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sigmoid, %r5 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %sigmoid_cb = ttl.attach_cb %sigmoid, %cb6 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_neg
  // CHECK: ttl.tile_store
  %r6 = ttl.cb_reserve %cb7 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %neg = ttl.neg %sigmoid_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %neg, %r6 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %neg_cb = ttl.attach_cb %neg, %cb7 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_abs
  // CHECK: ttl.tile_store
  %r7 = ttl.cb_reserve %cb8 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %abs = ttl.abs %neg_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %abs, %r7 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %abs_cb = ttl.attach_cb %abs, %cb8 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_relu
  // CHECK: ttl.tile_store
  %r8 = ttl.cb_reserve %cb9 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %relu = ttl.relu %abs_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %relu, %r8 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %relu_cb = ttl.attach_cb %relu, %cb9 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_exp2
  // CHECK: ttl.tile_store
  %r9 = ttl.cb_reserve %cb10 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %exp2 = ttl.exp2 %relu_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %exp2, %r9 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %exp2_cb = ttl.attach_cb %exp2, %cb10 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_ceil
  // CHECK: ttl.tile_store
  %r10 = ttl.cb_reserve %cb11 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %ceil = ttl.ceil %exp2_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %ceil, %r10 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %ceil_cb = ttl.attach_cb %ceil, %cb11 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sign
  // CHECK: ttl.tile_store
  %r11 = ttl.cb_reserve %cb12 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sign = ttl.sign %ceil_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sign, %r11 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %sign_cb = ttl.attach_cb %sign, %cb12 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_gelu
  // CHECK: ttl.tile_store
  %r12 = ttl.cb_reserve %cb13 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %gelu = ttl.gelu %sign_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %gelu, %r12 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %gelu_cb = ttl.attach_cb %gelu, %cb13 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_silu
  // CHECK: ttl.tile_store
  %r13 = ttl.cb_reserve %cb14 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %silu = ttl.silu %gelu_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %silu, %r13 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %silu_cb = ttl.attach_cb %silu, %cb14 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_hardsigmoid
  // CHECK: ttl.tile_store
  %r14 = ttl.cb_reserve %cb15 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %hardsigmoid = ttl.hardsigmoid %silu_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %hardsigmoid, %r14 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %hardsigmoid_cb = ttl.attach_cb %hardsigmoid, %cb15 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_expm1
  // CHECK: ttl.tile_store
  %r15 = ttl.cb_reserve %cb16 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %expm1 = ttl.expm1 %hardsigmoid_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %expm1, %r15 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %expm1_cb = ttl.attach_cb %expm1, %cb16 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_square
  // CHECK: ttl.tile_store
  %r16 = ttl.cb_reserve %cb17 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %square = ttl.square %expm1_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %square, %r16 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %square_cb = ttl.attach_cb %square, %cb17 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_softsign
  // CHECK: ttl.tile_store
  %r17 = ttl.cb_reserve %cb18 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %softsign = ttl.softsign %square_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %softsign, %r17 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %softsign_cb = ttl.attach_cb %softsign, %cb18 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_signbit
  // CHECK: ttl.tile_store
  %r18 = ttl.cb_reserve %cb19 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %signbit = ttl.signbit %softsign_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %signbit, %r18 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %signbit_cb = ttl.attach_cb %signbit, %cb19 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_frac
  // CHECK: ttl.tile_store
  %r19 = ttl.cb_reserve %cb20 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %frac = ttl.frac %signbit_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %frac, %r19 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  %frac_cb = ttl.attach_cb %frac, %cb20 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_trunc
  // CHECK: ttl.tile_store
  %r20 = ttl.cb_reserve %cb21 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %trunc = ttl.trunc %frac_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %trunc, %r20 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %trunc : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: DST assignment on chain of binary and unary ops

// CHECK-LABEL: func.func @dst_assignment_chain
// CHECK-SAME:  (%[[A:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[C:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @dst_assignment_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>, %c: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-DAG: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-DAG: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-DAG: %[[CB3:.+]] = ttl.bind_cb{cb_index = 3
  // CHECK-DAG: %[[CB4:.+]] = ttl.bind_cb{cb_index = 4
  // CHECK-DAG: %[[CB5:.+]] = ttl.bind_cb{cb_index = 5
  // CHECK: %[[A_CB:.+]] = ttl.attach_cb %[[A:arg0]], %[[CB0]]
  // CHECK: %[[B_CB:.+]] = ttl.attach_cb %[[B:arg1]], %[[CB1]]
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: binary add
  // CHECK: %[[COMP0:.+]] = ttl.compute
  // CHECK: ttl.tile_add {{.*}} into dst[%c0]
  // CHECK: ttl.tile_store
  %r0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %r0 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[COMP0_CB:.+]] = ttl.attach_cb %[[COMP0]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary exp
  // CHECK: %[[COMP1:.+]] = ttl.compute
  // CHECK: ttl.tile_exp {{.*}} into dst[%c0]
  // CHECK: ttl.tile_store
  %r1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %1 = ttl.exp %add_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %1, %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[COMP1_CB:.+]] = ttl.attach_cb %[[COMP1]]
  // CHECK: %[[C_CB:.+]] = ttl.attach_cb %[[C]]
  %exp_cb = ttl.attach_cb %1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Third compute: binary mul
  // CHECK: %[[COMP2:.+]] = ttl.compute
  // CHECK: ttl.tile_mul {{.*}} into dst[%c0]
  // CHECK: ttl.tile_store
  // CHECK: return %[[COMP2]]
  %r2 = ttl.cb_reserve %cb5 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %2 = ttl.mul %exp_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %2, %r2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %2 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
