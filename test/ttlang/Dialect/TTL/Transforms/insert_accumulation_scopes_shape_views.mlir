// Singleton-dimension views retain the seed store when selecting the initial
// mode for DFB accumulation, including a same-guard conditional acquisition.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-accumulation-scopes{kind=dfb}))' --split-input-file | FileCheck %s

// A collapsed output aliases a seed written through the original reserve.
// CHECK-LABEL: func.func @collapsed_seed_view
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: %[[VIEW:.*]] = tensor.collapse_shape %[[RESERVE]]
// CHECK-NEXT: ttl.store %arg0, %[[RESERVE]]
// CHECK-NEXT: ttl.accumulation_scope outs(%[[VIEW]]
// CHECK: scf.for
// CHECK-NEXT: ttl.store %arg1, %[[VIEW]] {accumulate}
// CHECK: } initial_modes([accumulate_existing])
func.func @collapsed_seed_view(
    %seed: tensor<1x2x3x!ttcore.tile<32x32, bf16>>,
    %increment: tensor<2x3x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb
      : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  %view = tensor.collapse_shape %reserve [[0, 1], [2]]
      : tensor<1x2x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %seed, %reserve
      : tensor<1x2x3x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %increment, %view {accumulate}
        : tensor<2x3x!ttcore.tile<32x32, bf16>>,
          tensor<2x3x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// An expanded output retains storage identity through an attached reserve.
// CHECK-LABEL: func.func @expanded_attached_seed_view
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: %[[ATTACHED:.*]] = ttl.attach_cb %[[RESERVE]]
// CHECK-NEXT: %[[VIEW:.*]] = tensor.expand_shape %[[ATTACHED]]
// CHECK-NEXT: ttl.store %arg0, %[[RESERVE]]
// CHECK-NEXT: ttl.accumulation_scope outs(%[[VIEW]]
// CHECK: scf.for
// CHECK-NEXT: ttl.store %arg1, %[[VIEW]] {accumulate}
// CHECK: } initial_modes([accumulate_existing])
func.func @expanded_attached_seed_view(
    %seed: tensor<2x3x!ttcore.tile<32x32, f32>>,
    %increment: tensor<2x1x3x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %reserve = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, f32>, 2>
      -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %attached = ttl.attach_cb %reserve, %cb
      : (tensor<2x3x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %view = tensor.expand_shape %attached [[0], [1, 2]] output_shape [2, 1, 3]
      : tensor<2x3x!ttcore.tile<32x32, f32>>
        into tensor<2x1x3x!ttcore.tile<32x32, f32>>
  ttl.store %seed, %reserve : tensor<2x3x!ttcore.tile<32x32, f32>>,
      tensor<2x3x!ttcore.tile<32x32, f32>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %increment, %view {accumulate}
        : tensor<2x1x3x!ttcore.tile<32x32, f32>>,
          tensor<2x1x3x!ttcore.tile<32x32, f32>>
  }
  return
}

// -----

// A shape view of a conditional result retains its same-guard initialization.
// CHECK-LABEL: func.func @guarded_expanded_seed_view
// CHECK: %[[GUARDED:.*]] = scf.if %arg0
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %arg1, %[[RESERVE]]
// CHECK: scf.yield %[[RESERVE]]
// CHECK: %[[VIEW:.*]] = tensor.expand_shape %[[GUARDED]]
// CHECK-NEXT: ttl.accumulation_scope outs(%[[VIEW]]
// CHECK: scf.for
// CHECK-NEXT: scf.if %arg0
// CHECK-NEXT: ttl.store %arg2, %[[VIEW]] {accumulate}
// CHECK: } initial_modes([accumulate_existing])
func.func @guarded_expanded_seed_view(
    %condition: i1,
    %seed: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %increment: tensor<1x2x3x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %guarded = scf.if %condition -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
    %reserve = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    ttl.store %seed, %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<2x3x!ttcore.tile<32x32, bf16>>
    scf.yield %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>
  } else {
    %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb}
        : () -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    scf.yield %inactive : tensor<2x3x!ttcore.tile<32x32, bf16>>
  }
  %view = tensor.expand_shape %guarded [[0, 1], [2]] output_shape [1, 2, 3]
      : tensor<2x3x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    scf.if %condition {
      ttl.store %increment, %view {accumulate}
          : tensor<1x2x3x!ttcore.tile<32x32, bf16>>,
            tensor<1x2x3x!ttcore.tile<32x32, bf16>>
    }
  }
  return
}
