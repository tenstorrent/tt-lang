// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes, ttl-assign-dst))' | FileCheck %s --check-prefix=ASSIGN

// Summary: A multi-tile additive recurrence assigns one DST slot per output tile
// and keeps that slot consistent across the initial copy, the per-iteration
// accumulate, and the final store after ttl-assign-dst.

// ASSIGN-LABEL: func.func @multi_tile_recurrence
// ASSIGN: ttl.dst_section
// Four output tiles initialize four distinct DST slots.
// ASSIGN: %[[S0:.*]] = arith.constant 0 : index
// ASSIGN: ttl.copy_tile {{.*}} into dst[%[[S0]]]
// ASSIGN: %[[S1:.*]] = arith.constant 1 : index
// ASSIGN: ttl.copy_tile {{.*}} into dst[%[[S1]]]
// ASSIGN: %[[S2:.*]] = arith.constant 2 : index
// ASSIGN: ttl.copy_tile {{.*}} into dst[%[[S2]]]
// ASSIGN: %[[S3:.*]] = arith.constant 3 : index
// ASSIGN: ttl.copy_tile {{.*}} into dst[%[[S3]]]
// Each per-iteration accumulate targets the slot initialized for its tile.
// ASSIGN: scf.for
// ASSIGN: ttl.tile_accumulate {{.*}} into dst[%[[S0]]]
// ASSIGN: ttl.tile_accumulate {{.*}} into dst[%[[S1]]]
// ASSIGN: ttl.tile_accumulate {{.*}} into dst[%[[S2]]]
// ASSIGN: ttl.tile_accumulate {{.*}} into dst[%[[S3]]]
// Each final store reads the slot its tile accumulated into.
// ASSIGN: ttl.tile_store {{.*}} from dst[%[[S0]]]
// ASSIGN: ttl.tile_store {{.*}} from dst[%[[S1]]]
// ASSIGN: ttl.tile_store {{.*}} from dst[%[[S2]]]
// ASSIGN: ttl.tile_store {{.*}} from dst[%[[S3]]]
func.func @multi_tile_recurrence() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%acc = %init)
      -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    scf.yield %next : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}
