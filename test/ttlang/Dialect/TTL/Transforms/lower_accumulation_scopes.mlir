// Verifies ttl-lower-accumulation-scopes consumes tensor accumulation scopes
// through concrete strategy selection.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=dst}))' | FileCheck %s --check-prefix=DST
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=l1-pack}))' | FileCheck %s --check-prefix=L1
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' | FileCheck %s --check-prefix=AUTO

// DST strategy lowers the recurrence to one reduction compute and consumes the
// semantic scope.
// DST-LABEL: func.func @tensor_accumulation_scope
// DST: %[[RESERVE:.*]] = ttl.cb_reserve
// DST: ttl.cb_wait %{{.*}} {num_tiles = 3 : i64}
// DST: ttl.compute
// DST: ttl.tile_accumulate_add
// DST: ttl.tile_store %{{.*}}, %[[RESERVE]]
// DST: ttl.cb_pop %{{.*}} {num_tiles = 3 : i64}
// DST-NOT: ttl.accumulation_scope
// DST-NOT: scf.for
//
// L1 packer strategy materializes an explicit initial store and one
// accumulating store inside an annotated loop.
// L1-LABEL: func.func @tensor_accumulation_scope
// L1: %[[RESERVE:.*]] = ttl.cb_reserve
// L1: ttl.store %{{.*}}, %[[RESERVE]]
// L1: scf.for {{.*}} {
// L1: ttl.store %{{.*}}, %[[RESERVE]] {accumulate}
// L1: } {ttl.l1_acc_loop}
// L1-NOT: ttl.accumulation_scope
// L1-NOT: iter_args
//
// Auto selects DST when the scope satisfies the DST strategy legality rules.
// AUTO-LABEL: func.func @tensor_accumulation_scope
// AUTO: ttl.compute
// AUTO: ttl.tile_accumulate_add
// AUTO-NOT: ttl.accumulation_scope
func.func @tensor_accumulation_scope() {
  %cb_init = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_delta = ttl.bind_cb {cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %cb_init : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %cb_init : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  ttl.accumulation_scope outs(%reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [2 : i32]}
  func.return
}
