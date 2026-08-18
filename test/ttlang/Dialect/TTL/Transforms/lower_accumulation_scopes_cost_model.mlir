// Verifies tensor accumulation strategy planning reports architecture-specific
// cost scores for legal DST and L1 packer candidates.
//
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' -debug-only=ttl-lower-accumulation-scopes 2>&1 | FileCheck %s

// Purpose: Blackhole uses the initial Track A DFB cost weights.
// CHECK: accumulation cost model target_arch=blackhole
// CHECK-NEXT:   candidate strategy=dst legal=true estimated_cost=1738 one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=1 dst_live_tiles=1 pack_reconfigs=0
// CHECK-NEXT:   candidate strategy=l1-pack legal=true estimated_cost=1939 one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=2 dst_live_tiles=0 pack_reconfigs=2
// CHECK-NEXT:   selected strategy=dst
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @cost_model_blackhole() {
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
    ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %state) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } initial_modes([init])
    func.return
  }
}

// -----

// Purpose: Wormhole uses higher traffic scores derived from tt-metal LLK perf
// data.
// CHECK: accumulation cost model target_arch=wormhole_b0
// CHECK-NEXT:   candidate strategy=dst legal=true estimated_cost=2546 one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=1 dst_live_tiles=1 pack_reconfigs=0
// CHECK-NEXT:   candidate strategy=l1-pack legal=true estimated_cost=2954 one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=2 dst_live_tiles=0 pack_reconfigs=2
// CHECK-NEXT:   selected strategy=dst
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @cost_model_wormhole() {
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
    ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %state) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } initial_modes([init])
    func.return
  }
}

// -----

// Purpose: IR without a target architecture reports unknown estimates and uses
// structural feature counts for deterministic auto selection.
// CHECK: accumulation cost model target_arch=unknown
// CHECK-NEXT:   candidate strategy=dst legal=true estimated_cost=unknown one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=1 dst_live_tiles=1 pack_reconfigs=0
// CHECK-NEXT:   candidate strategy=l1-pack legal=true estimated_cost=unknown one_time_dfb_hops=1 per_iteration_dfb_hops=2 one_time_pack_unpack_tiles=1 per_iteration_pack_unpack_tiles=2 dst_live_tiles=0 pack_reconfigs=2
// CHECK-NEXT:   selected strategy=dst
module {
  func.func @cost_model_unknown() {
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
    ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %state) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
        %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } initial_modes([init])
    func.return
  }
}
