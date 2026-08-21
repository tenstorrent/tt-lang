// Verify that automatic DFB synchronization rejects external release effects
// that cannot be relocated without deleting the external call.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --verify-diagnostics

// A nested external push cannot satisfy an entry-block reserve.
module {
  func.func @nested_external_push(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      // expected-error @below {{external DFB push effect must be in the same block as its acquisition}}
      ttl.opaque_call "publish" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A nested external pop cannot satisfy an entry-block wait.
module {
  func.func @nested_external_pop(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      // expected-error @below {{external DFB pop effect must be in the same block as its acquisition}}
      ttl.opaque_call "release" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A conditional reserve must be released under the same condition.
module {
  func.func @guarded_reserve_missing_push(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      // expected-error @below {{conditional dataflow buffer push requires an explicit release after its guarded uses under the same condition}}
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %arg0, %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.for %i = %c0 to %c1 step %c1 {
      scf.if %condition {
        ttl.store %arg0, %view {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
    }
    return
  }
}

// -----

// A release in the acquiring region cannot close a DFB view that escapes as an
// scf.if result and is used by a later guarded region.
module {
  func.func @guarded_reserve_release_before_external_use(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      // expected-error @below {{conditional dataflow buffer push requires an explicit release after its guarded uses under the same condition}}
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %condition {
      ttl.store %arg0, %view {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    return
  }
}
