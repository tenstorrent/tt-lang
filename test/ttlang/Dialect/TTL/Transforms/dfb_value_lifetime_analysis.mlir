// Tests acquisition identity, tile-counted release ownership, and conservative
// partial-release availability without modifying IR.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-dfb-value-lifetimes))' -o /dev/null 2>&1 | FileCheck %s

// Same-block releases own same-DFB acquisitions in FIFO order.
// CHECK-LABEL: DFB value lifetimes @same_block_fifo
// CHECK: A0 consumer tiles=1
// CHECK-NEXT: A1 consumer tiles=1
// CHECK-NEXT: R0 consumer tiles=1 owner=exact A0
// CHECK-NEXT: R1 consumer tiles=1 owner=exact A1
func.func @same_block_fifo()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %first = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  return
}

// -----

// A partial release records its actual count and invalidates the complete
// tensor because the analysis does not yet track tensor tile ranges.
// CHECK-LABEL: DFB value lifetimes @partial_consumer_release
// CHECK: A0 consumer tiles=2
// CHECK-NEXT: R0 consumer tiles=1 owner=exact A0
// CHECK: ttl.signpost A0=may-be-released
func.func @partial_consumer_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb {num_tiles = 2 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb {num_tiles = 1 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.signpost "after partial release"
  return
}

// -----

// Producer acquisitions and releases use the same tile-counted ownership
// model as consumer transactions.
// CHECK-LABEL: DFB value lifetimes @producer_transaction
// CHECK: A0 producer tiles=2
// CHECK-NEXT: R0 producer tiles=2 owner=exact A0
// CHECK: ttl.signpost A0=may-be-released
func.func @producer_transaction()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_reserve %dfb {num_tiles = 2 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %dfb {num_tiles = 2 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.signpost "after publication"
  return
}

// -----

// A hidden pop conservatively invalidates concrete acquired storage.
// CHECK-LABEL: DFB value lifetimes @external_release
// CHECK: A0 consumer tiles=1
// CHECK: ttl.signpost A0=may-be-released
func.func @external_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.opaque_call "release" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
  ttl.signpost "after external release"
  return
}

// -----

// Unknown access invalidates user-managed storage but cannot name a
// compiler-created intermediate that is absent from the external contract.
// CHECK-LABEL: DFB value lifetimes @unknown_user_access
// CHECK: A0 consumer tiles=1
// CHECK-NEXT: A1 consumer tiles=1
// CHECK: ttl.signpost A0=may-be-released
// CHECK-NEXT: {{.*}}ttl.signpost A1=available
func.func @unknown_user_access()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %intermediate = ttl.bind_cb {cb_index = 1, block_count = 2}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %user_input = ttl.cb_wait %user
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %intermediate_input = ttl.cb_wait %intermediate
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.opaque_call "unknown" () {header = "effects.hpp", unknown_dfb_access} : () -> ()
  ttl.signpost "after unknown access"
  return
}

// -----

// A tile-counted release can consume several earlier acquisitions without
// releasing a later acquisition on the same DFB.
// CHECK-LABEL: DFB value lifetimes @multi_acquisition_release
// CHECK: A0 consumer tiles=1
// CHECK-NEXT: A1 consumer tiles=1
// CHECK-NEXT: A2 consumer tiles=1
// CHECK-NEXT: R0 consumer tiles=2 owner=multiple [A0, A1]
// CHECK: ttl.signpost A0=may-be-released
// CHECK-NEXT: {{.*}}ttl.signpost A1=may-be-released
// CHECK-NEXT: {{.*}}ttl.signpost A2=available
func.func @multi_acquisition_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %first = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %third = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb {num_tiles = 2 : i64}
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  ttl.signpost "after two-tile release"
  return
}

// -----

// An association records storage identity but does not acquire the DFB. A
// release before the association therefore remains visible afterward.
// CHECK-LABEL: DFB value lifetimes @association_does_not_reacquire
// CHECK: R0 consumer tiles=1 owner=exact A0
// CHECK: ttl.signpost S0=may-be-released
func.func @association_does_not_reacquire(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waited = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %associated = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "after association"
  return
}

// -----

// A standalone association represents DFB storage supplied at kernel entry.
// It remains available until a release on the associated DFB.
// CHECK-LABEL: DFB value lifetimes @kernel_input_association
// CHECK: ttl.signpost S0=available
func.func @kernel_input_association(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %associated = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "after association"
  return
}

// -----

// An unconditional DST section preserves incoming DFB availability both in
// its body and after the region.
// CHECK-LABEL: DFB value lifetimes @unconditional_dst_section
// CHECK-COUNT-2: ttl.signpost A0=available
func.func @unconditional_dst_section()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    ttl.signpost "inside DST section"
    ttl.yield
  }
  ttl.signpost "after DST section"
  return
}
