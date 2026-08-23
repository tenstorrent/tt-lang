// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))'

// Verify that waited-block replacement rejects every protocol or ordering case
// for which the compiler cannot prove complete occupied-slot replacement.

func.func @overwrite_before_read()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.fill 1.000000e+00
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement must read the acquired value before writing}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @multi_block_cursor_state()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires one complete DFB block}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @partial_one_block_acquisition()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.cb_wait' op result tensor has 2 tiles but num_tiles attribute is 1}}
  %wait = ttl.cb_wait %dfb {num_tiles = 1 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %replacement, %wait
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @repeated_replacement()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires exactly one store per acquisition}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @conditional_replacement(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    %replacement = ttl.add %block, %block
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires one straight-line entry-block execution}}
    ttl.store %replacement, %wait
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

func.func @iterated_replacement()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iteration = %c0 to %c1 step %c1 {
    %replacement = ttl.add %block, %block
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires one straight-line entry-block execution}}
    ttl.store %replacement, %wait
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

func.func @live_old_generation()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %stale = ttl.exp %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement has an unsupported or unordered use of the acquired generation}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %stale, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A replacement-trace intermediate still denotes the original generation and
// cannot be recomputed after the in-place write for an independent store.
func.func @original_generation_intermediate_escape()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %old_intermediate = ttl.exp %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %old_intermediate, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires values derived from the original DFB contents to remain within the replacement computation}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %old_intermediate, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

func.func @overlapping_producer_acquisition()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %producer = ttl.cb_reserve %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement cannot overlap a producer acquisition}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @prior_partial_consumer_transaction()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  ttl.opaque_call "consume_one" dfb_dependencies(
      %dfb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
      dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                   #ttl.dfb_protocol_effect<pop, 0, 1>]
      () {header = "consumer.hpp"} : () -> ()
  %wait = ttl.cb_wait %dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires no earlier access to the same DFB in its compute kernel}}
  ttl.store %replacement, %wait
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

func.func @release_before_replacement()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement cannot execute after its consumer release}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @missing_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires a matching consumer release}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @mismatched_release_count()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement requires equal wait and pop tile counts}}
  ttl.store %replacement, %wait
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb {num_tiles = 1 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

func.func @replacement_read_after_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement value cannot be read after its consumer release}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %late = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %late, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

func.func @opaque_access_during_replacement()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement cannot overlap another DFB access}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.opaque_call "inspect" dfb_dependencies(
      %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      () {header = "inspect.hpp", unknown_dfb_access} : () -> ()
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

func.func @unknown_opaque_access_during_replacement()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement cannot overlap another DFB access}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.opaque_call "inspect" ()
      {header = "inspect.hpp", unknown_dfb_access} : () -> ()
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

func.func @nested_dfb_access_during_replacement(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %replacement = ttl.add %block, %block
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: wait-backed replacement cannot overlap a nested DFB access}}
  ttl.store %replacement, %wait
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    ttl.opaque_call "inspect" dfb_dependencies(
        %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        () {header = "inspect.hpp"} : () -> ()
  }
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}
