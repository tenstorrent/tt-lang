// Verifies that `ComputeOp` creation diagnoses inconsistent output publication
// before modifying the source kernel.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))'

// Stores from one reserve cannot mix a published store with a store that has
// no later publication because they do not define one output transaction.
func.func @missing_second_publication()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %input, %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{stores from one reserve do not precede the same ttl.cb_push operation}}
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Two partial publications can balance one reserve's tile count, but stores
// from that reserve still define one output transaction and therefore cannot
// refer to different publication operations.
func.func @multiple_publications_for_one_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb {num_tiles = 2 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %input, %input
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb {num_tiles = 1 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{stores from one reserve do not precede the same ttl.cb_push operation}}
  ttl.store %result, %output
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb {num_tiles = 1 : i64}
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// Final conversion requires every tensor store to be absorbed into a planned
// ttl.compute or converted to a DFB-to-DFB passthrough. Running it without the
// required intermediate materialization diagnoses the rejected store before
// any rewrite executes.
func.func @unassigned_final_store()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %result = ttl.add %input, %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: moving tensor evaluation to the final output store would read a dataflow buffer value after its pop}}
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// An attached passthrough input still requires live acquired storage. The
// final preflight reports the store's lifetime rejection rather than treating
// ttl.attach_cb as a source operation for `ComputeOp` creation.
func.func @released_passthrough_input()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: passthrough store would read a dataflow buffer value after release}}
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Conditional routing to different output DFBs has no single operation at
// which one ttl.compute can preserve both stores. The diagnostic must report
// that source-level rejection instead of misclassifying the tensor input.
func.func @conditional_output_routing(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %input, %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    %then_output = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{cannot lower tensor store to ttl.compute: output stores are in different blocks}}
    ttl.store %sum, %then_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_output = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %else_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
