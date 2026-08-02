// Verifies that intermediate DFB planning uses control-flow-aware value
// availability across branches, loops, and repeated acquisitions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-dfb-value-lifetimes))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=ANALYSIS

// A release on one branch makes the pre-branch expression unavailable at the
// join. Materialization before the branch preserves the expression.
// CHECK-LABEL: func.func @conditional_release_before_consumer
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: scf.if
// CHECK:   ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]]
// ANALYSIS-LABEL: DFB value lifetimes @conditional_release_before_consumer
// ANALYSIS: R0 consumer tiles=1 owner=unresolved [A0]
// ANALYSIS: ttl.exp A0=may-be-released
func.func @conditional_release_before_consumer(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
    ttl.cb_pop %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  %result = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A nested block may inherit an outstanding acquisition. Its local wait/pop
// pair therefore cannot initialize an independent FIFO: the pop may consume
// the enclosing acquisition first. Unresolved ownership invalidates both
// identities conservatively.
// ANALYSIS-LABEL: DFB value lifetimes @nested_fifo_has_incoming_state
// ANALYSIS: R0 consumer tiles=1 owner=unresolved [A0, A2, A4]
// ANALYSIS-NEXT: R1 consumer tiles=1 owner=unresolved [A0, A2, A4]
// ANALYSIS: ttl.signpost A0=may-be-released
// ANALYSIS: ttl.signpost A2=may-be-released
// ANALYSIS: func.return A0=may-be-released
// ANALYSIS: func.return A4=may-be-released
func.func @nested_fifo_has_incoming_state(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %outer_output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inner_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %later_output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %outer_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %outer = ttl.attach_cb %outer_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %outer_output = ttl.cb_reserve %outer_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %outer, %outer_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    %inner_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inner = ttl.attach_cb %inner_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inner_output = ttl.cb_reserve %inner_output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %inner, %inner_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    ttl.signpost "after nested release"
  }
  %later_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %later = ttl.attach_cb %later_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %later_output = ttl.cb_reserve %later_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %later, %later_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  return
}

// -----

// Control flow without a release keeps the input available and requires no
// compiler-created DFB.
// CHECK-LABEL: func.func @conditional_without_release
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.if
// CHECK: ttl.compute
// ANALYSIS-LABEL: DFB value lifetimes @conditional_without_release
// ANALYSIS: ttl.exp A0=available
func.func @conditional_without_release(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
    ttl.signpost "no release"
  }
  %result = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A loop may execute and release the input. Materialization before the loop
// makes the pre-loop expression available after every loop exit.
// CHECK-LABEL: func.func @loop_release_before_consumer
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: scf.for
// CHECK:   ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]]
// ANALYSIS-LABEL: DFB value lifetimes @loop_release_before_consumer
// ANALYSIS: R0 consumer tiles=1 owner=unresolved [A0]
// ANALYSIS: ttl.exp A0=may-be-released
func.func @loop_release_before_consumer(%upper_bound: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
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
  scf.for %iteration = %zero to %upper_bound step %one {
    ttl.cb_pop %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  %result = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A second wait establishes a new acquisition identity. Releasing the first
// acquisition must not invalidate a value obtained from the second.
// CHECK-LABEL: func.func @release_then_reacquire
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: ttl.cb_wait %[[INPUT_DFB]]
// CHECK: ttl.cb_pop %[[INPUT_DFB]]
// CHECK: %[[SECOND_WAIT:.*]] = ttl.cb_wait %[[INPUT_DFB]]
// CHECK: %[[SECOND_INPUT:.*]] = ttl.attach_cb %[[SECOND_WAIT]], %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[SECOND_INPUT]]
// ANALYSIS-LABEL: DFB value lifetimes @release_then_reacquire
// ANALYSIS: R0 consumer tiles=1 owner=exact A0
// ANALYSIS: ttl.exp A0=may-be-released
// ANALYSIS-NEXT: {{.*}}ttl.exp A1=available
func.func @release_then_reacquire()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first = ttl.attach_cb %first_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.attach_cb %second_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %second
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A wait inside a loop re-establishes availability on every iteration. A
// compute and store before the matching pop require no intermediate DFB.
// CHECK-LABEL: func.func @loop_local_acquisition
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.for
// CHECK:   %[[INPUT_WAIT:.*]] = ttl.cb_wait
// CHECK:   %[[INPUT:.*]] = ttl.attach_cb %[[INPUT_WAIT]]
// CHECK:   ttl.compute ins(%[[INPUT]]
// CHECK:   ttl.cb_pop
// ANALYSIS-LABEL: DFB value lifetimes @loop_local_acquisition
// ANALYSIS: ttl.exp A0=available
func.func @loop_local_acquisition(%upper_bound: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %iteration = %zero to %upper_bound step %one {
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.exp %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  return
}

// -----

// A constant-false branch has no runtime consumers. Dead-code analysis marks
// its block non-executable, so missing dense lattices satisfy availability
// requirements vacuously and must not force a compiler DFB.
// CHECK-LABEL: func.func @statically_false_branch
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.if
// CHECK:   ttl.compute
// ANALYSIS-LABEL: DFB value lifetimes @statically_false_branch
// ANALYSIS: ttl.reduce A0=available
// ANALYSIS-NEXT: {{.*}}ttl.reduce A1=available
func.func @statically_false_branch()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %never = arith.constant false
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %never {
    %reduced = ttl.reduce %input, %scaler 0 : i32 [1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A statically zero-trip loop has the same availability contract as other
// non-executable regions: no runtime read occurs, and no compiler DFB is
// required solely to serve operations in its body.
// CHECK-LABEL: func.func @zero_trip_loop
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.for
// CHECK:   ttl.compute
// ANALYSIS-LABEL: DFB value lifetimes @zero_trip_loop
// ANALYSIS: ttl.reduce A0=available
// ANALYSIS-NEXT: {{.*}}ttl.reduce A1=available
func.func @zero_trip_loop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iteration = %zero to %zero step %one {
    %reduced = ttl.reduce %input, %scaler 0 : i32 [1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
