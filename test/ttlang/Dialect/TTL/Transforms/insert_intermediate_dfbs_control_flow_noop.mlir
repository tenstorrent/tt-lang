// Verifies control-flow cases that ttl-insert-intermediate-dfbs must leave
// unchanged.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs))' | FileCheck %s

// Function arguments have no producer operation to materialize in this pass.
// The control-flow store normalization only applies to computed values.

// CHECK-LABEL: func.func @func_argument_stores_not_computed_value
// CHECK-NOT: ttl.compiler_allocated
// CHECK: scf.if
// CHECK: ttl.store %arg0
// CHECK: } else {
// CHECK: ttl.store %arg0
// CHECK: return
func.func @func_argument_stores_not_computed_value(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %then_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  scf.if %condition {
    %then_view = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %arg0, %then_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_view = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %arg0, %else_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A DFB-backed value already has storage and is not materialized again when
// stored from multiple blocks.

// CHECK-LABEL: func.func @dfb_backed_value_stores_unchanged
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[INPUT:.*]] = ttl.attach_cb
// CHECK: scf.if
// CHECK: ttl.store %[[INPUT]]
// CHECK: } else {
// CHECK: ttl.store %[[INPUT]]
// CHECK: return
func.func @dfb_backed_value_stores_unchanged(%condition: i1)
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

  scf.if %condition {
    %then_view = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %input, %then_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_view = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %input, %else_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// Declaration-only functions are ignored by the function pass.

// CHECK-LABEL: func.func private @external_kernel
func.func private @external_kernel(tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>}
