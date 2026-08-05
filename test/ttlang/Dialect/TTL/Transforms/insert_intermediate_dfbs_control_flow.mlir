// Verifies intermediate DFB materialization for values stored from multiple
// control-flow blocks.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs))' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs,ttl-insert-cb-sync,convert-ttl-to-compute))' | FileCheck %s --check-prefix=PIPELINE

// -----

// A value stored from both arms of one branch is routed through compiler DFB
// storage before the branch.

// CHECK-LABEL: func.func @stored_value_across_scf_if
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.*]] = ttl.exp
// CHECK: %[[RESERVED:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// CHECK: ttl.store %[[VALUE]], %[[RESERVED]]
// CHECK: %[[WAITED:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[ATTACHED:.*]] = ttl.attach_cb %[[WAITED]], %[[INTERMEDIATE_DFB]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: } else {
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @stored_value_across_scf_if
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @stored_value_across_scf_if(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
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
  %value = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %condition {
    %then_view = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %then_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_view = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %else_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A value with one branch-local store does not need compiler DFB storage.

// CHECK-LABEL: func.func @single_branch_store_unchanged
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[VALUE:.*]] = ttl.exp
// CHECK: scf.if
// CHECK: ttl.store %[[VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @single_branch_store_unchanged
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @single_branch_store_unchanged(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
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
  %value = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %condition {
    %output_view = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %output_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// When one store is in the defining block and another is in a branch, both
// stores use the materialized value so final compute creation has one output
// publication for the original producer.

// CHECK-LABEL: func.func @defining_block_and_branch_stores
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.*]] = ttl.exp
// CHECK: ttl.store %[[VALUE]],
// CHECK: %[[WAITED:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[ATTACHED:.*]] = ttl.attach_cb %[[WAITED]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return
func.func @defining_block_and_branch_stores(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %first_view = ttl.cb_reserve %first_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %value, %first_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %condition {
    %second_view = ttl.cb_reserve %second_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %second_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A multi-result compute can also have one result stored from multiple blocks.
// The per-result materialization adds one compiler DFB output without changing
// the other compute result.

// CHECK-LABEL: func.func @multi_result_compute_result_stored_across_scf_if
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[RESERVED:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// CHECK: %{{.*}}:3 = ttl.compute
// CHECK: ttl.tile_store {{.*}}, %[[RESERVED]]
// CHECK: %[[WAITED:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[ATTACHED:.*]] = ttl.attach_cb %[[WAITED]], %[[INTERMEDIATE_DFB]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: } else {
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @multi_result_compute_result_stored_across_scf_if
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @multi_result_compute_result_stored_across_scf_if(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 5 : i32, ttl.crta_indices = []} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sum_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %product_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_dfb = ttl.bind_cb {cb_index = 5, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c-1 = arith.constant -1 : index

  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sum_reserve = ttl.cb_reserve %sum_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %product_reserve = ttl.cb_reserve %product_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_output = ttl.attach_cb %sum_empty, %sum_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %product_empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %product_output = ttl.attach_cb %product_empty, %product_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                     tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%sum_output, %product_output
           : tensor<1x1x!ttcore.tile<32x32, bf16>>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %sum_output_tile: !ttcore.tile<32x32, bf16>,
       %product_output_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum_tile = ttl.tile_add %lhs_tile, %rhs_tile into dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    %product_tile = ttl.tile_mul %lhs_tile, %rhs_tile into dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum_tile, %sum_reserve[%i, %j] from dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %product_tile, %product_reserve[%i, %j] from dst[%c-1]
        {ttl.dst_placeholder}
        : !ttcore.tile<32x32, bf16>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>)

  scf.if %condition {
    %then_view = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result#0, %then_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_view = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result#0, %else_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}
