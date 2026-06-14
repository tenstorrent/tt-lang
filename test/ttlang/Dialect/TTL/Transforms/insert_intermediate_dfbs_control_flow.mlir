// Tests branch-local cloning and fallback DFB materialization for values stored
// from multiple control-flow regions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs))' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs,ttl-insert-cb-sync,convert-ttl-to-compute))' | FileCheck %s --check-prefix=PIPELINE

// -----

// A clone-supported value stored by mutually exclusive branches is cloned into
// each branch, avoiding an intermediate compiler-managed DFB.

// CHECK-LABEL: func.func @store_fanout_across_scf_if
// CHECK-NOT: ttl.compiler_allocated
// CHECK-NOT: ttl.exp
// CHECK: scf.if
// CHECK: %[[THEN_VALUE:.+]] = ttl.exp
// CHECK: ttl.store %[[THEN_VALUE]]
// CHECK: } else {
// CHECK: %[[ELSE_VALUE:.+]] = ttl.exp
// CHECK: ttl.store %[[ELSE_VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @store_fanout_across_scf_if
// PIPELINE-COUNT-2: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @store_fanout_across_scf_if(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %then_reserve = ttl.cb_reserve %then_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %then_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_reserve = ttl.cb_reserve %else_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %else_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// Nested if/else regions clone the producer when every store block is pairwise
// mutually exclusive.

// CHECK-LABEL: func.func @nested_if_store_fanout_clones
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[INPUT:.+]] = ttl.attach_cb
// CHECK-NOT: ttl.exp
// CHECK: scf.if
// CHECK: scf.if
// CHECK: %[[FIRST_VALUE:.+]] = ttl.exp %[[INPUT]]
// CHECK: ttl.store %[[FIRST_VALUE]]
// CHECK: } else {
// CHECK: %[[SECOND_VALUE:.+]] = ttl.exp %[[INPUT]]
// CHECK: ttl.store %[[SECOND_VALUE]]
// CHECK: } else {
// CHECK: %[[THIRD_VALUE:.+]] = ttl.exp %[[INPUT]]
// CHECK: ttl.store %[[THIRD_VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @nested_if_store_fanout_clones
// PIPELINE-COUNT-3: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @nested_if_store_fanout_clones(%outer_cond: i1, %inner_cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 9 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %third_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %outer_cond {
    scf.if %inner_cond {
      %first_reserve = ttl.cb_reserve %first_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %value, %first_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %second_reserve = ttl.cb_reserve %second_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %value, %second_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  } else {
    %third_reserve = ttl.cb_reserve %third_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %third_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A multi-op backward slice stored more than once in one branch block is cloned
// once before the earliest store in that block.

// CHECK-LABEL: func.func @multi_store_branch_clones_once_per_block
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[LHS:.+]] = ttl.attach_cb
// CHECK: %[[RHS:.+]] = ttl.attach_cb
// CHECK-NOT: ttl.add
// CHECK-NOT: ttl.exp
// CHECK: scf.if
// CHECK: %[[THEN_ADD:.+]] = ttl.add %[[LHS]], %[[RHS]]
// CHECK: %[[THEN_VALUE:.+]] = ttl.exp %[[THEN_ADD]]
// CHECK: ttl.store %[[THEN_VALUE]]
// CHECK: ttl.store %[[THEN_VALUE]]
// CHECK: } else {
// CHECK: %[[ELSE_ADD:.+]] = ttl.add %[[LHS]], %[[RHS]]
// CHECK: %[[ELSE_VALUE:.+]] = ttl.exp %[[ELSE_ADD]]
// CHECK: ttl.store %[[ELSE_VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @multi_store_branch_clones_once_per_block
// PIPELINE-COUNT-2: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @multi_store_branch_clones_once_per_block(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 5 : i32, ttl.crta_indices = []} {
  %lhs_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_first_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_second_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_cb = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %lhs_wait = ttl.cb_wait %lhs_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %sum : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %then_first_reserve = ttl.cb_reserve %then_first_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %then_first_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    %then_second_reserve = ttl.cb_reserve %then_second_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %then_second_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_reserve = ttl.cb_reserve %else_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %else_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// Independent sibling ifs are not structurally mutually exclusive, so the
// branch stores use a compiler-managed DFB.

// CHECK-LABEL: func.func @sibling_if_store_fanout_materializes
// CHECK: %[[COMPILER_DFB:.+]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.+]] = ttl.exp
// CHECK: %[[RESERVED:.+]] = ttl.cb_reserve %[[COMPILER_DFB]]
// CHECK: ttl.store %[[VALUE]], %[[RESERVED]]
// CHECK: %[[WAITED:.+]] = ttl.cb_wait %[[COMPILER_DFB]]
// CHECK: %[[ATTACHED:.+]] = ttl.attach_cb %[[WAITED]], %[[COMPILER_DFB]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @sibling_if_store_fanout_materializes
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @sibling_if_store_fanout_materializes(%cond_a: i1, %cond_b: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 6 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond_a {
    %first_reserve = ttl.cb_reserve %first_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %first_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  scf.if %cond_b {
    %second_reserve = ttl.cb_reserve %second_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %second_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// Sibling ifs under a common outer branch are still not pairwise exclusive.

// CHECK-LABEL: func.func @nested_sibling_if_store_fanout_materializes
// CHECK: %[[COMPILER_DFB:.+]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.+]] = ttl.exp
// CHECK: %[[RESERVED:.+]] = ttl.cb_reserve %[[COMPILER_DFB]]
// CHECK: ttl.store %[[VALUE]], %[[RESERVED]]
// CHECK: %[[WAITED:.+]] = ttl.cb_wait %[[COMPILER_DFB]]
// CHECK: %[[ATTACHED:.+]] = ttl.attach_cb %[[WAITED]], %[[COMPILER_DFB]]
// CHECK: scf.if
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: } else {
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @nested_sibling_if_store_fanout_materializes
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @nested_sibling_if_store_fanout_materializes(%outer_cond: i1, %cond_a: i1, %cond_b: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 10 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %third_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %outer_cond {
    scf.if %cond_a {
      %first_reserve = ttl.cb_reserve %first_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %value, %first_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %cond_b {
      %second_reserve = ttl.cb_reserve %second_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %value, %second_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  } else {
    %third_reserve = ttl.cb_reserve %third_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %third_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A single branch-local store cannot trigger isBeforeInBlock across blocks, so
// it remains on the original value without a compiler-managed DFB.

// CHECK-LABEL: func.func @single_branch_reduce_store_unchanged
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[VALUE:.+]] = ttl.reduce
// CHECK: scf.if
// CHECK: ttl.store %[[VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @single_branch_reduce_store_unchanged
// PIPELINE-COUNT-1: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @single_branch_reduce_store_unchanged(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 7 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scale_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale_wait = ttl.cb_wait %scale_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale = ttl.attach_cb %scale_wait, %scale_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.reduce %input, %scale 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %out_reserve = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %out_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// Multiple stores in the same branch-local block are legal same-block
// multi-output stores and do not need a compiler-managed DFB.

// CHECK-LABEL: func.func @same_branch_reduce_multi_store_unchanged
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[VALUE:.+]] = ttl.reduce
// CHECK: scf.if
// CHECK: ttl.store %[[VALUE]]
// CHECK: ttl.store %[[VALUE]]
// CHECK: return

// PIPELINE-LABEL: func.func @same_branch_reduce_multi_store_unchanged
// PIPELINE-COUNT-1: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @same_branch_reduce_multi_store_unchanged(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 8 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scale_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_out_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale_wait = ttl.cb_wait %scale_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale = ttl.attach_cb %scale_wait, %scale_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.reduce %input, %scale 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %first_out_reserve = ttl.cb_reserve %first_out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %first_out_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_out_reserve = ttl.cb_reserve %second_out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %second_out_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A producer outside the current clone-supported set uses the compiler-managed
// DFB fallback.

// CHECK-LABEL: func.func @reduce_fanout_across_scf_if
// CHECK: %[[COMPILER_DFB:.+]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.+]] = ttl.reduce
// CHECK: %[[RESERVED:.+]] = ttl.cb_reserve %[[COMPILER_DFB]]
// CHECK: ttl.store %[[VALUE]], %[[RESERVED]]
// CHECK: %[[WAITED:.+]] = ttl.cb_wait %[[COMPILER_DFB]]
// CHECK: %[[ATTACHED:.+]] = ttl.attach_cb %[[WAITED]], %[[COMPILER_DFB]]
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: } else {
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @reduce_fanout_across_scf_if
// PIPELINE-COUNT-3: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @reduce_fanout_across_scf_if(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scale_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale_wait = ttl.cb_wait %scale_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale = ttl.attach_cb %scale_wait, %scale_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.reduce %input, %scale 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %then_reserve = ttl.cb_reserve %then_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %then_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_reserve = ttl.cb_reserve %else_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %else_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}

// -----

// A value stored both in its defining block and in a branch spans two blocks,
// but the stores are not mutually exclusive. The branch store uses the DFB
// fallback and the defining-block store keeps the original value.

// CHECK-LABEL: func.func @store_fanout_defining_block_and_branch_materializes
// CHECK: %[[COMPILER_DFB:.+]] = ttl.bind_cb{{.*}}{ttl.compiler_allocated}
// CHECK: %[[VALUE:.+]] = ttl.exp
// CHECK: %[[RESERVED:.+]] = ttl.cb_reserve %[[COMPILER_DFB]]
// CHECK: ttl.store %[[VALUE]], %[[RESERVED]]
// CHECK: %[[WAITED:.+]] = ttl.cb_wait %[[COMPILER_DFB]]
// CHECK: %[[ATTACHED:.+]] = ttl.attach_cb %[[WAITED]], %[[COMPILER_DFB]]
// CHECK: ttl.store %[[VALUE]], %{{.+}}
// CHECK: scf.if
// CHECK: ttl.store %[[ATTACHED]]
// CHECK: return

// PIPELINE-LABEL: func.func @store_fanout_defining_block_and_branch_materializes
// PIPELINE: ttl.compute
// PIPELINE-NOT: ttl.store
// PIPELINE: return
func.func @store_fanout_defining_block_and_branch_materializes(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 9 : i32, ttl.crta_indices = []} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %always_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %branch_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %value = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %always_reserve = ttl.cb_reserve %always_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %value, %always_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %branch_reserve = ttl.cb_reserve %branch_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %branch_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }

  return
}
