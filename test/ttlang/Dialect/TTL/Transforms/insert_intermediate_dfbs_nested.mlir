// Compiler-allocated bind_cb must live at the function body entry so its
// cb_index is function-scoped and dominates every nested reserve/wait.
// When the intermediate's defining op sits inside a structured region
// (scf.for, scf.if), only bind_cb hoists; reserve/store/wait/attach
// stay at the def site to keep per-invocation accounting intact.

// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs,ttl-insert-cb-sync),ttl-finalize-dfb-indices)' | FileCheck %s

// -----

// Intermediate produced inside scf.for: bind_cb must be at function body
// entry, the rest of the materialization bundle stays inside the loop.

// CHECK-LABEL: func.func @intermediate_in_scf_for
// CHECK: ttl.bind_cb{cb_index = 3, block_count = 2} {ttl.compiler_allocated}
// CHECK: scf.for
// CHECK:   ttl.add
// CHECK:   ttl.cb_reserve
// CHECK:   ttl.store
// CHECK:   ttl.cb_push
// CHECK:   ttl.cb_wait
// CHECK:   ttl.attach_cb
// CHECK:   ttl.reduce
// CHECK:   ttl.cb_pop
// CHECK: }
// CHECK: return
func.func @intermediate_in_scf_for()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  scf.for %iv = %c0 to %c4 step %c1 {
    %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s = ttl.attach_cb %s_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %add = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %add, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  return
}

// -----

// Intermediate produced inside scf.if then-branch: bind_cb hoists to
// function body entry, not inside the if region.

// CHECK-LABEL: func.func @intermediate_in_scf_if
// CHECK: ttl.bind_cb{cb_index = 3, block_count = 2} {ttl.compiler_allocated}
// CHECK: scf.if
// CHECK-NOT: ttl.compiler_allocated
// CHECK:   ttl.add
// CHECK:   ttl.cb_reserve
// CHECK:   ttl.store
// CHECK:   ttl.cb_wait
// CHECK:   ttl.attach_cb
// CHECK:   ttl.reduce
// CHECK: }
// CHECK: return
func.func @intermediate_in_scf_if(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  scf.if %cond {
    %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s = ttl.attach_cb %s_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %add = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %add, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  return
}

// -----

// Mirrors the issue #518 layernorm shape: outer scf.for with an inner
// scf.if; the intermediate add -> reduce sequence lives inside the if.
// Before the fix, this crashed in TTLFinalizeDFBIndices.

// CHECK-LABEL: func.func @intermediate_in_nested_for_if
// CHECK: ttl.bind_cb{cb_index = 3, block_count = 2} {ttl.compiler_allocated}
// CHECK: scf.for
// CHECK:   scf.if
// CHECK-NOT: ttl.compiler_allocated
// CHECK:     ttl.add
// CHECK:     ttl.reduce
// CHECK:   }
// CHECK: }
// CHECK: return
func.func @intermediate_in_nested_for_if(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  scf.for %iv = %c0 to %c4 step %c1 {
    scf.if %cond {
      %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %s_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %s = ttl.attach_cb %s_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

      %add = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %reduced = ttl.reduce %add, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

      ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.cb_pop %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
  }
  return
}
