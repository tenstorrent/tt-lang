// Tests invalid control-flow store fanout when compiler DFBs are disabled.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs{enable=false}))'

// -----

// A non-rematerializable value cannot be stored from branch blocks when
// compiler-managed DFBs are disabled.

func.func @reduce_fanout_across_scf_if_disabled(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scale_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale_wait = ttl.cb_wait %scale_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale = ttl.attach_cb %scale_wait, %scale_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.reduce' op result is stored from a different block and cannot be rematerialized without changing the producer placement; enable compiler DFBs or store the intermediate to a user-declared DFB before the control-flow split}}
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

// Cheap sibling-if fanout is not rematerialized because the stores are not
// contained by one suitable if/else region. It still needs compiler storage.

func.func @sibling_if_store_fanout_disabled(%cond_a: i1, %cond_b: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.exp' op result is stored from a different block and cannot be rematerialized without changing the producer placement; enable compiler DFBs or store the intermediate to a user-declared DFB before the control-flow split}}
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
