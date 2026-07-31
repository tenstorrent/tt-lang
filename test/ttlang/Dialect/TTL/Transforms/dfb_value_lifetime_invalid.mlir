// Tests definite malformed DFB lifecycles rejected before availability
// analysis. Each split violates one producer or consumer condition.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-dfb-value-lifetimes))' --verify-diagnostics

// A consumer release cannot exist without a consumer acquisition.
func.func @pop_without_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer release has no same-kind acquisition in the enclosing kernel}}
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// A producer release cannot exist without a producer acquisition.
func.func @push_without_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer release has no same-kind acquisition in the enclosing kernel}}
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// An entry-block release cannot consume an acquisition that occurs later.
func.func @release_before_acquisition()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer release exceeds preceding entry-block acquisitions}}
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// An entry-block release cannot consume more tiles than all preceding
// acquisitions provide.
func.func @release_exceeds_acquisitions()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %first = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{dataflow buffer release exceeds preceding entry-block acquisitions}}
  ttl.cb_pop %dfb {num_tiles = 3 : i64}
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  return
}

// -----

// A nested acquisition executes after an entry-block release, if it executes
// at all, and therefore cannot supply tiles retroactively.
func.func @release_before_nested_acquisition(%upper_bound: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  // expected-error @below {{dataflow buffer release exceeds preceding entry-block acquisitions}}
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.for %iteration = %zero to %upper_bound step %one {
    %input = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
