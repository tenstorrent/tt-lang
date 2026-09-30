// Compiler-created DFB access-order checks remain enabled in both modes.
// RUN: ttlang-opt %s --ttl-insert-cb-sync --split-input-file --verify-diagnostics
// RUN: ttlang-opt %s --ttl-insert-cb-sync='sync-user-dfbs=false' --split-input-file --verify-diagnostics

func.func @compiler_release_before_store(
    %input_value: tensor<1x1x!ttcore.tile<32x32, bf16>>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %compiler_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  scf.if %condition {
    %slot = ttl.cb_reserve %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
    ttl.cb_push %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.store %input_value, %slot : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
