// Verifies that allocation failure leaves provisional DFB indices and kernel
// configuration unchanged and does not create runtime metadata.
// RUN: ttlang-opt %s --verify-diagnostics --mlir-print-ir-after=ttl-finalize-dfb-indices --mlir-print-ir-after-failure -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' 2>&1 | FileCheck %s --implicit-check-not=ttl.compiler_allocated_dfbs

// expected-error @below {{need 33 DFB indices but hardware supports at most 32 (1 compiler-allocated after reuse)}}
module {
  func.func @user_index_31()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %user_dfb = ttl.bind_cb {cb_index = 31, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  // CHECK-LABEL: func.func @compiler_dfb
  // CHECK-SAME: ttl.base_cta_index = 1 : i32
  // CHECK: ttl.bind_cb{cb_index = 0, block_count = 1} {ttl.compiler_allocated}
  func.func @compiler_dfb()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32} {
    %compiler_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %reserve = ttl.cb_reserve %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %wait = ttl.cb_wait %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
