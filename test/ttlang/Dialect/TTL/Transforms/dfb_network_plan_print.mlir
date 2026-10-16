// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-print-dfb-network-plan)' -o /dev/null 2>&1 | FileCheck %s

// Summary: Verifies the printed DFB network plan: each admitted network with
// the bindings and kernel-side ops of every DFB its records name, networks in
// module order including one in a nested module, the unsupported features of
// a rejected plan, and a module without networks.

// Test that a fork's source and outputs resolve to the kernel ops on the ends
// the kernels keep: the source's producer and each output's consumer.
// CHECK-LABEL: DFB network plan: planned
// CHECK-NEXT:    network @copy
// CHECK-NEXT:      ttl.dfb.fork @fork
// CHECK-NEXT:        DFB 0 source: bindings=[@reader] kernel-ops=[@reader ttl.cb_reserve, @reader ttl.cb_push]
// CHECK-NEXT:        DFB 1 output: bindings=[@writer] kernel-ops=[@writer ttl.cb_wait, @writer ttl.cb_pop]
// CHECK-NEXT:        DFB 2 output: bindings=[@writer] kernel-ops=[]
module {
  ttl.dfb.network @copy {
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }

  func.func @reader() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @writer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that networks are listed in module order, including one in a nested
// module.
// CHECK-LABEL: DFB network plan: planned
// CHECK-NEXT:    network @plan_net_1
// CHECK-NEXT:    network @plan_net_nested
module {
  ttl.dfb.network @plan_net_1 {}
  module @inner {
    ttl.dfb.network @plan_net_nested {}
  }
}

// -----

// Test that a rejected plan lists every unsupported feature.
// CHECK-LABEL: DFB network plan: rejected
// CHECK-NEXT:    @two_records: networks with more than one record are not supported yet; connect the operators through a kernel
// CHECK-NEXT:    shared fork storage is not supported yet; use storage = replicated
module {
  ttl.dfb.network @two_records {
    ttl.dfb.fork @first 0 : index -> [1, 2] storage = <replicated>
    ttl.dfb.fork @second 3 : index -> [4, 5] storage = <shared>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb3 = ttl.bind_cb{cb_index = 3, block_count = 2} {dfb_id = 3 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb4 = ttl.bind_cb{cb_index = 4, block_count = 2} {dfb_id = 4 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb5 = ttl.bind_cb{cb_index = 5, block_count = 2} {dfb_id = 5 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a module without networks prints an empty planned result.
// CHECK-LABEL: DFB network plan: planned
// CHECK-NOT:     network
module {
  func.func @no_network() {
    return
  }
}
