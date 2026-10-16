// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-expand-dfb-networks | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-to-ttkernel-pipeline -o /dev/null

// Summary: Verifies that the expansion pass reports every DFB network as not
// supported yet, both when the pass runs alone and as the first pass of the
// TTL-to-TTKernel pipeline, and that a module without networks passes through
// unchanged.

// Test that each network in a module is reported, not only the first.
module {
  // expected-error @below {{'ttl.dfb.network' op @reject_net_1: DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_1 {}
  // expected-error @below {{'ttl.dfb.network' op @reject_net_2: DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_2 {}
}

// -----

// Test that a network with records, whose DFBs are all bound, is rejected.
module {
  // expected-error @below {{'ttl.dfb.network' op @reject_net_records: DFB networks are not supported yet}}
  ttl.dfb.network @reject_net_records {
    ttl.dfb.split @s 0 : index -> [1, 2] policy = <round_robin>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a network in a nested module is rejected.
module {
  module @inner {
    // expected-error @below {{'ttl.dfb.network' op @reject_net_nested: DFB networks are not supported yet}}
    ttl.dfb.network @reject_net_nested {}
  }
}

// -----

// Test that a fork with shared storage is not supported.
module {
  ttl.dfb.network @shared_fork {
    // expected-error @below {{'ttl.dfb.fork' op shared fork storage is not supported yet; use storage = replicated}}
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <shared>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a merge is not supported.
module {
  ttl.dfb.network @merge_net {
    // expected-error @below {{'ttl.dfb.merge' op merge records are not supported yet}}
    ttl.dfb.merge @merge [0 : index, 1 : index] policy = <round_robin>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a network with more than one record is not supported, and that
// every unsupported feature of one network is reported.
module {
  // expected-error @below {{'ttl.dfb.network' op @two_records: networks with more than one record are not supported yet; connect the operators through a kernel}}
  ttl.dfb.network @two_records {
    ttl.dfb.fork @first 0 : index -> [1, 2] storage = <replicated>
    // expected-error @below {{'ttl.dfb.fork' op auto fork storage is not supported yet; use storage = replicated}}
    ttl.dfb.fork @second 3 : index -> [4, 5] storage = <auto>
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

// Test that an allocation-grouped network DFB is not supported.
module {
  ttl.dfb.network @grouped {
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.bind_cb' op DFB 1 of network @grouped has an allocation group, which is not supported yet}}
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a native call on a network DFB is not supported.
module {
  ttl.dfb.network @native {
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op native calls on DFB 1 of network @native are not supported yet}}
    ttl.opaque_call "use_output" dfb_dependencies(
        %dfb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Test that a module without networks passes through unchanged.
// CHECK-LABEL: func.func @no_network
// CHECK-NEXT: return
module {
  func.func @no_network() {
    return
  }
}
