// Verifies tensor-backing identity and alias diagnostics before allocation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

// One logical DFB requires the same tensor backing in every kernel.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{logical DFB 0 has inconsistent tensor backing across kernel functions}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Partially overlapping ranges cannot describe distinct physical DFBs on one
// launch node.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @partial_overlap()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{tensor-backed DFB byte ranges partially overlap on a shared launch node}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Identical ranges on one launch node require one physical DFB identity.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @duplicate_range()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{identical tensor-backed DFB ranges require one proven shared physical index on a shared launch node}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
