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

// Exact-empty tensor-backed domains retain the existing issue #813 rejection.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @empty_tensor_backing(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    // expected-error @below {{tensor-backed physical DFB requires an exact non-empty launch-node domain}}
    %tensor_backed_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 2 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %one : index
    %inactive_condition = arith.andi %runtime_condition, %outside_grid : i1
    scf.if %inactive_condition {
      ttl.opaque_call "inactive_tensor_access" (%tensor_backed_dfb)
          {header = "inactive_tensor_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// An analysis-only representative node cannot define tensor-backed residency.
module {
  func.func @unknown_grid_tensor_backing()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    // expected-error @below {{tensor-backed physical DFB requires an exact non-empty launch-node domain}}
    %tensor_backed_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "access" (%tensor_backed_dfb)
        {header = "access.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
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
