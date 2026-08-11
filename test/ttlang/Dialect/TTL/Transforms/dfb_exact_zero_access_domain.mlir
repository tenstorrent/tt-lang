// Tests exact-zero refinement of otherwise unknown DFB access domains.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=REPORT

// An access that executes zero times on every launch node has an exact empty
// domain and may share a compatible scratch descriptor with an unknown DFB.

// REUSE-LABEL: func.func @all_nodes_inactive
// REUSE: %[[INACTIVE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: %[[UNKNOWN:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

// REPORT: DFB logical_id=0 bounded=0 compiler_created=0
// REPORT-SAME: domain={}
// REPORT: access 0 effect=none tiles=0 sequence=0 domain={}
// REPORT: DFB logical_id=1 bounded=0 compiler_created=0
// REPORT-SAME: domain=unknown
// REPORT-NOT: DFB conflict lhs=0 rhs=1

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @all_nodes_inactive(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %inactive_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unknown_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %two : index
    %inactive_condition = arith.andi %runtime_condition, %outside_grid : i1
    scf.if %inactive_condition {
      ttl.opaque_call "inactive_access" (%inactive_dfb)
          {header = "inactive_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %runtime_condition {
      ttl.opaque_call "unknown_access" (%unknown_dfb)
          {header = "unknown_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Zero executions on only part of the launch grid do not establish an empty
// domain. Both unknown DFBs remain distinct.

// REUSE-LABEL: func.func @partially_inactive
// REUSE: %[[PARTIAL:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 10 : index}
// REUSE-NEXT: %[[PARTIAL_UNKNOWN:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 11 : index}

// REPORT: DFB logical_id=10 bounded=0 compiler_created=0
// REPORT-SAME: domain=unknown
// REPORT: DFB conflict lhs=10 rhs=11 reason=unknown-launch-node-domain

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @partially_inactive(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %partial_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unknown_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %first_node = arith.cmpi eq, %core_x, %zero : index
    %partial_condition = arith.andi %runtime_condition, %first_node : i1
    scf.if %partial_condition {
      ttl.opaque_call "partial_access" (%partial_dfb)
          {header = "partial_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %runtime_condition {
      ttl.opaque_call "unknown_access" (%unknown_dfb)
          {header = "unknown_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// A nonzero exact count prevents empty-domain refinement even when launch-node
// domain analysis cannot otherwise resolve the condition.

// REUSE-LABEL: func.func @all_nodes_nonzero
// REUSE: %[[NONZERO:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 12 : index}
// REUSE-NEXT: %[[NONZERO_UNKNOWN:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 13 : index}

// REPORT: DFB logical_id=12 bounded=0 compiler_created=0
// REPORT-SAME: domain=unknown
// REPORT: DFB conflict lhs=12 rhs=13 reason=unknown-launch-node-domain

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @all_nodes_nonzero(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %nonzero_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 12 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unknown_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 13 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %inside_grid = arith.cmpi ne, %core_x, %two : index
    %nonzero_condition = arith.ori %runtime_condition, %inside_grid : i1
    scf.if %nonzero_condition {
      ttl.opaque_call "nonzero_access" (%nonzero_dfb)
          {header = "nonzero_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %runtime_condition {
      ttl.opaque_call "unknown_access" (%unknown_dfb)
          {header = "unknown_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Exact inactivity does not override CircularBufferType compatibility.

// REUSE-LABEL: func.func @descriptor_mismatch
// REUSE: %[[INACTIVE_NARROW:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 20 : index}
// REUSE-NEXT: %[[UNKNOWN_WIDE:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 21 : index}

// REPORT: DFB logical_id=20 bounded=0 compiler_created=0
// REPORT-SAME: domain={}
// REPORT: DFB conflict lhs=20 rhs=21 reason=descriptor-mismatch

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @descriptor_mismatch(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %inactive_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 20 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unknown_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %two : index
    %inactive_condition = arith.andi %runtime_condition, %outside_grid : i1
    scf.if %inactive_condition {
      ttl.opaque_call "inactive_access" (%inactive_dfb)
          {header = "inactive_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %runtime_condition {
      ttl.opaque_call "unknown_access" (%unknown_dfb)
          {header = "unknown_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Empty scratch domains remain distinct from tensor-backed descriptors until
// tensor-backed empty-domain allocation is defined by issue #813.

// REUSE-LABEL: func.func @tensor_backing_is_distinct
// REUSE: %[[INACTIVE_SCRATCH:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 30 : index}
// REUSE-NEXT: %[[TENSOR_BACKED:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 31 : index, tensor_backing = #ttl.tensor_backing

// REPORT: DFB logical_id=30 bounded=0 compiler_created=0
// REPORT-SAME: domain={}
// REPORT: DFB conflict lhs=30 rhs=31 reason=storage-mismatch node=none

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @tensor_backing_is_distinct(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %inactive_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 30 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %tensor_backed_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 31 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "tensor_access" (%tensor_backed_dfb)
        {header = "tensor_access.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %two : index
    %inactive_condition = arith.andi %runtime_condition, %outside_grid : i1
    scf.if %inactive_condition {
      ttl.opaque_call "inactive_access" (%inactive_dfb)
          {header = "inactive_access.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}
