// Tests exact access-domain refinement from per-node execution counts.
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

// Exact positive counts on every launch node establish the full launch domain
// even when launch-node domain analysis cannot resolve the condition.

// REUSE-LABEL: func.func @all_nodes_nonzero
// REUSE: %[[NONZERO:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 12 : index}
// REUSE-NEXT: %[[NONZERO_UNKNOWN:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 13 : index}

// REPORT: DFB logical_id=12 bounded=0 compiler_created=0
// REPORT-SAME: domain={(0,0), (1,0)}
// REPORT: access 0 effect=none tiles=0 sequence=0 domain={(0,0), (1,0)}
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

// Per-node counts establish disjoint strict subsets. The DFBs therefore share
// one physical index without a local lifetime-order proof.

// REUSE-LABEL: func.func @exact_disjoint_subsets_reuse
// REUSE: %[[FIRST_SUBSET:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 14 : index}
// REUSE-NEXT: %[[SECOND_SUBSET:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 15 : index}

// REPORT: DFB logical_id=14 bounded=0 compiler_created=0
// REPORT-SAME: domain={(0,0)}
// REPORT: access 0 effect=none tiles=0 sequence=0 domain={(0,0)}
// REPORT: DFB logical_id=15 bounded=0 compiler_created=0
// REPORT-SAME: domain={(1,0)}
// REPORT-NOT: DFB conflict lhs=14 rhs=15
// REPORT: DFB assignment: logical DFB 14 -> physical index 0 (unbounded)
// REPORT-NEXT: DFB assignment: logical DFB 15 -> physical index 0 (unbounded)

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @exact_disjoint_subsets_reuse(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 14 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 15 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %first_runtime = arith.cmpi eq, %first_runtime_sum, %zero : index
    %first_node = arith.cmpi eq, %core_x, %zero : index
    %first_masked_runtime = arith.andi %first_runtime, %first_node : i1
    %first_condition = arith.ori %first_masked_runtime, %first_node : i1
    scf.if %first_condition {
      ttl.opaque_call "first" (%first)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    %second_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %second_runtime = arith.cmpi eq, %second_runtime_sum, %zero : index
    %one = arith.constant 1 : index
    %second_node = arith.cmpi eq, %core_x, %one : index
    %second_masked_runtime = arith.andi %second_runtime, %second_node : i1
    %second_condition = arith.ori %second_masked_runtime, %second_node : i1
    scf.if %second_condition {
      ttl.opaque_call "second" (%second)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// One unresolved access prevents the union for its logical DFB from becoming
// exact, even when the complete protocol access has an exact domain.

// REUSE-LABEL: func.func @one_unresolved_access
// REUSE: %[[MIXED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 16 : index}
// REUSE-NEXT: %[[EXACT:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 17 : index}

// REPORT: DFB logical_id=16 bounded=0 compiler_created=0
// REPORT-SAME: domain=unknown
// REPORT: access 0 effect=reserve tiles=1 sequence=0 domain={(0,0), (1,0)}
// REPORT: access 4 effect=none tiles=0 sequence=0 domain=unknown
// REPORT: DFB logical_id=17 bounded=1 compiler_created=0
// REPORT-SAME: domain={(0,0), (1,0)}
// REPORT: DFB conflict lhs=16 rhs=17 reason=unknown-launch-node-domain

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @one_unresolved_access(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %mixed = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 16 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %exact = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 17 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %known_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %known_runtime = arith.cmpi eq, %known_runtime_sum, %zero : index
    %inside_grid = arith.cmpi ne, %core_x, %two : index
    %known_condition = arith.ori %known_runtime, %inside_grid : i1
    scf.if %known_condition {
      ttl.opaque_call "mixed_protocol" dfb_dependencies(%mixed : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    %unknown_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %unknown_condition = arith.cmpi eq, %unknown_runtime_sum, %zero : index
    scf.if %unknown_condition {
      ttl.opaque_call "mixed_unknown_access" (%mixed)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    ttl.opaque_call "exact" dfb_dependencies(%exact : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A count greater than one establishes domain membership but does not turn a
// repeated protocol into a single bounded lifecycle.

// REUSE-LABEL: func.func @repeated_exact_count
// REUSE: %[[REPEATED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 18 : index}
// REUSE-NEXT: %[[AFTER_REPEATED:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 19 : index}

// REPORT: DFB logical_id=18 bounded=0 compiler_created=0
// REPORT-SAME: domain={(0,0), (1,0)}
// REPORT: node (0,0) quiescence=unsupported-control-flow
// REPORT-SAME: occurrences=[0:2, 1:2, 2:2, 3:2]
// REPORT: DFB conflict lhs=18 rhs=19 reason=unproven-quiescence node=(0,0)

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @repeated_exact_count(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %repeated = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 18 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %after_repeated = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 19 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    %inside_grid = arith.cmpi ne, %core_x, %two : index
    %known_condition = arith.ori %runtime_condition, %inside_grid : i1
    scf.if %known_condition {
      scf.for %iteration = %zero to %two step %one {
        ttl.opaque_call "repeated" dfb_dependencies(%repeated : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
    }
    ttl.opaque_call "after_repeated" dfb_dependencies(%after_repeated : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Refinement preserves domains that launch-node analysis already proved.

// REUSE-LABEL: func.func @known_domain_is_preserved
// REUSE: %[[KNOWN_FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 22 : index}
// REUSE-NEXT: %[[KNOWN_SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 23 : index}

// REPORT: DFB logical_id=22 bounded=1 compiler_created=0
// REPORT-SAME: domain={(0,0), (1,0)}
// REPORT: access 0 effect=reserve tiles=1 sequence=0 domain={(0,0), (1,0)}
// REPORT: DFB logical_id=23 bounded=1 compiler_created=0
// REPORT-SAME: domain={(0,0), (1,0)}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @known_domain_is_preserved()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 22 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 23 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A generic external storage access receives the same exact-domain refinement
// before it is conservatively attached to every user-managed DFB.

// REUSE-LABEL: func.func @generic_access_exact_subset
// REUSE: %[[GENERIC_SUBSET:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 24 : index}

// REPORT: DFB logical_id=24 bounded=0 compiler_created=0
// REPORT-SAME: domain={(0,0)}
// REPORT: access 0 effect=none tiles=0 sequence=0 domain={(0,0)}
// REPORT: access 1 effect=none tiles=0 sequence=0 domain={(0,0)}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @generic_access_exact_subset(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %subset = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 24 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %direct_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %direct_runtime = arith.cmpi eq, %direct_runtime_sum, %zero : index
    %direct_node = arith.cmpi eq, %core_x, %zero : index
    %direct_masked_runtime = arith.andi %direct_runtime, %direct_node : i1
    %direct_condition = arith.ori %direct_masked_runtime, %direct_node : i1
    scf.if %direct_condition {
      ttl.opaque_call "direct_access" (%subset)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    %generic_runtime_sum = arith.addi %core_x, %runtime_offset : index
    %generic_runtime = arith.cmpi eq, %generic_runtime_sum, %zero : index
    %generic_node = arith.cmpi eq, %core_x, %zero : index
    %generic_masked_runtime = arith.andi %generic_runtime, %generic_node : i1
    %generic_condition = arith.ori %generic_masked_runtime, %generic_node : i1
    scf.if %generic_condition {
      ttl.opaque_call "generic_access" ()
          {header = "effects.hpp", unknown_dfb_access} : () -> ()
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
