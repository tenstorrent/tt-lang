// Tests DFB protocol order through structured regions with exact invocation counts.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=REPORT

// Exact single-invocation conditions preserve each external effect event. The
// return transaction orders the two forward DFBs, and the second forward
// transaction orders the two return DFBs. An unrelated unresolved access in
// the same one-iteration loop remains conservative without discarding those
// exact events.

// REUSE: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32
// REUSE-LABEL: func.func @exact_nested_noc
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 3 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 4 : index}
// REUSE-LABEL: func.func @exact_nested_compute
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 3 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 4 : index}

// REPORT: DFB logical_id=0 bounded=1 compiler_created=0
// REPORT: DFB logical_id=1 bounded=1 compiler_created=0
// REPORT: DFB logical_id=2 bounded=1 compiler_created=0
// REPORT: DFB logical_id=3 bounded=1 compiler_created=0
// REPORT: DFB logical_id=4 bounded=0 compiler_created=0
// REPORT: DFB assignment: logical DFB 0 -> physical index 0 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 2 -> physical index 0 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 3 -> physical index 1 (bounded)

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @exact_nested_noc(%runtime_condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unresolved = ttl.bind_cb {cb_index = 4, block_count = 2}
        {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %active = arith.cmpi eq, %core_x, %zero : index
    scf.for %iteration = %zero to %one step %one {
      scf.if %active {
        ttl.opaque_call "nested_noc" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 1, 1>, #ttl.dfb_protocol_effect<pop, 1, 1>, #ttl.dfb_protocol_effect<reserve, 2, 1>, #ttl.dfb_protocol_effect<push, 2, 1>, #ttl.dfb_protocol_effect<wait, 3, 1>, #ttl.dfb_protocol_effect<pop, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
      }
      scf.if %runtime_condition {
        ttl.opaque_call "unresolved_noc" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] (%unresolved) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
      }
    }
    return
  }

  func.func @exact_nested_compute(%runtime_condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unresolved = ttl.bind_cb {cb_index = 4, block_count = 2}
        {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %active = arith.cmpi eq, %core_x, %zero : index
    scf.for %iteration = %zero to %one step %one {
      scf.if %active {
        ttl.opaque_call "nested_compute" dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 1, 1>, #ttl.dfb_protocol_effect<push, 1, 1>, #ttl.dfb_protocol_effect<wait, 2, 1>, #ttl.dfb_protocol_effect<pop, 2, 1>, #ttl.dfb_protocol_effect<reserve, 3, 1>, #ttl.dfb_protocol_effect<push, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
      }
      scf.if %runtime_condition {
        ttl.opaque_call "unresolved_compute" dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] (%unresolved) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
      }
    }
    return
  }
}

// -----

// Runtime conditions in separate kernel functions do not prove a common
// single invocation. Their nested effects retain top-level projection.

// REUSE-LABEL: func.func @unknown_nested_noc
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 10 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 11 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 12 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 3, block_count = 2} {dfb_id = 13 : index}
// REPORT: DFB logical_id=10 bounded=0 compiler_created=0
// REPORT: DFB logical_id=11 bounded=0 compiler_created=0
// REPORT: DFB logical_id=12 bounded=0 compiler_created=0
// REPORT: DFB logical_id=13 bounded=0 compiler_created=0

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @unknown_nested_noc(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    scf.if %condition {
      ttl.opaque_call "unknown_nested_noc" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 1, 1>, #ttl.dfb_protocol_effect<pop, 1, 1>, #ttl.dfb_protocol_effect<reserve, 2, 1>, #ttl.dfb_protocol_effect<push, 2, 1>, #ttl.dfb_protocol_effect<wait, 3, 1>, #ttl.dfb_protocol_effect<pop, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }

  func.func @unknown_nested_compute(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    scf.if %condition {
      ttl.opaque_call "unknown_nested_compute" dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 1, 1>, #ttl.dfb_protocol_effect<push, 1, 1>, #ttl.dfb_protocol_effect<wait, 2, 1>, #ttl.dfb_protocol_effect<pop, 2, 1>, #ttl.dfb_protocol_effect<reserve, 3, 1>, #ttl.dfb_protocol_effect<push, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Two loop iterations prevent single-invocation region order. The four DFBs
// therefore retain distinct physical indices.

// REUSE-LABEL: func.func @repeated_nested_noc
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 20 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 21 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 22 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 3, block_count = 2} {dfb_id = 23 : index}
// REPORT: DFB logical_id=20 bounded=0 compiler_created=0
// REPORT: DFB logical_id=21 bounded=0 compiler_created=0
// REPORT: DFB logical_id=22 bounded=0 compiler_created=0
// REPORT: DFB logical_id=23 bounded=0 compiler_created=0

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @repeated_nested_noc()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    scf.for %iteration = %zero to %two step %one {
      ttl.opaque_call "repeated_nested_noc" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 1, 1>, #ttl.dfb_protocol_effect<pop, 1, 1>, #ttl.dfb_protocol_effect<reserve, 2, 1>, #ttl.dfb_protocol_effect<push, 2, 1>, #ttl.dfb_protocol_effect<wait, 3, 1>, #ttl.dfb_protocol_effect<pop, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }

  func.func @repeated_nested_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %forward_one = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_one = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %forward_two = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %return_two = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    scf.for %iteration = %zero to %two step %one {
      ttl.opaque_call "repeated_nested_compute" dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 1, 1>, #ttl.dfb_protocol_effect<push, 1, 1>, #ttl.dfb_protocol_effect<wait, 2, 1>, #ttl.dfb_protocol_effect<pop, 2, 1>, #ttl.dfb_protocol_effect<reserve, 3, 1>, #ttl.dfb_protocol_effect<push, 3, 1>] (%forward_one, %return_one, %forward_two, %return_two) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}
