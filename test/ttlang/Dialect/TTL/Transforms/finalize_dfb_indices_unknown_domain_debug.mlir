// Tests possible-domain lifetime diagnostics for an unknown launch-node domain.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' > %t.no-report.mlir 2> %t.no-report.log
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices > %t.report.mlir 2> %t.report.log
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices > %t.repeat.mlir 2> %t.repeat.log
// RUN: diff %t.no-report.mlir %t.report.mlir
// RUN: diff %t.report.mlir %t.repeat.mlir
// RUN: diff %t.report.log %t.repeat.log
// RUN: FileCheck %s < %t.report.log
// RUN: FileCheck %s --check-prefix=NO-REPORT --allow-empty < %t.no-report.log

// CHECK: DFB allocation liveness report
// CHECK: DFB logical_id=0 bounded=0 compiler_created=0
// CHECK-SAME: access_completion_proven=0
// CHECK-SAME: domain=unknown
// CHECK: access 0 effect=reserve tiles=1 sequence=0 domain=unknown
// CHECK: access 4 effect=none tiles=0 sequence=0 opaque_external=1 domain=unknown
// CHECK-SAME: operation=ttl.opaque_call kernel=@unknown_domain
// CHECK-SAME: unresolved_at=arith.cmpi kernel=@unknown_domain
// CHECK: possible_nodes lifecycle_completion=missing-protocol-effect domain_assumption=unknown-possible may_be_active=1 conditional_execution=0 node_count=10 exemplar=(0,0)
// CHECK-SAME: occurrences=[0:unresolved, 1:unresolved, 2:unresolved, 3:unresolved, 4:unresolved]
// CHECK: DFB logical_id=1 bounded=0 compiler_created=0
// CHECK-SAME: access_completion_proven=0
// CHECK-SAME: domain=unknown
// CHECK: possible_nodes lifecycle_completion=missing-protocol-effect domain_assumption=unknown-possible may_be_active=1 conditional_execution=0 node_count=10 exemplar=(0,0)
// CHECK-SAME: occurrences=[0:unresolved]
// CHECK: DFB logical_id=2 bounded=0 compiler_created=0
// CHECK-SAME: access_completion_proven=1
// CHECK-SAME: domain=unknown
// CHECK: possible_nodes lifecycle_completion=unsupported-control-flow domain_assumption=unknown-possible may_be_active=1 conditional_execution=0 node_count=1 nodes={(0,0)}
// CHECK-SAME: occurrences=[0:unresolved, 1:unresolved, 2:unresolved, 3:unresolved]
// CHECK: possible_nodes lifecycle_completion=complete domain_assumption=unknown-possible may_be_active=0 conditional_execution=0 node_count=9 exemplar=(1,0)
// CHECK-SAME: occurrences=[0:0, 1:0, 2:0, 3:0]
// CHECK: DFB logical_id=3 bounded=0 compiler_created=0
// CHECK-SAME: access_completion_proven=0
// CHECK-SAME: domain=unknown
// CHECK: possible_nodes lifecycle_completion=missing-protocol-effect domain_assumption=unknown-possible may_be_active=1 conditional_execution=0 node_count=10 exemplar=(0,0)
// CHECK-SAME: occurrences=[0:unresolved, 1:1, 2:1, 3:1, 4:1]
// CHECK-SAME: transactions=[]
// CHECK: DFB conflict lhs=0 rhs=1 reason=unknown-launch-node-domain node=none
// CHECK: DFB allocation liveness report end
// CHECK-NEXT: Total DFB count: 4
// CHECK-NEXT: DFB assignment: logical DFB 0 -> physical index 0 (unbounded)
// CHECK-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (unbounded)
// CHECK-NEXT: DFB assignment: logical DFB 2 -> physical index 2 (unbounded)
// CHECK-NEXT: DFB assignment: logical DFB 3 -> physical index 3 (unbounded)

// NO-REPORT-NOT: DFB allocation liveness report

module attributes {ttl.launch_grid = [10 : i64, 1 : i64]} {
  func.func @unknown_domain(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %missing_effect_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %is_first_node = arith.cmpi eq, %core_x, %c0 : index
    %sum = arith.addi %core_x, %offset : index
    %unknown_condition = arith.cmpi eq, %sum, %c0 : index
    scf.if %unknown_condition {
      ttl.opaque_call "custom_use" (%missing_effect_dfb)
          {header = "custom_use.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
      scf.if %is_first_node {
        %reserved = ttl.cb_reserve %dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<1x16, bf16>>
        ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        %available = ttl.cb_wait %dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<1x16, bf16>>
        ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      }
      ttl.opaque_call "custom_use" (%dfb) {header = "custom_use.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }

  func.func @zero_edge_producer(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %zero_edge_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %is_first_node = arith.cmpi eq, %core_x, %c0 : index
    %sum = arith.addi %core_x, %offset : index
    %unknown_condition = arith.cmpi eq, %sum, %c0 : index
    scf.if %unknown_condition {
      ttl.opaque_call "custom_use" (%target_dfb) {header = "custom_use.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %is_first_node {
      scf.if %unknown_condition {
        %reserved = ttl.cb_reserve %zero_edge_dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<1x16, bf16>>
        ttl.cb_push %zero_edge_dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      }
    }
    return
  }

  func.func @zero_edge_consumer(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %zero_edge_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %target_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %is_first_node = arith.cmpi eq, %core_x, %c0 : index
    %sum = arith.addi %core_x, %offset : index
    %unknown_condition = arith.cmpi eq, %sum, %c0 : index
    scf.if %is_first_node {
      scf.if %unknown_condition {
        %available = ttl.cb_wait %zero_edge_dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<1x16, bf16>>
        ttl.cb_pop %zero_edge_dfb
            : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      }
    }
    %reserved = ttl.cb_reserve %target_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %target_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %available = ttl.cb_wait %target_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %target_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
