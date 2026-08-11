// Tests counterfactual lifetime diagnostics for an unknown launch-node domain.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s

// CHECK: DFB allocation liveness report
// CHECK: DFB logical_id=0 bounded=0 compiler_created=0
// CHECK-SAME: domain=unknown
// CHECK: access 0 effect=reserve tiles=1 sequence=0 domain=unknown
// CHECK: access 4 effect=none tiles=0 sequence=0 domain=unknown
// CHECK-SAME: operation=ttl.opaque_call kernel=@unknown_domain
// CHECK-SAME: unresolved_at=arith.cmpi kernel=@unknown_domain
// CHECK: diagnostic_nodes quiescence=unsupported-control-flow domain_assumption=all-unknown-active may_be_active=1 node_count=2 nodes={(0,0), (1,0)}
// CHECK-SAME: occurrences=[0:unresolved, 1:unresolved, 2:unresolved, 3:unresolved, 4:unresolved]
// CHECK: DFB logical_id=1 bounded=0 compiler_created=0
// CHECK-SAME: domain=unknown
// CHECK: diagnostic_nodes quiescence=missing-protocol-effect domain_assumption=all-unknown-active may_be_active=1 node_count=2 nodes={(0,0), (1,0)}
// CHECK-SAME: occurrences=[0:unresolved]
// CHECK: DFB conflict lhs=0 rhs=1 reason=unknown-launch-node-domain node=none
// CHECK: DFB allocation liveness report end
// CHECK-NEXT: Total DFB count: 2
// CHECK-NEXT: DFB assignment: logical DFB 0 -> physical index 0 (unbounded)
// CHECK-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (unbounded)

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
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
}
