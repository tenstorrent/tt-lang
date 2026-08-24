// Tests explicit allocation groups accepted by the unsafe handoff policy.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' 2>%t.warning | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// The source declares one must-alias allocation group but omits consumer
// effects and a synchronization relation between its producer lifecycles.

// CHECK: module attributes {ttl.assumed_dfb_allocation_groups = [{allocation_group = #ttl.dfb_allocation_group<0>, assumptions = [{lhs = 0 : i64, reason = "access-completion-not-proven", rhs = 1 : i64}, {lhs = 0 : i64, reason = "unproven-cursor-order", rhs = 1 : i64}], members = [0, 1]}], ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32
// CHECK-LABEL: func.func @assumed_handoff
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}

// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<0> members=[0, 1] without compiler proof: access-completion-not-proven(0,1), unproven-cursor-order(0,1)

module {
  func.func @assumed_handoff()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
