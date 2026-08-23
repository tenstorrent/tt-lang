// Verifies selected reset writes conflict with live non-target DFB aliases.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// The selected reset of logical DFB 0 executes on both launch nodes. Logical
// DFB 1 is live across that boundary on node (1,0), so the two exact-domain
// lifecycles require distinct physical indices.
// An inactive wait precedes the active reserve under one projected scf.if.
// Conflict evidence must name the active reserve.
// CHECK: DFB conflict lhs=0 rhs=1 reason=reset-domain-write node=(1,0)
// CHECK-SAME: lhs_operation=ttl.reset_dfbs
// CHECK-SAME: rhs_operation=ttl.cb_reserve
// CHECK: DFB assignment: logical DFB 0 -> physical index 0
// CHECK-NEXT: DFB assignment: logical DFB 1 -> physical index 1

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_alias_domain">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_core_zero = arith.cmpi eq, %core_x, %zero : index
    %is_core_one = arith.cmpi eq, %core_x, %one : index
    %is_outside_grid = arith.cmpi eq, %core_x, %two : index
    scf.if %is_core_zero {
      %slot = ttl.cb_reserve %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.if %is_core_one {
      scf.if %is_outside_grid {
        %inactive = ttl.cb_wait %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      %slot = ttl.cb_reserve %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_alias_domain">, <kind = data_movement, identity = "reader", operation = "reset_alias_domain">, <kind = data_movement, identity = "writer", operation = "reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_alias_domain">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_core_zero = arith.cmpi eq, %core_x, %zero : index
    %is_core_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_core_zero {
      %slot = ttl.cb_wait %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %selected
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_alias_domain">, <kind = data_movement, identity = "reader", operation = "reset_alias_domain">, <kind = data_movement, identity = "writer", operation = "reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    scf.if %is_core_one {
      %slot = ttl.cb_wait %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_alias_domain">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_alias_domain">, <kind = data_movement, identity = "reader", operation = "reset_alias_domain">, <kind = data_movement, identity = "writer", operation = "reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// The DFBs have complementary possible launch nodes, so ordinary possible-
// domain analysis permits sharing. The reset of DFB 0 still writes its index
// on node (0,0) while DFB 1 is active there. A zero-trip wait shares DFB 1's
// projected producer event but cannot supply evidence.
// CHECK: DFB logical_id=0 bounded=0 compiler_created=0 conditionally_bounded=1
// CHECK: possible_nodes quiescence=none domain_assumption=unknown-possible
// CHECK: DFB logical_id=1 bounded=0 compiler_created=0 conditionally_bounded=1
// CHECK: possible_nodes quiescence=none domain_assumption=unknown-possible
// CHECK: DFB conflict lhs=0 rhs=1 reason=reset-domain-write node=(0,0)
// CHECK-SAME: lhs_operation=ttl.reset_dfbs
// CHECK-SAME: rhs_operation=ttl.cb_reserve

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @possible_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "possible_reader", operation = "possible_reset_alias_domain">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_i64 = arith.constant 0 : i64
    %condition_value = ttl.opaque_call "reader_condition" (%core_x)
        {condition_result = #ttl.dispatch_condition<0, i64>,
         header = "condition.hpp"} : (index) -> i64
    %condition = arith.cmpi ne, %condition_value, %zero_i64 : i64
    %is_core_zero = arith.cmpi eq, %core_x, %zero : index
    %is_core_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_core_one {
      scf.if %condition {
        %reserved = ttl.cb_reserve %selected
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %selected
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    scf.if %is_core_zero {
      scf.for %inactive_iteration = %zero to %zero step %one {
        %inactive = ttl.cb_wait %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      scf.if %condition {
        %reserved = ttl.cb_reserve %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "possible_compute", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_reader", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_writer", operation = "possible_reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @possible_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "possible_compute", operation = "possible_reset_alias_domain">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_i64 = arith.constant 0 : i64
    %condition_value = ttl.opaque_call "compute_condition" (%core_x)
        {condition_result = #ttl.dispatch_condition<0, i64>,
         header = "condition.hpp"} : (index) -> i64
    %condition = arith.cmpi ne, %condition_value, %zero_i64 : i64
    %is_core_zero = arith.cmpi eq, %core_x, %zero : index
    %is_core_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_core_one {
      scf.if %condition {
        %available = ttl.cb_wait %selected
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %selected
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "possible_compute", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_reader", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_writer", operation = "possible_reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    scf.if %is_core_zero {
      scf.if %condition {
        %available = ttl.cb_wait %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %crossing
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    return
  }

  func.func @possible_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "possible_writer", operation = "possible_reset_alias_domain">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "possible_compute", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_reader", operation = "possible_reset_alias_domain">, <kind = data_movement, identity = "possible_writer", operation = "possible_reset_alias_domain">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
