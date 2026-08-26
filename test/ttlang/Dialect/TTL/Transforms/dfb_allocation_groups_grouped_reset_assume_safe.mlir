// Tests grouped reset completion and unsafe assumptions for unresolved
// allocation-group lifecycles.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' > %t.output 2> %t.warning
// RUN: FileCheck %s --check-prefix=OUTPUT < %t.output
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// A reset expanded to every allocation-group member completes an exact untyped
// access before the reset without an unsafe assumption.

// OUTPUT-LABEL: func.func @exact_reader
// OUTPUT: %[[EXACT_SELECTED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// OUTPUT-NEXT: %[[EXACT_BEFORE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}

// WARNING-NOT: reset-domain-write
// WARNING-NOT: #ttl.dfb_allocation_group<0>

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @exact_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "exact_reset_completed">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "untyped_access" dfb_dependencies(%before : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "exact_reset_completed">, <kind = data_movement, identity = "reader", operation = "exact_reset_completed">, <kind = data_movement, identity = "writer", operation = "exact_reset_completed">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %selected_slot = ttl.cb_reserve %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @exact_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "exact_reset_completed">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "exact_reset_completed">, <kind = data_movement, identity = "reader", operation = "exact_reset_completed">, <kind = data_movement, identity = "writer", operation = "exact_reset_completed">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %selected_slot = ttl.cb_wait %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @exact_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "exact_reset_completed">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "exact_reset_completed">, <kind = data_movement, identity = "reader", operation = "exact_reset_completed">, <kind = data_movement, identity = "writer", operation = "exact_reset_completed">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// Incomplete access operations that execute before the first reset or after
// the last reset do not overlap any repeated interface write.

// OUTPUT-LABEL: func.func @repeated_outside_reader
// OUTPUT: %[[REPEATED_SELECTED_READER:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
// OUTPUT-NEXT: %[[REPEATED_BEFORE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 7 : index}
// OUTPUT-LABEL: func.func @repeated_outside_writer
// OUTPUT: %[[REPEATED_SELECTED_WRITER:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
// OUTPUT-NEXT: %[[REPEATED_AFTER:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 8 : index}

// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<3> members=[6, 7, 8] without compiler proof:
// WARNING-SAME: access-completion-not-proven
// WARNING-NOT: reset-domain-write

module attributes {
  ttl.launch_grid = array<i64: 1, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @repeated_outside_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "repeated_unproven_outside">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 7 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "untyped_access" dfb_dependencies(%before : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "effects.hpp"} : () -> ()
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_outside">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @repeated_outside_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "repeated_unproven_outside">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_outside">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @repeated_outside_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "repeated_unproven_outside">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %after = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 8 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "reader", operation = "repeated_unproven_outside">, <kind = data_movement, identity = "writer", operation = "repeated_unproven_outside">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    ttl.opaque_call "untyped_access" dfb_dependencies(%after : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A runtime predicate makes the untyped-access domain possible rather than exact.
// Its activity still remains before the reset on every possible launch node.

// OUTPUT-LABEL: func.func @possible_reader
// OUTPUT: %[[POSSIBLE_SELECTED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
// OUTPUT-NEXT: %[[POSSIBLE_BEFORE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}

// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<1> members=[2, 3] without compiler proof: unknown-launch-node-domain(2,3)
// WARNING-NOT: reset-domain-write

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @possible_reader(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "possible_incomplete_before">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %active = arith.cmpi eq, %runtime_sum, %zero : index
    %selected_active = arith.cmpi eq, %runtime_sum, %one : index
    scf.if %active {
      ttl.opaque_call "untyped_access" dfb_dependencies(%before : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "effects.hpp"} : () -> ()
    }
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "possible_incomplete_before">, <kind = data_movement, identity = "reader", operation = "possible_incomplete_before">, <kind = data_movement, identity = "writer", operation = "possible_incomplete_before">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    scf.if %selected_active {
      ttl.opaque_call "untyped_access" dfb_dependencies(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "effects.hpp"} : () -> ()
    }
    return
  }

  func.func @possible_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "possible_incomplete_before">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "possible_incomplete_before">, <kind = data_movement, identity = "reader", operation = "possible_incomplete_before">, <kind = data_movement, identity = "writer", operation = "possible_incomplete_before">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @possible_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "possible_incomplete_before">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "possible_incomplete_before">, <kind = data_movement, identity = "reader", operation = "possible_incomplete_before">, <kind = data_movement, identity = "writer", operation = "possible_incomplete_before">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// A complete transaction before the reset cancels on its own side. A later
// producer-consumer pair is externally conditioned to execute together, but
// the compiler cannot relate kernel arguments across the two participants.

// OUTPUT-LABEL: func.func @complete_before_reader
// OUTPUT: %[[COMPLETE_SELECTED:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
// OUTPUT-NEXT: %[[COMPLETE_CROSSING:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}

// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<2> members=[4, 5] without compiler proof: unknown-launch-node-domain(4,5)
// WARNING-NOT: reset-domain-write

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @complete_before_reader(%producer_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "complete_before_consumer_after">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "reader", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "writer", operation = "complete_before_consumer_after">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %producer_sum = arith.addi %core_x, %producer_offset : index
    %producer_active = arith.cmpi eq, %producer_sum, %zero : index
    scf.if %producer_active {
      %after_slot = ttl.cb_reserve %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    %selected_slot = ttl.cb_reserve %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @complete_before_compute(%consumer_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "complete_before_consumer_after">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %crossing = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before_slot = ttl.cb_wait %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %crossing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "reader", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "writer", operation = "complete_before_consumer_after">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %consumer_sum = arith.addi %core_x, %consumer_offset : index
    %consumer_active = arith.cmpi eq, %consumer_sum, %zero : index
    scf.if %consumer_active {
      %after_slot = ttl.cb_wait %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %crossing
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    %selected_slot = ttl.cb_wait %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %selected
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @complete_before_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "complete_before_consumer_after">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %selected = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "reader", operation = "complete_before_consumer_after">, <kind = data_movement, identity = "writer", operation = "complete_before_consumer_after">]>(%selected : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
