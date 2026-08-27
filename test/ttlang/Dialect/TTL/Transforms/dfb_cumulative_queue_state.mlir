// Tests cumulative readiness and independent DFB ring-pointer movement.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=ALLOC
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// Each eight-tile reservation permits two four-tile publications. Later
// reservations depend on consumer progress because the DFB holds 12 tiles.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4, 4, 4]

module {
  func.func @high_water_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "high_water_producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @explicit_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "explicit_consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Allocation-group cursor simulation preserves distinct write and read
// movements across member handoff. Eight tiles followed by four tiles wrap
// both pointers in one 12-tile physical DFB.

// ALLOC-LABEL: func.func @group_cursor_handoff
// ALLOC: ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// ALLOC-NEXT: ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=24576 handoff=proven
// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4] write_cursor_runs=[8] read_cursor_runs=[4, 4]

module {
  func.func @group_cursor_handoff()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// One publication may satisfy repeated readiness checks. The normalized
// sequence retains both consumer pointer movements while the write pointer
// moves once by eight tiles.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4] write_cursor_runs=[8] read_cursor_runs=[4, 4]

module {
  func.func @coalesced_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "coalesced_producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @subblock_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "subblock_consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// One high-water wait may protect two smaller pointer advancements.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4]

module {
  func.func @subblock_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "subblock_producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @high_water_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "high_water_consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Equal effect counts do not imply transaction pairing. Cumulative positions
// prove differently partitioned producer and consumer cursor movement.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4, 4] write_cursor_runs=[4, 8] read_cursor_runs=[8, 4]

module {
  func.func @differently_partitioned_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "differently_partitioned_producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @differently_partitioned_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "differently_partitioned_consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// The helper DFB orders two producer kernels for the cumulative DFB. The
// cumulative proof must observe that exact synchronization even though the
// cumulative DFB has the lower logical identity.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: node (0,0) lifecycle_completion=complete
// CHECK-SAME: transactions=[4, 4, 4] write_cursor_runs=[4, 8] read_cursor_runs=[8, 4]

module {
  func.func @split_producer_first()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %cumulative = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %ordering = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "split_producer_first"
        dfb_dependencies(
          %cumulative, %ordering
          : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>,
            !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 1, 4>,
                     #ttl.dfb_protocol_effect<push, 1, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @split_producer_second()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %cumulative = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %ordering = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "split_producer_second"
        dfb_dependencies(
          %cumulative, %ordering
          : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>,
            !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 1, 4>,
                     #ttl.dfb_protocol_effect<pop, 1, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @split_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %cumulative = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "split_consumer"
        dfb_dependencies(
          %cumulative : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}
