// Tests typed DFB allocation groups with compiler-proved lifecycle handoff.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// One allocation group requires two sequential DFBs with different logical
// block shapes, block counts, and transaction sizes to use one physical index.
// The runtime descriptor uses the larger total capacity, and the cumulative
// cursor remains within that envelope.

// CHECK: module attributes {ttl.dfb_allocations = [{block_count = 4 : i32, dfb_index = 0 : i32
// CHECK-LABEL: func.func @capacity_envelope
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 4} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}

// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=8192 handoff=proven removed_conflicts=[descriptor-mismatch(0,1)]
// REPORT-NOT: DFB conflict lhs=0 rhs=1

module {
  func.func @capacity_envelope()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 4}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %first_producer = ttl.cb_reserve %first
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %first_consumer = ttl.cb_wait %first
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %second_producer = ttl.cb_reserve %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second_consumer = ttl.cb_wait %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}

// -----

// Three sequential members receive unique proof-derived ranks without using
// the happens-before relation as a sorting comparator. Declaration order is
// first, third, second; only execution order first, second, third avoids
// crossing the four-tile physical ring boundary (3 + 1 + 2 tiles).

// CHECK-LABEL: func.func @three_member_group
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 8 : index}
// CHECK-NEXT: %[[THIRD:.*]] = ttl.bind_cb{cb_index = 0, block_count = 4} {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 10 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 4} {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 9 : index}

// REPORT: DFB allocation group #ttl.dfb_allocation_group<8> members=[8, 10, 9] envelope_bytes=8192 handoff=proven
// REPORT-NOT: DFB conflict lhs=8 rhs=9
// REPORT-NOT: DFB conflict lhs=8 rhs=10
// REPORT-NOT: DFB conflict lhs=9 rhs=10

module {
  func.func @three_member_group()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 8 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %third = ttl.bind_cb {cb_index = 1, block_count = 4}
        {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second = ttl.bind_cb {cb_index = 2, block_count = 4}
        {allocation_group = #ttl.dfb_allocation_group<8>, dfb_id = 9 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %first_producer = ttl.cb_reserve %first {num_tiles = 3 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first {num_tiles = 3 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %first_consumer = ttl.cb_wait %first {num_tiles = 3 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first {num_tiles = 3 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %third_producer = ttl.cb_reserve %third {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %third {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %third_consumer = ttl.cb_wait %third {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %third {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}

// -----

// Group members on disjoint launch nodes share one index without a temporal
// ordering relation. Each node retains its own tensor-backed storage segment.

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\], \[1, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_segments = [{nodes = {{\[\[0, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}, {nodes = {{\[\[1, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}]}], ttl.launch_grid = array<i64: 2, 1>}
// CHECK-LABEL: func.func @disjoint_node_group
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 6 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 7 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1

// REPORT: DFB allocation group #ttl.dfb_allocation_group<4> members=[6, 7] envelope_bytes=64 handoff=proven removed_conflicts=[]
// REPORT-NOT: DFB conflict lhs=6 rhs=7

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @disjoint_node_group()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 6 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 7 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_node_zero = arith.cmpi eq, %core_x, %zero : index
    %is_node_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_node_zero {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %is_node_one {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// Exact tensor-backed declarations may use a group when they retain the same
// storage range and descriptor across sequential lifecycles.

// CHECK-LABEL: func.func @tensor_backed_group
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 4 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 5 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @tensor_backed_group()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 4 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 5 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Distinct group identities do not request co-allocation. Their overlapping
// lifecycles retain an interference edge and receive distinct indices.

// CHECK-LABEL: func.func @distinct_groups
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 3 : index}

module {
  func.func @distinct_groups()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
