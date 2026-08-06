// Summary: Verify repeated scalar PipeNet transfers become bounded grouped
// transfers while preserving scalar residual execution.
// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s --check-prefix=AUTO
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=4' | FileCheck %s --check-prefix=BOUND
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=8' | FileCheck %s --check-prefix=UPPER
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=1' | FileCheck %s --check-prefix=DISABLED
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=24576' | FileCheck %s --check-prefix=NOFIT
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=57376' | FileCheck %s --check-prefix=EXACT-FIT
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=57375' | FileCheck %s --check-prefix=BELOW-FIT
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=61440' | FileCheck %s --check-prefix=SCRATCH-BUDGET
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=4 l1-budget-override=98304' | FileCheck %s --check-prefix=ADDRESS-BUDGET
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=OVERLAP
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' | FileCheck %s --check-prefix=PAGES
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=4},convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' | FileCheck %s --check-prefix=RESIDUAL
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' | FileCheck %s --check-prefix=GROUPED-WRITE
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s --check-prefix=PIPELINE

#layout = #ttl.layout<
    shape = [32, 384], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// Automatic selection handles a nonzero lower bound and uses two destination
// groups so the sender and receiver can overlap.
// AUTO-LABEL: func.func @point_to_point
// AUTO: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5} {dfb_id = 0 : index}
// AUTO-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 10} {dfb_id = 1 : index}
// AUTO-DAG: %[[LOWER:.*]] = arith.constant 2 : index
// AUTO-DAG: %[[PIPE:.*]] = ttl.create_pipe
// AUTO: %[[GROUPED_TRANSFER:.*]] = ttl.pipe_transfer.create %[[PIPE]] {block_span = 5 : i64, destination_group_depth = 2 : i64
// AUTO: scf.for %[[ITER:.*]] = %[[LOWER]] to %{{.*}} step %{{.*}} {
// AUTO: %[[SEND_RESERVE:.*]] = ttl.cb_reserve %[[SRC]] {num_tiles = 5 : i64}
// AUTO-SAME: -> tensor<1x5x!ttcore.tile<32x32, f32>>
// AUTO: ttl.tensor_slice %{{.*}}[%{{.*}}, %[[ITER]]]
// AUTO-SAME: -> tensor<1x5x!ttcore.tile<32x32, f32>,
// AUTO: ttl.pipe_transfer.send %[[GROUPED_TRANSFER]], %[[SRC]]
// AUTO: %[[RECV_RESERVE:.*]] = ttl.cb_reserve %[[DST]] {num_tiles = 5 : i64}
// AUTO-SAME: -> tensor<1x5x!ttcore.tile<32x32, f32>>
// AUTO: ttl.pipe_transfer.post %[[GROUPED_TRANSFER]], %[[RECV_RESERVE]]
// AUTO: ttl.cb_pop %[[DST]] {num_tiles = 5 : i64}

// Capacity synchronization and the two proven receiver groups select bounded
// overlap for the grouped stream.
// OVERLAP: PipeTransport: stream 0 transfer 0
// OVERLAP-SAME: synchronization=capacity schedule=overlapped credit_completion=iteration_domain group=5
// OVERLAP-NEXT: PipeTransport:   source blocks=5 block_span=5 stage_depth=1 ownership=transport scratch_offset=0 scratch_bytes=20480 pages=5
// OVERLAP-NEXT: PipeTransport:   endpoint 0
// OVERLAP-SAME: block_count=10 slot_span=5 group_depth=2
// OVERLAP-SAME: ownership=transport scratch_offset=0 scratch_bytes=40960
// OVERLAP-SAME: address=recurrence(initial=0, stride=5, modulus=10, executions=2)

// An overlapped payload larger than one NoC burst is decomposed into pages.
// Transport scratch replaces the grouped DFB lifecycle. Setup executes once
// under the sender predicate; page source and destination addresses advance by
// the same byte offset inside the group loop.
// PAGES: module attributes
// PAGES-SAME: ttl.pipe_sram_scratch_bytes = 40960 : i64
// PAGES-LABEL: func.func @point_to_point
// PAGES-NOT: ttl.pipe_computed_address_dfb_indices
// PAGES-DAG: %[[PAGE_BYTES:.*]] = arith.constant 4096 : i32
// PAGES-DAG: %[[SCRATCH_ARG:.*]] = arith.constant 2 : index
// PAGES: ttkernel.get_common_arg_val(%[[SCRATCH_ARG]])
// PAGES-NOT: ttkernel.get_compile_time_arg_val
// PAGES-NOT: ttkernel.cb_
// PAGES-NOT: ttkernel.get_read_ptr
// PAGES-NOT: ttkernel.get_write_ptr
// PAGES: scf.if %[[IS_SOURCE:.*]] {
// PAGES-NEXT: ttkernel.noc_async_write_one_packet_set_state(%{{.*}}, %[[PAGE_BYTES]]
// PAGES: scf.for %[[GROUP_ITER:.*]] =
// PAGES-NOT: noc_async_write_one_packet_set_state
// PAGES: scf.if %[[IS_SOURCE]] {
// PAGES: scf.for %[[PAGE_ITER:.*]] =
// PAGES: %[[PAGE_I32:.*]] = arith.index_cast %[[PAGE_ITER]] : index to i32
// PAGES-NEXT: %[[PAGE_OFFSET:.*]] = arith.muli %[[PAGE_I32]], %[[PAGE_BYTES]]
// PAGES-NEXT: %[[PAGE_SOURCE:.*]] = arith.addi %{{.*}}, %[[PAGE_OFFSET]]
// PAGES-NEXT: %[[PAGE_DEST:.*]] = arith.addi %{{.*}}, %[[PAGE_OFFSET]]
// PAGES-NEXT: ttkernel.noc_async_write_one_packet_with_state(%[[PAGE_SOURCE]], %[[PAGE_DEST]]
// PAGES: }
// PAGES-NEXT: ttkernel.noc_async_write_barrier
// PAGES: ttkernel.noc_semaphore_inc
// PAGES-NOT: ttkernel.noc_async_atomic_barrier
// PAGES: } {ttkernel.execution_core_ranges = [#ttcore.core_range<(0,1), (0,1)>]}
// PAGES-NEXT: }
// PAGES-NEXT: ttkernel.noc_async_atomic_barrier
// PAGES-NOT: ttkernel.cb_
// PAGES-NEXT: return

// Grouped storage remains independent of a scalar residual that uses the
// original DFB. The grouped loop uses a two-slot transport ring and batches its
// credit completion; only the residual loop lowers DFB lifecycle operations.
// RESIDUAL: module attributes
// RESIDUAL-SAME: ttl.pipe_sram_scratch_bytes = 32800 : i64
// RESIDUAL-LABEL: func.func @point_to_point
// RESIDUAL-DAG: %[[ADDRESS_TABLE_OFFSET:.*]] = arith.constant 32768 : i32
// RESIDUAL: scf.for %{{.*}} = %[[LOWER:.*]] to %[[GROUP_END:.*]] step %[[GROUP_STEP:.*]] {
// RESIDUAL-NOT: ttkernel.cb_
// RESIDUAL: ttkernel.noc_async_atomic_barrier
// RESIDUAL: %[[SCRATCH_BASE:.*]] = ttkernel.get_common_arg_val(%c2)
// RESIDUAL-NEXT: %[[ADDRESS_TABLE:.*]] = arith.addi %[[SCRATCH_BASE]], %[[ADDRESS_TABLE_OFFSET]]
// RESIDUAL: scf.for %{{.*}} = %[[GROUP_END]] to %{{.*}} step %{{.*}} {
// RESIDUAL: ttkernel.cb_reserve_back
// RESIDUAL: ttkernel.cb_push_back
// RESIDUAL: ttkernel.cb_wait_front
// RESIDUAL: ttkernel.cb_pop_front

// Grouped receiver-post execution retains one contiguous write because it does
// not satisfy the bounded-overlap protocol.
// GROUPED-WRITE-LABEL: func.func @point_to_point
// GROUPED-WRITE-DAG: %[[PAYLOAD_BYTES:.*]] = arith.constant 20480 : i32
// GROUPED-WRITE-NOT: noc_async_write_one_packet_set_state
// GROUPED-WRITE: ttkernel.noc_async_write
// GROUPED-WRITE-SAME: %[[PAYLOAD_BYTES]]

// The registered pipeline preserves DFB synchronization until transport
// ownership is selected, then removes the transport-owned lifecycle.
// PIPELINE: module attributes
// PIPELINE-SAME: ttl.pipe_sram_scratch_bytes = 40960 : i64
// PIPELINE-LABEL: func.func @point_to_point
// PIPELINE-NOT: ttkernel.cb_
// PIPELINE: ttkernel.noc_async_write_one_packet_set_state
// PIPELINE: scf.for
// PIPELINE: ttkernel.noc_async_write_one_packet_with_state
// PIPELINE: ttkernel.noc_async_atomic_barrier

// An explicit upper bound produces two groups of four and a two-transfer
// scalar residual. The grouped and scalar loops use distinct transfer values.
// BOUND-LABEL: func.func @point_to_point
// BOUND: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 8} {dfb_id = 0 : index}
// BOUND-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 8} {dfb_id = 1 : index}
// BOUND-DAG: %[[SCALAR_TRANSFER:.*]] = ttl.pipe_transfer.create %{{.*}} {kind = #ttl.pipe_transfer_kind<point_to_point>}
// BOUND-DAG: %[[GROUPED_TRANSFER:.*]] = ttl.pipe_transfer.create %{{.*}} {block_span = 4 : i64, destination_group_depth = 2 : i64
// BOUND: scf.for %[[GROUP_ITER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// BOUND: ttl.tensor_slice %{{.*}}[%{{.*}}, %[[GROUP_ITER]]]
// BOUND-SAME: -> tensor<1x4x!ttcore.tile<32x32, f32>,
// BOUND: ttl.pipe_transfer.send %[[GROUPED_TRANSFER]], %[[SRC]]
// BOUND: ttl.pipe_transfer.post %[[GROUPED_TRANSFER]]
// BOUND: scf.for %[[RESIDUAL_ITER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// BOUND: ttl.tensor_slice %{{.*}}[%{{.*}}, %[[RESIDUAL_ITER]]]
// BOUND-SAME: -> tensor<1x1x!ttcore.tile<32x32, f32>,
// BOUND: ttl.pipe_transfer.send %[[SCALAR_TRANSFER]], %[[SRC]]
// BOUND: ttl.pipe_transfer.post %[[SCALAR_TRANSFER]]

// A bound larger than half the trip count still selects two receiver groups.
// UPPER-LABEL: func.func @point_to_point
// UPPER: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5} {dfb_id = 0 : index}
// UPPER-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 10} {dfb_id = 1 : index}
// UPPER: ttl.pipe_transfer.create %{{.*}} {block_span = 5 : i64, destination_group_depth = 2 : i64

// Disabled grouping leaves the scalar protocol unchanged.
// DISABLED-LABEL: func.func @point_to_point
// DISABLED: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5} {dfb_id = 0 : index}
// DISABLED-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index}
// DISABLED-NOT: block_span
// DISABLED: scf.for
// DISABLED: ttl.cb_reserve %[[SRC]]
// DISABLED-NOT: num_tiles
// DISABLED: ttl.copy %[[SRC]], %{{.*}}

// A budget that fits only the original allocations retains scalar transfers.
// NOFIT-LABEL: func.func @point_to_point
// NOFIT: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5} {dfb_id = 0 : index}
// NOFIT-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index}
// NOFIT-NOT: block_span
// NOFIT: scf.for
// NOFIT: ttl.pipe_transfer.send

// A group whose final DFB and scratch allocations exactly equal the L1 budget
// remains eligible.
// EXACT-FIT-LABEL: func.func @point_to_point
// EXACT-FIT: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 6} {dfb_id = 0 : index}
// EXACT-FIT-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 4} {dfb_id = 1 : index}
// EXACT-FIT: %[[STEP:.*]] = arith.constant 2 : index
// EXACT-FIT: ttl.pipe_transfer.create %{{.*}} {block_span = 2 : i64, destination_group_depth = 2 : i64
// EXACT-FIT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]]

// Reducing the exact-fit budget by one byte rejects every grouped transport.
// BELOW-FIT-LABEL: func.func @point_to_point
// BELOW-FIT: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5} {dfb_id = 0 : index}
// BELOW-FIT-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index}
// BELOW-FIT-NOT: block_span
// BELOW-FIT: scf.for
// BELOW-FIT: ttl.pipe_transfer.send

// Group selection counts the final DFB allocations and transport scratch
// together. This budget admits R=2 but rejects every larger overlapping group.
// SCRATCH-BUDGET-LABEL: func.func @point_to_point
// SCRATCH-BUDGET: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 6} {dfb_id = 0 : index}
// SCRATCH-BUDGET-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 4} {dfb_id = 1 : index}
// SCRATCH-BUDGET: %[[STEP:.*]] = arith.constant 2 : index
// SCRATCH-BUDGET: ttl.pipe_transfer.create %{{.*}} {block_span = 2 : i64, destination_group_depth = 2 : i64
// SCRATCH-BUDGET: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]]

// Receiver-published address metadata shares the L1 budget with DFB and
// transport scratch allocations. R=4 needs 98336 bytes after final planning;
// the exact 98304-byte bound therefore selects R=3.
// ADDRESS-BUDGET-LABEL: func.func @point_to_point
// ADDRESS-BUDGET: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 6} {dfb_id = 0 : index}
// ADDRESS-BUDGET-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 6} {dfb_id = 1 : index}
// ADDRESS-BUDGET: %[[STEP:.*]] = arith.constant 3 : index
// ADDRESS-BUDGET: ttl.pipe_transfer.create %{{.*}} {block_span = 3 : i64, destination_group_depth = 2 : i64
// ADDRESS-BUDGET: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]]

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @point_to_point(
      %input: tensor<1x12x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x12x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 5} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c12_i64 = arith.constant 12 : i64
    %c12 = arith.index_cast %c12_i64 : i64 to index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c2 to %c12 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 5>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x12x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %read = ttl.copy %input_slice, %src_dfb
            : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>)
            -> !ttl.transfer_handle<read>
        ttl.wait %read : !ttl.transfer_handle<read>
        ttl.cb_push %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 5>
        %ready = ttl.cb_wait %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 5>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %send = ttl.copy %src_dfb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.cb_pop %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 5>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %recv_handle = ttl.copy %pipe, %recv
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %recv_handle : !ttl.transfer_handle
        ttl.cb_push %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        %received = ttl.cb_wait %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %output_slice = ttl.tensor_slice %output[%c0, %iter]
            : tensor<1x12x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        %write = ttl.copy %dst_dfb, %output_slice
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
            -> !ttl.transfer_handle<write>
        ttl.wait %write : !ttl.transfer_handle<write>
        ttl.cb_pop %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }
    func.return
  }
}
