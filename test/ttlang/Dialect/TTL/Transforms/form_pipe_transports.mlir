// Summary: Verify repeated scalar PipeNet transfers become bounded grouped
// transfers while preserving scalar residual execution.
// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s --check-prefix=AUTO
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=4' | FileCheck %s --check-prefix=BOUND
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=8' | FileCheck %s --check-prefix=UPPER
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=1' | FileCheck %s --check-prefix=DISABLED
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=24576' | FileCheck %s --check-prefix=NOFIT
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=61440' | FileCheck %s --check-prefix=NONMONOTONIC
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=OVERLAP

#layout = #ttl.layout<
    shape = [32, 320], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// Automatic selection uses two destination groups so the sender and receiver
// can overlap without allocating more receiver storage than required.
// AUTO-LABEL: func.func @point_to_point
// AUTO: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// AUTO-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 10}
// AUTO: %[[PIPE:.*]] = ttl.create_pipe
// AUTO: %[[GROUPED_TRANSFER:.*]] = ttl.pipe_transfer.create %[[PIPE]] {block_span = 5 : i64, destination_group_depth = 2 : i64
// AUTO: scf.for %[[ITER:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
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
// OVERLAP-SAME: synchronization=capacity schedule=overlapped group=5
// OVERLAP-NEXT: PipeTransport:   source blocks=5 block_span=5 stage_depth=1 pages=5
// OVERLAP-NEXT: PipeTransport:   endpoint 0
// OVERLAP-SAME: block_count=10 slot_span=5 group_depth=2
// OVERLAP-SAME: address=recurrence(initial=0, stride=5, modulus=10, executions=2)

// An explicit upper bound produces two groups of four and a two-transfer
// scalar residual. The grouped and scalar loops use distinct transfer values.
// BOUND-LABEL: func.func @point_to_point
// BOUND: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 8}
// BOUND-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 8}
// BOUND: %[[SCALAR_TRANSFER:.*]] = ttl.pipe_transfer.create %{{.*}} {expectedReceivers
// BOUND: %[[GROUPED_TRANSFER:.*]] = ttl.pipe_transfer.create %{{.*}} {block_span = 4 : i64, destination_group_depth = 2 : i64
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
// UPPER: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// UPPER-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 10}
// UPPER: ttl.pipe_transfer.create %{{.*}} {block_span = 5 : i64, destination_group_depth = 2 : i64

// Disabled grouping leaves the scalar protocol unchanged.
// DISABLED-LABEL: func.func @point_to_point
// DISABLED: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// DISABLED-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1}
// DISABLED-NOT: block_span
// DISABLED: scf.for
// DISABLED: ttl.cb_reserve %[[SRC]]
// DISABLED-NOT: num_tiles
// DISABLED: ttl.copy %[[SRC]], %{{.*}}

// A budget that fits only the original allocations retains scalar transfers.
// NOFIT-LABEL: func.func @point_to_point
// NOFIT: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// NOFIT-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1}
// NOFIT-NOT: block_span
// NOFIT: scf.for
// NOFIT: ttl.pipe_transfer.send

// DFB block-count alignment makes allocation size non-monotonic in the group
// size. R=5 with two destination groups fits this budget even though R=4 does
// not.
// NONMONOTONIC-LABEL: func.func @point_to_point
// NONMONOTONIC: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// NONMONOTONIC-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 10}
// NONMONOTONIC: %[[STEP:.*]] = arith.constant 5 : index
// NONMONOTONIC: ttl.pipe_transfer.create %{{.*}} {block_span = 5 : i64, destination_group_depth = 2 : i64
// NONMONOTONIC: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]]

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @point_to_point(
      %input: tensor<1x10x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x10x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 5}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %iter = %c0 to %c10 step %c1 {
      %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserved = ttl.cb_reserve %src_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 5>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %input_slice = ttl.tensor_slice %input[%c0, %iter]
            : tensor<1x10x!ttcore.tile<32x32, f32>, #layout>
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
            : tensor<1x10x!ttcore.tile<32x32, f32>, #layout>
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
