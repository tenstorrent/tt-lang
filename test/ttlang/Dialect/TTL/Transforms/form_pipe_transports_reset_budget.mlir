// Verifies PipeTransport grouping reserves synchronized-reset scratch in L1.
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=57600' | FileCheck %s --check-prefix=EXACT
// RUN: ttlang-opt %s --ttl-form-pipe-transports='l1-budget-override=57599' | FileCheck %s --check-prefix=BELOW

#layout = #ttl.layout<
    shape = [32, 384], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

// The selected R=2 grouping uses 57,600 target-aligned DFB and runtime bytes.
// Its reset record shares the existing allocator-rounded scratch allocation.
// EXACT-LABEL: func.func @point_to_point
// EXACT: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 6}
// EXACT-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 4}
// EXACT: ttl.pipe_transfer.create {{.*}} {block_span = 2 : i64

// One byte below the combined allocation retains scalar transfers.
// BELOW-LABEL: func.func @point_to_point
// BELOW: %[[SRC:.*]] = ttl.bind_cb{cb_index = 0, block_count = 5}
// BELOW-NEXT: %[[DST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1}
// BELOW-NOT: block_span
// BELOW: scf.for
// BELOW: ttl.pipe_transfer.send

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
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
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>
    func.return
  }
}
