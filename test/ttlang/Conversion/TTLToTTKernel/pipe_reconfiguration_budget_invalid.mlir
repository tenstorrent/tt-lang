// Verifies exact combined-resource validation accepts the packed allocation
// and rejects a one-byte-short budget.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=1 l1-budget-override=21760},ttl-finalize-dfb-indices{reuse-user-dfbs=true l1-budget-override=21760},convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=true pipe-global-semaphores-only=true l1-budget-override=21760})' -o /dev/null
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-form-pipe-transports{group-size=1 l1-budget-override=21759},ttl-finalize-dfb-indices{reuse-user-dfbs=true l1-budget-override=21759},convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=true pipe-global-semaphores-only=true l1-budget-override=21759})'

#layout = #ttl.layout<
    shape = [32, 384], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>
#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Physical DFB storage on the busiest launch node (20480), PipeNet scratch
// (64), two global semaphores (128), and boundary state (1088) fit separately,
// but their 21760-byte total exceeds the budget by one byte.
// expected-error @below {{combined DFB and runtime resources require 21760 L1 bytes but the budget is 21759 (DFB=20480, scratch=64, global semaphores=128, reconfiguration state=1088)}}
module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @writer(
      %input: tensor<1x12x!ttcore.tile<32x32, f32>, #layout>,
      %output: tensor<1x12x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1],
                  ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 5}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
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
            -> !ttl.receive_request
        ttl.wait %recv_handle : !ttl.receive_request
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
    ttl.dfb_reconfiguration #boundary
    func.return
  }

  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary
    func.return
  }

  func.func @reader() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    func.return
  }
}
