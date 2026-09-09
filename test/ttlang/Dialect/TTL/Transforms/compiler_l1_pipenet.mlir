// Summary: Verifies that PipeNet transfers coexist with compiler-managed DFB allocation.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing})' | FileCheck %s --check-prefix=SRAM
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing})' | FileCheck %s --check-prefix=SRAM
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact})' | FileCheck %s --check-prefix=SRAM
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=multi-order-decreasing})' | FileCheck %s --check-prefix=SRAM
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=metal-cb})' | FileCheck %s --check-prefix=METAL
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing' --convert-ttkernel-to-emitc -o /dev/null

// The storage model changes DFB allocation metadata without changing the
// transfer contract consumed by later PipeNet planning and lowering.
// SRAM: module attributes {ttl.dfb_allocations = [
// SRAM-SAME: dfb_index = 0 : i32
// SRAM-SAME: dfb_index = 1 : i32
// SRAM-SAME: ttl.l1_arena_bytes = 8224 : i64
// SRAM-SAME: ttl.memory_model = "compiler-l1"
// SRAM-LABEL: func.func @compiler_l1_pipenet
// SRAM: ttl.pipe_transfer.create
// SRAM: ttl.pipe_transfer.post
// SRAM: ttl.pipe_transfer.send
// METAL-NOT: ttl.memory_model
// METAL-LABEL: func.func @compiler_l1_pipenet
// METAL: ttl.pipe_transfer.create
// METAL: ttl.pipe_transfer.post
// METAL: ttl.pipe_transfer.send
module attributes {ttl.launch_grid = array<i64: 2, 1>, ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @compiler_l1_pipenet()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %destination = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
          -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %receive_block = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive_token = ttl.pipe_transfer.post %transfer, %receive_block
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %receive_token : !ttl.pipe_token<net 0>
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %consumed = ttl.cb_wait %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %produced = ttl.cb_reserve %source
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %source
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %send = ttl.pipe_transfer.send %transfer, %source
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    return
  }
}
