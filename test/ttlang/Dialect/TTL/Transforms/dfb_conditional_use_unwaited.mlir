// A final-only transfer without completion before release cannot prove the
// dataflow buffer lifetime.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// CHECK: DFB logical_id=0 bounded=0
// CHECK: node (0,0) lifecycle_completion=incomplete-use-order

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @unwaited_final_transfer(%input: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        %copy = ttl.copy %input, %dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      }
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}
