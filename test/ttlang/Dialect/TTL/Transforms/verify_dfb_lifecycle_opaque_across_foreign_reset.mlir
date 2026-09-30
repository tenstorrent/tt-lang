// An opaque push closes a user reserve across a reset of another DFB.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-lifecycle)' | FileCheck %s

// CHECK-LABEL: func.func @reader
// CHECK: ttl.cb_reserve
// CHECK: ttl.reset_dfbs
// CHECK: ttl.opaque_call "publish"
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "op">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %carried = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %other = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %reserved = ttl.cb_reserve %carried
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "op">,
                                    <kind = data_movement, identity = "reader", operation = "op">,
                                    <kind = data_movement, identity = "writer", operation = "op">]>
        (%other : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.opaque_call "publish"
        dfb_dependencies(%carried : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "op">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %other = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "op">,
                                    <kind = data_movement, identity = "reader", operation = "op">,
                                    <kind = data_movement, identity = "writer", operation = "op">]>
        (%other : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    func.return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "op">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %other = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "op">,
                                    <kind = data_movement, identity = "reader", operation = "op">,
                                    <kind = data_movement, identity = "writer", operation = "op">]>
        (%other : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    func.return
  }
}
