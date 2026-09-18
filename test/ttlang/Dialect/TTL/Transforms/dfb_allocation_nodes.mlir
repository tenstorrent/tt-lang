// Verifies finalized DFB residency metadata independently from kernel specialization.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s

// The first DFB is accessed only on core (0, 0), the second is unreachable on
// the launch grid, and the third has a runtime-dependent domain. Exact domains
// are serialized, including the empty domain; the unresolved domain omits
// allocation_nodes so the runtime remains conservative.

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32}, {allocation_nodes = [], block_count = 2 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32}, {block_count = 2 : i32, dfb_index = 2 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 2 : i32}], ttl.launch_grid = array<i64: 2, 1>}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @allocation_nodes(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %first_node = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %unreachable = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %runtime_dependent = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>

    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %on_first_node = arith.cmpi eq, %core_x, %zero : index
    %outside_grid = arith.cmpi eq, %core_x, %two : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index

    scf.if %on_first_node {
      ttl.opaque_call "first_node_access" (%first_node)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %outside_grid {
      ttl.opaque_call "unreachable_access" (%unreachable)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    scf.if %runtime_condition {
      ttl.opaque_call "runtime_access" (%runtime_dependent)
          {header = "effects.hpp"}
          : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Without a launch grid, the liveness proof uses one representative node but
// runtime residency remains unknown and therefore omits allocation_nodes.

// CHECK: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32}]}

module {
  func.func @unknown_runtime_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "access" (%dfb)
        {header = "effects.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    return
  }
}

// -----

// Dense data-flow analysis omits operation lattices in unreachable CFG
// blocks. Dead-code state proves the first DFB's domain empty; the reachable
// access retains the full launch domain.

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = [], block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32}, {allocation_nodes = {{\[\[0, 0\], \[1, 0\]\]}}, block_count = 2 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32}], ttl.launch_grid = array<i64: 2, 1>}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @unreachable_cfg_block()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %dead_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %live_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    cf.br ^live
  ^dead:
    ttl.opaque_call "dead_access" (%dead_dfb)
        {header = "effects.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    cf.br ^exit
  ^live:
    ttl.opaque_call "live_access" (%live_dfb)
        {header = "effects.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    cf.br ^exit
  ^exit:
    return
  }
}
