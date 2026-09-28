// A local tensor-backed DFB cannot share a physical index with a
// remote_uniform DFB on other nodes: the shared index would take the stricter
// scope, which the tensor's own shard addresses cannot provide.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-SAME: tensor_backing
// CHECK-SAME: address_scope = "remote_uniform"
// CHECK-SAME: dfb_index = 1 : i32

module attributes {ttl.launch_grid = [3, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @mixed()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %backed_local = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %remote_uniform = ttl.bind_cb {cb_index = 1, block_count = 1}
        {address_scope = "remote_uniform", dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %first_node = arith.cmpi ne, %node_x, %two : index
    %second_node = arith.cmpi eq, %node_x, %two : index
    scf.if %first_node {
      ttl.opaque_call "use_backed" dfb_dependencies(
          %backed_local : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
    scf.if %second_node {
      ttl.opaque_call "use_remote_uniform" dfb_dependencies(
          %remote_uniform : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}
