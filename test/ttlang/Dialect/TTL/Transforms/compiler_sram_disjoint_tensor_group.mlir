// Verifies that disjoint worker nodes may share a control-record offset while
// using different tensor backing.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='memory-model=compiler-sram reuse-user-dfbs=true' --convert-ttkernel-to-emitc -o /dev/null
// RUN: sed 's/#ttcore.arch<blackhole>/#ttcore.arch<wormhole_b0>/' %s | ttlang-opt - --ttl-to-ttkernel-pipeline='memory-model=compiler-sram reuse-user-dfbs=true' --convert-ttkernel-to-emitc -o /dev/null

// CHECK: ttl.dfb_allocations = [{
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_index = 0 : i32
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0
// CHECK-SAME: l1_offset = 0 : i64
// CHECK-SAME: storage_index = 0 : i32
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 1
// CHECK-LABEL: func.func @disjoint_node_tensor_group

module attributes {ttl.launch_grid = array<i64: 2, 1>,
                   ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @disjoint_node_tensor_group()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1]} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 1 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_node_zero = arith.cmpi eq, %core_x, %zero : index
    %is_node_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_node_zero {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %is_node_one {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}
