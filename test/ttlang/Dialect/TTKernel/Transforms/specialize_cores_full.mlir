// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores{full-specialization=true},canonicalize,cse)' | FileCheck %s

// Full specialization deliberately freezes both role predicates and addressing
// coordinates so every clone can be optimized for one exact core.

// CHECK-NOT: func.func @full_per_core()
// CHECK-NOT: ttkernel.my_logical_x_
// CHECK-NOT: ttkernel.my_logical_y_
// CHECK-LABEL: func.func @full_per_core_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: arith.constant 7 : index
// CHECK: ttkernel.get_noc_addr
// CHECK-LABEL: func.func @full_per_core_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: arith.constant 4103 : index
// CHECK: ttkernel.get_noc_addr
// CHECK-LABEL: func.func @full_per_core_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK: arith.constant 9 : index
// CHECK: ttkernel.get_noc_addr
// CHECK-LABEL: func.func @full_per_core_c1_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 1]]
// CHECK: arith.constant 4105 : index
// CHECK: ttkernel.get_noc_addr

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @full_per_core(
      %base: index, %l1_address: i32) -> (index, !ttkernel.noc_addr) {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %stride = arith.constant 4096 : index
    %x = "ttkernel.my_logical_x_"() : () -> index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %is_role_zero = arith.cmpi eq, %y, %c0 : index
    %role_bias = scf.if %is_role_zero -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    %shard_offset = arith.muli %x, %stride : index
    %shard_address = arith.addi %base, %shard_offset : index
    %result = arith.addi %shard_address, %role_bias : index
    %noc_address = ttkernel.get_noc_addr(%x, %y, %l1_address)
        : (index, index, i32) -> !ttkernel.noc_addr
    return %result, %noc_address : index, !ttkernel.noc_addr
  }
}

// -----

// Full mode also clones coordinate-addressed functions that have no
// coordinate-dependent control flow.

// CHECK-NOT: func.func @address_only()
// CHECK-LABEL: func.func @address_only_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-NOT: ttkernel.my_logical_x_
// CHECK-LABEL: func.func @address_only_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK-NOT: ttkernel.my_logical_x_

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @address_only(%l1_address: i32) -> !ttkernel.noc_addr {
    %x = "ttkernel.my_logical_x_"() : () -> index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %noc_address = ttkernel.get_noc_addr(%x, %y, %l1_address)
        : (index, index, i32) -> !ttkernel.noc_addr
    return %noc_address : !ttkernel.noc_addr
  }
}
