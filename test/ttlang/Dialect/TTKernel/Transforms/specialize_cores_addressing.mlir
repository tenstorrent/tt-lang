// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores,canonicalize,cse)' | FileCheck %s

// Summary: these tests ensure per-core specialization folds coordinate-driven
// branches while preserving dynamic coordinate reads used by shard, bank, ring,
// and NoC address expressions.

// -- Test 1: a y-only role branch also freezes an unrelated x shard offset. --
// The two rows require two control-flow variants. Replacing x as well creates
// four variants and bakes a different local shard address into each column.

// CHECK-NOT: func.func @unrelated_x_shard()
// CHECK-LABEL: func.func @unrelated_x_shard_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.muli
// CHECK-LABEL: func.func @unrelated_x_shard_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.muli
// CHECK-LABEL: func.func @unrelated_x_shard_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.muli
// CHECK-LABEL: func.func @unrelated_x_shard_c1_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 1]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.muli

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @unrelated_x_shard(%base: index) -> index {
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
    return %result : index
  }
}

// -----

// -- Test 2: role selection freezes a coordinate-derived router bank. --------
// K3 computes bank = (y - origin_y) * row_width + x. A role branch needs only
// the selected path, but replacing both coordinates makes every bank a
// different constant and prevents cores in the same role from sharing code.

// CHECK-NOT: func.func @router_bank()
// CHECK-LABEL: func.func @router_bank_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: %[[X00:.+]] = ttkernel.my_logical_x_
// CHECK: return %[[X00]] : index
// CHECK-LABEL: func.func @router_bank_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: %[[X10:.+]] = ttkernel.my_logical_x_
// CHECK: return %[[X10]] : index
// CHECK-LABEL: func.func @router_bank_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK: ttkernel.my_logical_x_
// CHECK: ttkernel.my_logical_y_
// CHECK: arith.muli
// CHECK-LABEL: func.func @router_bank_c1_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 1]]
// CHECK: ttkernel.my_logical_x_
// CHECK: ttkernel.my_logical_y_
// CHECK: arith.muli

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @router_bank() -> index {
    %c0 = arith.constant 0 : index
    %row_width = arith.constant 8 : index
    %x = "ttkernel.my_logical_x_"() : () -> index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %is_router_row = arith.cmpi eq, %y, %c0 : index
    %row = scf.if %is_router_row -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %y : index
    }
    %row_base = arith.muli %row, %row_width : index
    %bank = arith.addi %row_base, %x : index
    return %bank : index
  }
}

// -----

// -- Test 3: a two-role branch produces one binary per ring peer. ------------
// The control flow has only two roles (x < 2 and x >= 2), but the same x read
// computes next_x = (x + 1) % 4 for a NoC/ring destination. Freezing that data
// use creates four distinct peer constants instead of two shared role kernels.

// CHECK-NOT: func.func @ring_peer()
// CHECK-LABEL: func.func @ring_peer_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.remsi
// CHECK-LABEL: func.func @ring_peer_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.remsi
// CHECK-LABEL: func.func @ring_peer_c2_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}2, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.remsi
// CHECK-LABEL: func.func @ring_peer_c3_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}3, 0]]
// CHECK: ttkernel.my_logical_x_
// CHECK: arith.remsi

module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  func.func @ring_peer() -> index {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %x = "ttkernel.my_logical_x_"() : () -> index
    %is_left_role = arith.cmpi slt, %x, %c2 : index
    %role_value = scf.if %is_left_role -> (index) {
      scf.yield %c1 : index
    } else {
      scf.yield %c2 : index
    }
    %next = arith.addi %x, %c1 : index
    %peer = arith.remsi %next, %c4 : index
    %zero = arith.subi %role_value, %role_value : index
    %result = arith.addi %peer, %zero : index
    return %result : index
  }
}

// -----

// -- Test 4: a role branch freezes coordinates in an unrelated NoC address. --
// The NoC address is computed outside the branch and needs runtime x/y only for
// routing. A y-dependent role branch nevertheless replaces both operands.

// CHECK-NOT: func.func @noc_address_outside_branch()
// CHECK-LABEL: func.func @noc_address_outside_branch_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: %[[X00:.+]] = ttkernel.my_logical_x_
// CHECK: %[[Y00:.+]] = ttkernel.my_logical_y_
// CHECK: %[[NOC00:.+]] = ttkernel.get_noc_addr(%[[X00]], %[[Y00]], %arg0)
// CHECK: return %[[NOC00]] : !ttkernel.noc_addr
// CHECK-LABEL: func.func @noc_address_outside_branch_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: %[[X10:.+]] = ttkernel.my_logical_x_
// CHECK: %[[Y10:.+]] = ttkernel.my_logical_y_
// CHECK: %[[NOC10:.+]] = ttkernel.get_noc_addr(%[[X10]], %[[Y10]], %arg0)
// CHECK: return %[[NOC10]] : !ttkernel.noc_addr
// CHECK-LABEL: func.func @noc_address_outside_branch_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK: %[[X01:.+]] = ttkernel.my_logical_x_
// CHECK: %[[Y01:.+]] = ttkernel.my_logical_y_
// CHECK: %[[NOC01:.+]] = ttkernel.get_noc_addr(%[[X01]], %[[Y01]], %arg0)
// CHECK: return %[[NOC01]] : !ttkernel.noc_addr

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @noc_address_outside_branch(
      %l1_address: i32) -> !ttkernel.noc_addr {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %x = "ttkernel.my_logical_x_"() : () -> index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %noc_address = ttkernel.get_noc_addr(%x, %y, %l1_address)
        : (index, index, i32) -> !ttkernel.noc_addr
    %is_role_zero = arith.cmpi eq, %y, %c0 : index
    %unused_role = scf.if %is_role_zero -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    return %noc_address : !ttkernel.noc_addr
  }
}
