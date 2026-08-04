// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores,canonicalize,cse)' | FileCheck %s

// Summary: these tests characterize the address over-specialization caused by
// replacing every logical-coordinate read in a kernel that has any
// coordinate-dependent branch. The branch needs a role-specialized value, but
// unrelated address expressions should remain dynamic so cores with the same
// role can continue sharing one binary.

// -- Test 1: a y-only role branch also freezes an unrelated x shard offset. --
// The two rows require two control-flow variants. Replacing x as well creates
// four variants and bakes a different local shard address into each column.

// CHECK-NOT: func.func @unrelated_x_shard()
// CHECK-LABEL: func.func @unrelated_x_shard_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 0]]
// CHECK: %[[ROW0:.+]] = arith.addi %arg0, %{{.+}} : index
// CHECK: return %[[ROW0]] : index
// CHECK-LABEL: func.func @unrelated_x_shard_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: %[[X1_ROLE0_OFFSET:.+]] = arith.constant 4103 : index
// CHECK: %[[ROLE0_X1:.+]] = arith.addi %arg0, %[[X1_ROLE0_OFFSET]] : index
// CHECK: return %[[ROLE0_X1]] : index
// CHECK-LABEL: func.func @unrelated_x_shard_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-LABEL: func.func @unrelated_x_shard_c1_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 1]]
// CHECK: %[[X1_ROLE1_OFFSET:.+]] = arith.constant 4105 : index
// CHECK: arith.addi %arg0, %[[X1_ROLE1_OFFSET]] : index

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
// CHECK: %[[BANK0:.+]] = arith.constant 0 : index
// CHECK: return %[[BANK0]] : index
// CHECK-LABEL: func.func @router_bank_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: %[[BANK1:.+]] = arith.constant 1 : index
// CHECK: return %[[BANK1]] : index
// CHECK-LABEL: func.func @router_bank_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK: %[[BANK8:.+]] = arith.constant 8 : index
// CHECK: return %[[BANK8]] : index
// CHECK-LABEL: func.func @router_bank_c1_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 1]]
// CHECK: %[[BANK9:.+]] = arith.constant 9 : index
// CHECK: return %[[BANK9]] : index

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
// CHECK: %[[PEER1:.+]] = arith.constant 1 : index
// CHECK: return %[[PEER1]] : index
// CHECK-LABEL: func.func @ring_peer_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK: %[[PEER2:.+]] = arith.constant 2 : index
// CHECK: return %[[PEER2]] : index
// CHECK-LABEL: func.func @ring_peer_c2_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}2, 0]]
// CHECK: %[[PEER3:.+]] = arith.constant 3 : index
// CHECK: return %[[PEER3]] : index
// CHECK-LABEL: func.func @ring_peer_c3_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}3, 0]]
// CHECK: %[[PEER0:.+]] = arith.constant 0 : index
// CHECK: return %[[PEER0]] : index

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
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[NOC00:.+]] = ttkernel.get_noc_addr(%[[C0]], %[[C0]], %arg0)
// CHECK: return %[[NOC00]] : !ttkernel.noc_addr
// CHECK-LABEL: func.func @noc_address_outside_branch_c1_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}1, 0]]
// CHECK-DAG: %[[X1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[Y0:.+]] = arith.constant 0 : index
// CHECK: %[[NOC10:.+]] = ttkernel.get_noc_addr(%[[X1]], %[[Y0]], %arg0)
// CHECK: return %[[NOC10]] : !ttkernel.noc_addr
// CHECK-LABEL: func.func @noc_address_outside_branch_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-DAG: %[[X0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Y1:.+]] = arith.constant 1 : index
// CHECK: %[[NOC01:.+]] = ttkernel.get_noc_addr(%[[X0]], %[[Y1]], %arg0)
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
