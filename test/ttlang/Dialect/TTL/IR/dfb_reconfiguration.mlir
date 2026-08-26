// Summary: Verifies DFB reconfiguration operation parsing and printing.
// RUN: ttlang-opt %s | FileCheck %s

// CHECK-LABEL: func.func @boundary
// CHECK: ttl.dfb_reconfiguration <0, participants[<kind = compute>, <kind = data_movement, identity = "reader", operation = "operation">, <kind = data_movement, identity = "writer", operation = "operation">]>
func.func @boundary() {
  ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
  return
}
