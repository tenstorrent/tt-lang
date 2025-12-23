// RUN: ttlang-opt %s | FileCheck %s

// Test SliceAttr parsing and printing

// CHECK-LABEL: func.func @test_slice_attr
// CHECK-SAME: {s = #ttl.slice<start = 0, stop = 8, step = 1>}
func.func @test_slice_attr() attributes {s = #ttl.slice<start = 0, stop = 8, step = 1>} {
  return
}

// CHECK-LABEL: func.func @test_slice_attr_step2
// CHECK-SAME: {s = #ttl.slice<start = 0, stop = 16, step = 2>}
func.func @test_slice_attr_step2() attributes {s = #ttl.slice<start = 0, stop = 16, step = 2>} {
  return
}
