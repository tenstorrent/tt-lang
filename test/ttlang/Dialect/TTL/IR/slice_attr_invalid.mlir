// RUN: not ttlang-opt %s 2>&1 | FileCheck %s

// Test invalid SliceAttr syntax

// CHECK: expected integer value
func.func @test_slice_invalid_start() attributes {s = #ttl.slice<start = "foo", stop = 8, step = 1>} {
  return
}

// CHECK: expected integer value
func.func @test_slice_invalid_stop() attributes {s = #ttl.slice<start = 0, stop = "bar", step = 1>} {
  return
}

// CHECK: expected integer value
func.func @test_slice_invalid_step() attributes {s = #ttl.slice<start = 0, stop = 8, step = "baz">} {
  return
}

// CHECK: expected '='
func.func @test_slice_missing_equals() attributes {s = #ttl.slice<start 0, stop = 8, step = 1>} {
  return
}

// CHECK: expected identifier
func.func @test_slice_missing_field() attributes {s = #ttl.slice<start = 0, stop = 8>} {
  return
}
