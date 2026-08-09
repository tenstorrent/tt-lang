// RUN: ttlang-opt %s | FileCheck %s

// Tests parsing and printing portable logical-kernel metadata.

// A kind-only selector records the canonical kernel without an identity.
// CHECK-LABEL: func.func @canonical
// CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
func.func @canonical() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
  return
}

// An operation-owned handle records its stable operation-local identity.
// CHECK-LABEL: func.func @named
// CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
func.func @named() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">} {
  return
}

// Compiler-owned affinities use a logical role instead of an operation id.
// CHECK-LABEL: func.func @role
// CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">
func.func @role() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">} {
  return
}
