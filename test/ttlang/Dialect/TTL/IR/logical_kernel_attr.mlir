// RUN: ttlang-opt %s | FileCheck %s

module {
  func.func @canonical_compute() attributes {
    ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
  } {
    return
  }

  func.func @operation_sender() attributes {
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "sender", operation = "models.router:42">
  } {
    return
  }

  func.func @compiler_pipe_source() attributes {
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">
  } {
    return
  }
}

// CHECK: #ttl.logical_kernel<kind = compute>
// CHECK: #ttl.logical_kernel<kind = data_movement, identity = "sender", operation = "models.router:42">
// CHECK: #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">
