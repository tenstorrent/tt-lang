// Negative tests for TTKernel operation verification.
// Verifies that operations requiring a kernel function reject module scope.

// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// -----

// Test: dataflow buffer queue ops must appear inside a kernel function.
%cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
%count = arith.constant 1 : i32
// expected-error @below {{'ttkernel.cb_push_back' op CBPushBackOp must be inside a kernel function}}
ttkernel.cb_push_back(%cb, %count) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()

// -----

// Routing-plane runtime argument indices cannot be negative.
module {
  func.func @negative_routing_plane_runtime_arg_base() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %count = arith.constant 1 : i32
    %manager = ttkernel.routing_plane.create_connection_manager
      : !ttkernel.routing_plane_connection_manager
    // expected-error @below {{attribute 'runtimeArgBase' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
    %route_id = ttkernel.routing_plane.open_connections
      %manager, %count runtime_arg_base = -1
      : (!ttkernel.routing_plane_connection_manager, i32) -> i32
    func.return
  }
}

// -----

// A routing-plane operation must use the manager that opened its route.
module {
  func.func @mismatched_routing_plane_manager() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %count = arith.constant 1 : i32
    %connection_index = arith.constant 0 : i32
    %destination_device_id = arith.constant 1 : i32
    %destination_mesh_id = arith.constant 0 : i32
    %node = arith.constant 0 : index
    %semaphore = arith.constant 4096 : i32
    %increment = arith.constant 1 : i32
    %noc = arith.constant 0 : i8
    %semaphore_address = ttkernel.get_noc_addr(
      %node, %node, %semaphore, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
    %manager = ttkernel.routing_plane.create_connection_manager
      : !ttkernel.routing_plane_connection_manager
    %other_manager = ttkernel.routing_plane.create_connection_manager
      : !ttkernel.routing_plane_connection_manager
    %route_id = ttkernel.routing_plane.open_connections
      %manager, %count runtime_arg_base = 1
      : (!ttkernel.routing_plane_connection_manager, i32) -> i32
    // expected-error @below {{manager must match the manager that produced the route id}}
    ttkernel.routing_plane.atomic_inc(
      %other_manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %semaphore_address, %increment)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32,
         !ttkernel.noc_addr, i32) -> ()
    func.return
  }
}
