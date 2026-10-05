// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies graph PipeNet callbacks lower from separate graph and
// launch-node relations.

#records = #ttl.pipenet_records<net 0 name "exchange" mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func private @consume(index, index, index, index)

  // The callback iterates the current device's three outgoing edges. Selected
  // device and worker coordinates are derived without a 12-record table.
  // CHECK-LABEL: func.func @source
  // CHECK: %[[THREE:.*]] = arith.constant 3 : index
  // CHECK: %[[DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[DEVICE:.*]] = arith.index_cast %[[DEVICE_I32]] : i32 to index
  // CHECK: scf.for %[[EDGE:.*]] = %{{.*}} to %[[THREE]] step
  // CHECK-NOT: array<12xi64>
  // CHECK: func.call @consume({{.*}}, {{.*}}, {{.*}}, {{.*}})
  func.func @source() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination_device = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      %destination_x, %destination_y, %end_x, %end_y =
          ttl.selected_pipe_destination_coordinates %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination_device, %destination_x,
                         %destination_y, %end_x)
          : (index, index, index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // The destination callback uses the same relation but enumerates incoming
  // edges and preserves the declared destination worker coordinate.
  // CHECK-LABEL: func.func @destination
  // CHECK: %[[DST_THREE:.*]] = arith.constant 3 : index
  // CHECK: %[[DST_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[DST_DEVICE:.*]] = arith.index_cast %[[DST_DEVICE_I32]] : i32 to index
  // CHECK: scf.for %[[DST_EDGE:.*]] = %{{.*}} to %[[DST_THREE]] step
  // CHECK-NOT: array<12xi64>
  // CHECK: func.call @consume({{.*}}, {{.*}}, {{.*}}, {{.*}})
  func.func @destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source_device = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      %source_x, %source_y = ttl.selected_pipe_source_coordinates %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source_device, %source_x, %source_y, %source_x)
          : (index, index, index, index) -> ()
      ttl.yield
    }
    func.return
  }
}
