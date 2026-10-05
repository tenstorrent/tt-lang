// RUN: ttlang-opt %s -convert-ttl-to-ttkernel --canonicalize --cse | FileCheck %s --implicit-check-not='arith.select' --implicit-check-not='arith.addi'

// Summary: Verifies that an explicit graph emits one edge-table scan whose
// instruction count does not grow with the declared edge count. The guards
// reject the per-edge comparison chain an incident-ordinal mapping requires.

#records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [16]>>,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [0]>,
                                  destination = <coordinates = [1]>>,
               #ttl.transfer_edge<source = <coordinates = [2]>,
                                  destination = <coordinates = [3]>>,
               #ttl.transfer_edge<source = <coordinates = [4]>,
                                  destination = <coordinates = [5]>>,
               #ttl.transfer_edge<source = <coordinates = [6]>,
                                  destination = <coordinates = [7]>>,
               #ttl.transfer_edge<source = <coordinates = [8]>,
                                  destination = <coordinates = [9]>>,
               #ttl.transfer_edge<source = <coordinates = [10]>,
                                  destination = <coordinates = [11]>>,
               #ttl.transfer_edge<source = <coordinates = [12]>,
                                  destination = <coordinates = [13]>>,
               #ttl.transfer_edge<source = <coordinates = [14]>,
                                  destination = <coordinates = [15]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func private @consume(index, index)

  // One loop over the eight declared edges, one endpoint lookup, and one
  // comparison select the records this device participates in.
  // CHECK-LABEL: func.func @explicit_scaling_source
  // CHECK: %[[DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[DEVICE:.*]] = arith.index_cast %[[DEVICE_I32]]
  // CHECK: scf.for %[[ORDINAL:.*]] = %{{.*}} to %{{.*}} step
  // CHECK: ttkernel.experimental.constant_table_lookup %[[ORDINAL]], [1, 3, 5, 7, 9, 11, 13, 15]
  // CHECK: %[[SOURCE:.*]] = ttkernel.experimental.constant_table_lookup %[[ORDINAL]], [0, 2, 4, 6, 8, 10, 12, 14]
  // CHECK: %[[INCIDENT:.*]] = arith.cmpi eq, %[[DEVICE]], %[[SOURCE]]
  // CHECK: scf.if %[[INCIDENT]]
  // CHECK: func.call @consume(%[[DESTINATION:.*]], %[[DESTINATION]])
  // CHECK-NOT: arith.cmpi
  func.func @explicit_scaling_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }
}
