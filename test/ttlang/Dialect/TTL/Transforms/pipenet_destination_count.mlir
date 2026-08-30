// Verifies local and graph PipeNet destination-count lowering.
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

#records = #ttl.pipenet_records<net 0 name "gather" pipes [
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 1, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0,
   dstEndX = 2, dstEndY = 0>
]>

// Local PipeNet records have no logical-device identity. Their destination
// count depends only on the launch-node coordinate.
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
  func.func private @consume(index)

  // CHECK-LABEL: func.func @local_destination_count
  // CHECK: ttkernel.my_logical_x_
  // CHECK: ttkernel.my_logical_y_
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 0, 1, 4]
  func.func @local_destination_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %count = ttl.pipenet_destination_count {
        pipe_net_id = 0 : i64, records = #records} : index
    func.call @consume(%count) : (index) -> ()
    func.return
  }

}

// -----

#device_domain = #ttl.device_domain<
    components = <name = "device", extent = [3]>>
#device_records = #ttl.pipenet_records<net 1 name "device_gather" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #device_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [2]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #device_domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [2]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #device_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func private @consume(index)

  // CHECK-LABEL: func.func @device_destination_count
  // CHECK-NOT: scf.for
  // CHECK: ttkernel.get_common_arg_val
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 1, 2]
  func.func @device_destination_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %count = ttl.pipenet_destination_count {
        pipe_net_id = 1 : i64, records = #device_records} : index
    func.call @consume(%count) : (index) -> ()
    func.return
  }
}
