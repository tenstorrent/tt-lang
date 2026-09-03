// Summary: Verifies diagnostics for synchronized reconfiguration restrictions.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#alternate_writer = #ttl.logical_kernel<kind = data_movement, identity = "alternate_writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<0, participants[#compute, #alternate_writer, #reader]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary_a
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    // expected-error @below {{DFB reconfiguration ordinal identifies an inconsistent participant set}}
    ttl.dfb_reconfiguration #boundary_b
    return
  }
}

// -----

// Exact repeated execution outside a sequential loop is rejected.
#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#records = #ttl.pipenet_records<net 0 name "gather" pipes [
  #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
  #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
]>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      // expected-error @below {{repeated DFB reconfiguration with exact count 2 must execute once in every iteration of nested sequential loops with compile-time-known trip counts}}
      ttl.dfb_reconfiguration #boundary
      ttl.yield
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{DFB reconfiguration participants must use the same nested loop trip-count sequence}}
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#alternate_writer = #ttl.logical_kernel<kind = data_movement, identity = "alternate_writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<1, participants[#compute, #alternate_writer, #reader]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary_a
    // expected-error @below {{all DFB reconfiguration boundaries must declare the same participant set}}
    ttl.dfb_reconfiguration #boundary_b
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{repeated DFB reconfiguration requires at least two ordered reconfiguration calls in the loop}}
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute(%upper : index) attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{DFB reconfiguration must execute at most once per dispatch and launch node or once per iteration of nested sequential loops with compile-time-known trip counts}}
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @read(%upper : index) attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write(%upper : index) attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary
    // expected-error @below {{DFB reconfiguration has multiple dynamic instance candidates for one logical-kernel participant}}
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "compute_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "reader_condition" () {
        condition_result = #ttl.dispatch_condition<1, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      // expected-error @below {{DFB reconfiguration participants execute under different structured conditions}}
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "writer_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    // expected-error @below {{DFB reconfiguration has inconsistent participant execution at one launch node}}
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %inactive = arith.constant false
    scf.if %inactive {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    // expected-error @below {{DFB reconfiguration participants have inconsistent dynamic instance counts}}
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "reader_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "writer_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary_a
    ttl.dfb_reconfiguration #boundary_b
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    // expected-error @below {{DFB reconfiguration participants execute boundaries in different orders}}
    ttl.dfb_reconfiguration #boundary_b
    ttl.dfb_reconfiguration #boundary_a
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary_a
    ttl.dfb_reconfiguration #boundary_b
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    scf.if %first_node {
      // expected-error @below {{DFB reconfiguration boundaries execute in different orders across launch nodes}}
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    } else {
      ttl.dfb_reconfiguration #boundary_b
      ttl.dfb_reconfiguration #boundary_a
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    scf.if %first_node {
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    } else {
      ttl.dfb_reconfiguration #boundary_b
      ttl.dfb_reconfiguration #boundary_a
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    scf.if %first_node {
      ttl.dfb_reconfiguration #boundary_a
      ttl.dfb_reconfiguration #boundary_b
    } else {
      ttl.dfb_reconfiguration #boundary_b
      ttl.dfb_reconfiguration #boundary_a
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary_a = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary_b = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %first_iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_a
    }
    scf.for %second_iteration = %lower to %upper step %step {
      // expected-error @below {{repeated DFB reconfiguration requires every reconfiguration call on the launch node to use the same enclosing loop nest in each participant}}
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %first_iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_a
    }
    scf.for %second_iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %first_iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_a
    }
    scf.for %second_iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary_b
    }
    return
  }
}
