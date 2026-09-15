// Summary: Verifies exact finalized DFB reconfiguration-state accounting.
// RUN: ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=2111})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{
      dfb_index = 0 : i32,
      configurations = [{
        block_count = 1 : i32,
        element_type = !ttcore.tile<32x32, bf16>,
        entry_reconfiguration = 0 : i64,
        num_tiles = 1 : i32,
        page_size = 2048 : i32
      }]
    }]
  }
} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    // One 2048-byte DFB and one aligned 64-byte configuration allocation.
    // expected-error @below {{'ttl.bind_cb' op total DFB and fixed-state allocation (2112 bytes) exceeds L1 budget (2111 bytes)}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    func.return
  }

  func.func @reader() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    func.return
  }

  func.func @writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    func.return
  }
}
