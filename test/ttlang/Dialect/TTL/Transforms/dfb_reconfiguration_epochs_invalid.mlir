// Summary: Verifies diagnostics for invalid DFB reconfiguration epoch plans.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    // expected-error @below {{DFB reconfiguration is missing a declared logical-kernel participant}}
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
    return
  }
}

// -----

#compute_l1 = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "l1_operation">
#reader_l1 = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "l1_operation">
#writer_l1 = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "l1_operation">
#boundary_l1 = #ttl.dfb_reconfiguration<0, participants[#compute_l1, #reader_l1, #writer_l1]>

// expected-error @below {{selected DFB and fixed-state allocation uses 3136 L1 bytes, exceeding the 3000-byte budget (DFB=2048, reset scratch=0, reconfiguration state=1088)}}
module attributes {
  ttl.launch_grid = [1, 1],
  ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 3000, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>
} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute_l1
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "balanced" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #boundary_l1
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader_l1,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary_l1
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer_l1,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary_l1
    return
  }
}
