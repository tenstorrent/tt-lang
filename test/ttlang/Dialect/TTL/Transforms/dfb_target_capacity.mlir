// Tests the Blackhole physical DFB-index capacity through a default device.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// CHECK: ttl.dfb_allocations = [
// CHECK-COUNT-64: dfb_index = {{[0-9]+}} : i32
// CHECK-NOT: dfb_index
// CHECK-LABEL: func.func @blackhole_accepts_64_indices
// CHECK-SAME: ttl.base_cta_index = 64 : i32
// CHECK: ttl.bind_cb{cb_index = 63, block_count = 1}

module attributes {
  ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <blackhole>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 204800, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>
} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @blackhole_accepts_64_indices()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 64 : i32, ttl.crta_indices = []} {
    %dfb0 = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb1 = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb2 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb3 = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb4 = ttl.bind_cb {cb_index = 4, block_count = 1} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb5 = ttl.bind_cb {cb_index = 5, block_count = 1} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb6 = ttl.bind_cb {cb_index = 6, block_count = 1} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb7 = ttl.bind_cb {cb_index = 7, block_count = 1} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb8 = ttl.bind_cb {cb_index = 8, block_count = 1} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb9 = ttl.bind_cb {cb_index = 9, block_count = 1} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb10 = ttl.bind_cb {cb_index = 10, block_count = 1} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb11 = ttl.bind_cb {cb_index = 11, block_count = 1} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb12 = ttl.bind_cb {cb_index = 12, block_count = 1} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb13 = ttl.bind_cb {cb_index = 13, block_count = 1} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb14 = ttl.bind_cb {cb_index = 14, block_count = 1} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb15 = ttl.bind_cb {cb_index = 15, block_count = 1} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb16 = ttl.bind_cb {cb_index = 16, block_count = 1} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb17 = ttl.bind_cb {cb_index = 17, block_count = 1} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb18 = ttl.bind_cb {cb_index = 18, block_count = 1} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb19 = ttl.bind_cb {cb_index = 19, block_count = 1} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb20 = ttl.bind_cb {cb_index = 20, block_count = 1} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb21 = ttl.bind_cb {cb_index = 21, block_count = 1} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb22 = ttl.bind_cb {cb_index = 22, block_count = 1} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb23 = ttl.bind_cb {cb_index = 23, block_count = 1} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb24 = ttl.bind_cb {cb_index = 24, block_count = 1} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb25 = ttl.bind_cb {cb_index = 25, block_count = 1} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb26 = ttl.bind_cb {cb_index = 26, block_count = 1} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb27 = ttl.bind_cb {cb_index = 27, block_count = 1} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb28 = ttl.bind_cb {cb_index = 28, block_count = 1} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb29 = ttl.bind_cb {cb_index = 29, block_count = 1} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb30 = ttl.bind_cb {cb_index = 30, block_count = 1} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb31 = ttl.bind_cb {cb_index = 31, block_count = 1} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb32 = ttl.bind_cb {cb_index = 32, block_count = 1} {dfb_id = 32 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb33 = ttl.bind_cb {cb_index = 33, block_count = 1} {dfb_id = 33 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb34 = ttl.bind_cb {cb_index = 34, block_count = 1} {dfb_id = 34 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb35 = ttl.bind_cb {cb_index = 35, block_count = 1} {dfb_id = 35 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb36 = ttl.bind_cb {cb_index = 36, block_count = 1} {dfb_id = 36 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb37 = ttl.bind_cb {cb_index = 37, block_count = 1} {dfb_id = 37 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb38 = ttl.bind_cb {cb_index = 38, block_count = 1} {dfb_id = 38 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb39 = ttl.bind_cb {cb_index = 39, block_count = 1} {dfb_id = 39 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb40 = ttl.bind_cb {cb_index = 40, block_count = 1} {dfb_id = 40 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb41 = ttl.bind_cb {cb_index = 41, block_count = 1} {dfb_id = 41 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb42 = ttl.bind_cb {cb_index = 42, block_count = 1} {dfb_id = 42 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb43 = ttl.bind_cb {cb_index = 43, block_count = 1} {dfb_id = 43 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb44 = ttl.bind_cb {cb_index = 44, block_count = 1} {dfb_id = 44 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb45 = ttl.bind_cb {cb_index = 45, block_count = 1} {dfb_id = 45 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb46 = ttl.bind_cb {cb_index = 46, block_count = 1} {dfb_id = 46 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb47 = ttl.bind_cb {cb_index = 47, block_count = 1} {dfb_id = 47 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb48 = ttl.bind_cb {cb_index = 48, block_count = 1} {dfb_id = 48 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb49 = ttl.bind_cb {cb_index = 49, block_count = 1} {dfb_id = 49 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb50 = ttl.bind_cb {cb_index = 50, block_count = 1} {dfb_id = 50 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb51 = ttl.bind_cb {cb_index = 51, block_count = 1} {dfb_id = 51 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb52 = ttl.bind_cb {cb_index = 52, block_count = 1} {dfb_id = 52 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb53 = ttl.bind_cb {cb_index = 53, block_count = 1} {dfb_id = 53 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb54 = ttl.bind_cb {cb_index = 54, block_count = 1} {dfb_id = 54 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb55 = ttl.bind_cb {cb_index = 55, block_count = 1} {dfb_id = 55 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb56 = ttl.bind_cb {cb_index = 56, block_count = 1} {dfb_id = 56 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb57 = ttl.bind_cb {cb_index = 57, block_count = 1} {dfb_id = 57 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb58 = ttl.bind_cb {cb_index = 58, block_count = 1} {dfb_id = 58 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb59 = ttl.bind_cb {cb_index = 59, block_count = 1} {dfb_id = 59 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb60 = ttl.bind_cb {cb_index = 60, block_count = 1} {dfb_id = 60 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb61 = ttl.bind_cb {cb_index = 61, block_count = 1} {dfb_id = 61 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb62 = ttl.bind_cb {cb_index = 62, block_count = 1} {dfb_id = 62 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb63 = ttl.bind_cb {cb_index = 63, block_count = 1} {dfb_id = 63 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
