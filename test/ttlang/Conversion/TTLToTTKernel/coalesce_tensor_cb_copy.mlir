// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s
// Summary: Physically contiguous L1 tensor rows use multi-tile NoC transfers,
// split at the target burst limit. Unknown layouts retain one-tile transfers.

#layout = #ttl.layout<shape = [1, 7168], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = l1, grid = [2, 4], memory = height_sharded>

// A complete compact K3 row is one 14,336-byte Blackhole transfer.
// CHECK-LABEL: func.func @coalesce_compact_row_read
// CHECK: ttkernel.noc_async_read_tile({{.*}}) {num_tiles = 224 : i32}
// CHECK: ttkernel.noc_async_read_barrier
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @coalesce_compact_row_read(
      %tensor: tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
    %slice = ttl.tensor_slice %tensor[%c0, %c0]
        : tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>
          -> tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>
    %copy = ttl.copy %slice, %cb
        : (tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>,
           !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
    ttl.wait %copy : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#layout = #ttl.layout<shape = [1, 7168], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = l1, grid = [2, 4], memory = height_sharded>

// The reverse direction uses the same contiguous-row proof.
// CHECK-LABEL: func.func @coalesce_compact_row_write
// CHECK: ttkernel.noc_async_write_tile({{.*}}) {num_tiles = 224 : i32}
// CHECK: ttkernel.noc_async_write_barrier
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @coalesce_compact_row_write(
      %tensor: tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
    %slice = ttl.tensor_slice %tensor[%c0, %c0]
        : tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>
          -> tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>
    %copy = ttl.copy %cb, %slice
        : (!ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>,
           tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>)
          -> !ttl.transfer_handle<write>
    ttl.wait %copy : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#layout = #ttl.layout<shape = [1, 9600], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = l1, grid = [2, 4], memory = height_sharded>

// A 19,200-byte row is split into legal 16,384- and 2,816-byte bursts.
// CHECK-LABEL: func.func @split_at_blackhole_burst_limit
// CHECK: ttkernel.noc_async_read_tile({{.*}}) {num_tiles = 256 : i32}
// CHECK: ttkernel.noc_async_read_tile({{.*}}) {num_tiles = 44 : i32}
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @split_at_blackhole_burst_limit(
      %tensor: tensor<1x300x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 300], !ttcore.tile<1x32, bf16>, 2>
    %slice = ttl.tensor_slice %tensor[%c0, %c0]
        : tensor<1x300x!ttcore.tile<1x32, bf16>, #layout>
          -> tensor<1x300x!ttcore.tile<1x32, bf16>, #layout>
    %copy = ttl.copy %slice, %cb
        : (tensor<1x300x!ttcore.tile<1x32, bf16>, #layout>,
           !ttl.cb<[1, 300], !ttcore.tile<1x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
    ttl.wait %copy : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#layout = #ttl.layout<shape = [1, 128], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Interleaved placement does not prove physical adjacency.
// CHECK-LABEL: func.func @interleaved_falls_back_to_tiles
// CHECK-NOT: num_tiles
// CHECK: scf.for
// CHECK: ttkernel.noc_async_read_tile
module {
  func.func @interleaved_falls_back_to_tiles(
      %tensor: tensor<1x4x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<1x32, bf16>, 2>
    %slice = ttl.tensor_slice %tensor[%c0, %c0]
        : tensor<1x4x!ttcore.tile<1x32, bf16>, #layout>
          -> tensor<1x4x!ttcore.tile<1x32, bf16>, #layout>
    %copy = ttl.copy %slice, %cb
        : (tensor<1x4x!ttcore.tile<1x32, bf16>, #layout>,
           !ttl.cb<[1, 4], !ttcore.tile<1x32, bf16>, 2>)
          -> !ttl.transfer_handle<read>
    ttl.wait %copy : !ttl.transfer_handle<read>
    func.return
  }
}
