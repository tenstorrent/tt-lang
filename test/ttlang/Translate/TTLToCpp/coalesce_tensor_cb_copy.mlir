// RUN: ttlang-opt --allow-unregistered-dialect --ttl-to-ttkernel-pipeline --canonicalize --split-input-file %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc --split-input-file %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp --split-input-file -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --check-prefix=CPP --input-file=%t.cpp

// Verify contiguous tensor-row transfers keep their explicit byte count in
// generated TensorAccessor NoC calls in both directions.

// CPP-LABEL: void kernel_main() {
// CPP-DAG: int32_t [[READ_SIZE:v[0-9]+]] = 14336;
// CPP: noc0.async_read({{.*}}, CoreLocalMem<uint32_t>({{.*}}), [[READ_SIZE]], {.page_id = static_cast<uint32_t>({{.*}})}, {});
// CPP: noc0.async_read_barrier();

#layout = #ttl.layout<shape = [1, 7168], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = l1, grid = [2, 4], memory = height_sharded>

// A complete compact K3 row is one 14,336-byte Blackhole transfer.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @coalesce_compact_row_read(
      %tensor: tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
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

// CPP-LABEL: void kernel_main() {
// CPP-DAG: int32_t [[WRITE_SIZE:v[0-9]+]] = 14336;
// CPP: noc0.async_write(CoreLocalMem<uint32_t>({{.*}}), {{.*}}, [[WRITE_SIZE]], {} , {.page_id = static_cast<uint32_t>({{.*}})});
// CPP: noc0.async_write_barrier();

#layout = #ttl.layout<shape = [1, 7168], element_type = !ttcore.tile<1x32, bf16>,
                      buffer = l1, grid = [2, 4], memory = height_sharded>

// The reverse direction uses the same contiguous-row proof.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @coalesce_compact_row_write(
      %tensor: tensor<1x224x!ttcore.tile<1x32, bf16>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
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
