// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: a stable ttl.topk on a 2x2-tile block lowers through the whole
// pipeline to the metal TopK API calls the single-core stage expects, in the
// order the fused-key network needs them: transpose both inputs, fuse, local
// sort, merge, rebuild, defuse, and transpose back with the u16 index move.
// The module is the compiler input for the hardware test kernel of the same
// shape, so the data-movement threads are present for the lifecycle verifier.

// The fused keys require 32-bit DST; no TTL operation survives conversion.
// TTKERNEL-LABEL: func.func @select
// TTKERNEL-SAME: fp32_dest_acc_en = true
// TTKERNEL-NOT: ttl.topk
// TTKERNEL-NOT: ttl.tile_
// TTKERNEL-NOT: ttl.dst_section
// TTKERNEL-NOT: unrealized_conversion_cast

// Compile-time CB arguments: 0 values, 1 indices, 2 result values, 3 result
// indices, 4 and 5 the u32 fused-key banks, 6 and 7 the bf16/u16 staging
// buffers of the final extraction.
// CPP-LABEL: void kernel_main()
// CPP: for (size_t [[ROW:i[0-9]+]] = {{.*}}; [[ROW]] < {{.*}}; [[ROW]] += {{.*}}) {
// Initial sort: values and indices transposed into DST 0..3, fused, sorted.
// CPP: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(4));
// CPP-NEXT: tile_regs_acquire();
// CPP: transpose_wh_init(get_compile_time_arg_val(0), get_compile_time_arg_val(4));
// CPP-NEXT: transpose_wh_tile(get_compile_time_arg_val(0), {{.*}}, [[DST0:v[0-9]+]]);
// CPP: transpose_wh_tile(get_compile_time_arg_val(0), {{.*}}, [[DST1:v[0-9]+]]);
// CPP-NEXT: transpose_wh_init(get_compile_time_arg_val(1), get_compile_time_arg_val(1));
// CPP-NEXT: transpose_wh_tile(get_compile_time_arg_val(1), {{.*}}, [[DST2:v[0-9]+]]);
// CPP-NEXT: transpose_wh_tile(get_compile_time_arg_val(1), {{.*}}, [[DST3:v[0-9]+]]);
// CPP-NEXT: topk_tile_init<true>();
// CPP-NEXT: topk_fuse_tile<true>([[DST0]]);
// CPP-NEXT: topk_local_sort<false, DST_ACCUM_MODE, true>([[DST0]], {{.*}});
// CPP-NEXT: tile_regs_commit();
// CPP-NEXT: tile_regs_wait();
// CPP-NEXT: pack_reconfig_data_format(get_compile_time_arg_val(4));
// CPP-NEXT: pack_tile_block([[DST0]], get_compile_time_arg_val(4), [[DST2]]);
// CPP-NEXT: tile_regs_release();
// Merge: both key tiles copied from the first bank, result into the second.
// CPP: copy_tile_init(get_compile_time_arg_val(4));
// CPP-NEXT: copy_tile(get_compile_time_arg_val(4), [[DST0]], [[DST0]]);
// CPP-NEXT: copy_tile(get_compile_time_arg_val(4), [[DST1]], [[DST1]]);
// CPP-NEXT: topk_tile_init<true>();
// CPP-NEXT: topk_merge<false, false, DST_ACCUM_MODE, true>([[DST0]], {{.*}});
// CPP: pack_tile_block([[DST0]], get_compile_time_arg_val(5), [[DST2]]);
// Rebuild: the selected tile alone, skip_second set.
// CPP: copy_tile_init(get_compile_time_arg_val(5));
// CPP-NEXT: copy_tile(get_compile_time_arg_val(5), [[DST0]], [[DST0]]);
// CPP-NEXT: topk_tile_init<true>();
// CPP-NEXT: topk_rebuild<false, DST_ACCUM_MODE, true>([[DST0]], {{.*}});
// CPP: pack_tile<true>([[DST0]], get_compile_time_arg_val(4), [[DST0]]);
// The unselected column is copied across without a network call.
// CPP: copy_tile(get_compile_time_arg_val(5), [[DST1]], [[DST0]]);
// CPP-NOT: topk_
// CPP: pack_tile<true>([[DST0]], get_compile_time_arg_val(4), [[DST1]]);
// Defuse into the plain staging banks: values from DST 0, indices from DST 2.
// CPP: copy_tile(get_compile_time_arg_val(4), [[DST0]], [[DST0]]);
// CPP-NEXT: topk_tile_init<true>();
// CPP-NEXT: topk_defuse_tile<true>([[DST0]], {{.*}});
// CPP-NEXT: tile_regs_commit();
// CPP-NEXT: tile_regs_wait();
// CPP-NEXT: pack_tile<true>([[DST0]], get_compile_time_arg_val(6), [[DST0]]);
// CPP-NEXT: pack_reconfig_data_format(get_compile_time_arg_val(7));
// CPP-NEXT: pack_tile<true>([[DST2]], get_compile_time_arg_val(7), [[DST0]]);
// Final transposes into the result buffers; only the u16 index tile is moved
// into the packer half.
// CPP: transpose_wh_init(get_compile_time_arg_val(6), get_compile_time_arg_val(2));
// CPP-NEXT: transpose_wh_tile(get_compile_time_arg_val(6), [[DST0]], [[DST0]]);
// CPP-NEXT: tile_regs_commit();
// CPP: pack_tile<true>([[DST0]], get_compile_time_arg_val(2), [[ROW]]);
// CPP: transpose_wh_init(get_compile_time_arg_val(7), get_compile_time_arg_val(3));
// CPP-NEXT: transpose_wh_tile(get_compile_time_arg_val(7), [[DST0]], [[DST0]]);
// CPP-NEXT: topk_uint16_move_dest_tile_to_pack_half([[DST0]]);
// CPP-NEXT: tile_regs_commit();
// CPP: pack_tile<true>([[DST0]], get_compile_time_arg_val(3), [[ROW]]);
// CPP: }
// CPP-NOT: topk_
// CPP-NOT: tile_regs_acquire

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @select() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
    %values_cb = ttl.bind_cb{cb_index = 0, block_count = 1} {dfb_id = 0 : index} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
    %indices_cb = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index} : <[2, 2], !ttcore.tile<32x32, u16>, 1>
    %out_values_cb = ttl.bind_cb{cb_index = 2, block_count = 1} {dfb_id = 2 : index} : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
    %out_indices_cb = ttl.bind_cb{cb_index = 3, block_count = 1} {dfb_id = 3 : index} : <[2, 1], !ttcore.tile<32x32, u16>, 1>
    %values = ttl.cb_wait %values_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %values_attached = ttl.attach_cb %values, %values_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %indices = ttl.cb_wait %indices_cb : <[2, 2], !ttcore.tile<32x32, u16>, 1> -> tensor<2x2x!ttcore.tile<32x32, u16>>
    %indices_attached = ttl.attach_cb %indices, %indices_cb : (tensor<2x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, u16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, u16>>
    %values_view = ttl.cb_reserve %out_values_cb : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %values_view_attached = ttl.attach_cb %values_view, %out_values_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %indices_view = ttl.cb_reserve %out_indices_cb : <[2, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<2x1x!ttcore.tile<32x32, u16>>
    %indices_view_attached = ttl.attach_cb %indices_view, %out_indices_cb : (tensor<2x1x!ttcore.tile<32x32, u16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, u16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, u16>>
    %result_values, %result_indices = ttl.topk %values_attached, %indices_attached k = 32 {stable = true} : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, u16>>) -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, u16>>)
    ttl.store %result_values, %values_view : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result_indices, %indices_view : tensor<2x1x!ttcore.tile<32x32, u16>>, tensor<2x1x!ttcore.tile<32x32, u16>>
    ttl.cb_pop %values_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_pop %indices_cb : <[2, 2], !ttcore.tile<32x32, u16>, 1>
    ttl.cb_push %out_values_cb : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_push %out_indices_cb : <[2, 1], !ttcore.tile<32x32, u16>, 1>
    return
  }

  func.func @read(%indices: tensor<2x2x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>,
                  %values: tensor<2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
      attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [1 : i32, 0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %c0 = arith.constant 0 : index
    %indices_cb = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index} : <[2, 2], !ttcore.tile<32x32, u16>, 1>
    %values_cb = ttl.bind_cb{cb_index = 0, block_count = 1} {dfb_id = 0 : index} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
    %values_view = ttl.cb_reserve %values_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %values_slice = ttl.tensor_slice %values[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %values_xfer = ttl.copy %values_slice, %values_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> !ttl.transfer_handle<read>
    ttl.wait %values_xfer : !ttl.transfer_handle<read>
    ttl.cb_push %values_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
    %indices_view = ttl.cb_reserve %indices_cb : <[2, 2], !ttcore.tile<32x32, u16>, 1> -> tensor<2x2x!ttcore.tile<32x32, u16>>
    %indices_slice = ttl.tensor_slice %indices[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<2x2x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %indices_xfer = ttl.copy %indices_slice, %indices_cb : (tensor<2x2x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 64], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, u16>, 1>) -> !ttl.transfer_handle<read>
    ttl.wait %indices_xfer : !ttl.transfer_handle<read>
    ttl.cb_push %indices_cb : <[2, 2], !ttcore.tile<32x32, u16>, 1>
    return
  }

  func.func @write(%out_indices: tensor<2x1x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>,
                   %out_values: tensor<2x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
      attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [3 : i32, 2 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 1 : i32} {
    %c0 = arith.constant 0 : index
    %out_indices_cb = ttl.bind_cb{cb_index = 3, block_count = 1} {dfb_id = 3 : index} : <[2, 1], !ttcore.tile<32x32, u16>, 1>
    %out_values_cb = ttl.bind_cb{cb_index = 2, block_count = 1} {dfb_id = 2 : index} : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
    %values_view = ttl.cb_wait %out_values_cb : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %values_slice = ttl.tensor_slice %out_values[%c0, %c0] : tensor<2x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<2x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %values_xfer = ttl.copy %out_values_cb, %values_slice : (!ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>, tensor<2x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>) -> !ttl.transfer_handle<write>
    ttl.wait %values_xfer : !ttl.transfer_handle<write>
    ttl.cb_pop %out_values_cb : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
    %indices_view = ttl.cb_wait %out_indices_cb : <[2, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<2x1x!ttcore.tile<32x32, u16>>
    %indices_slice = ttl.tensor_slice %out_indices[%c0, %c0] : tensor<2x1x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<2x1x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %indices_xfer = ttl.copy %out_indices_cb, %indices_slice : (!ttl.cb<[2, 1], !ttcore.tile<32x32, u16>, 1>, tensor<2x1x!ttcore.tile<32x32, u16>, #ttl.layout<shape = [64, 32], element_type = !ttcore.tile<32x32, u16>, buffer = l1, grid = [1, 1], memory = interleaved>>) -> !ttl.transfer_handle<write>
    ttl.wait %indices_xfer : !ttl.transfer_handle<write>
    ttl.cb_pop %out_indices_cb : <[2, 1], !ttcore.tile<32x32, u16>, 1>
    return
  }
}
