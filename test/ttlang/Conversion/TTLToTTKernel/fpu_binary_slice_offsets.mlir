// The local tile coordinates match, but the rhs slice begins at DFB tile 3.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-set-compute-kernel-config,func.func(ttl-assign-dst,ttl-subblock-compute-for-dst{subblock-sync=true},ttl-lower-to-loops,ttl-schedule-operations,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse)' | FileCheck %s --check-prefix=FPU
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-set-compute-kernel-config,func.func(ttl-assign-dst,ttl-subblock-compute-for-dst{subblock-sync=true},ttl-lower-to-loops,ttl-schedule-operations,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse,lower-affine)' -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s --check-prefix=CPP

// FPU-LABEL: func.func @fpu_binary_slice_offsets
// FPU-DAG: %[[RHS_INDEX:.*]] = arith.constant 3 : index
// FPU-DAG: %[[LHS_INDEX:.*]] = arith.constant 0 : index
// FPU: ttkernel.mul_tiles(%{{.*}}, %{{.*}}, %[[LHS_INDEX]], %[[RHS_INDEX]], %[[LHS_INDEX]])

// CPP-LABEL: void kernel_main() {
// CPP-DAG: size_t [[RHS_INDEX:v[0-9]+]] = 3;
// CPP-DAG: size_t [[LHS_INDEX:v[0-9]+]] = 0;
// CPP: mul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(2), [[LHS_INDEX]], [[RHS_INDEX]], [[LHS_INDEX]]);

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fpu_binary_slice_offsets()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
  %lhs_ready = ttl.cb_wait %lhs_cb : <[4, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<4x1x!ttcore.tile<32x32, bf16>>
  %lhs_attached = ttl.attach_cb %lhs_ready, %lhs_cb : (tensor<4x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<4x1x!ttcore.tile<32x32, bf16>>
  %rhs_ready = ttl.cb_wait %rhs_cb : <[4, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<4x1x!ttcore.tile<32x32, bf16>>
  %rhs_attached = ttl.attach_cb %rhs_ready, %rhs_cb : (tensor<4x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<4x1x!ttcore.tile<32x32, bf16>>
  %lhs = tensor.extract_slice %lhs_attached[0, 0] [1, 1] [1, 1] : tensor<4x1x!ttcore.tile<32x32, bf16>> to tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = tensor.extract_slice %rhs_attached[3, 0] [1, 1] [1, 1] : tensor<4x1x!ttcore.tile<32x32, bf16>> to tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %out_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                        tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %row_index = ttl.iter_index 0 : index
    %column_index = ttl.iter_index 1 : index
    %zero = arith.constant 0 : index
    %product = ttl.tile_mul %lhs_tile, %rhs_tile into dst[%zero] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %product, %out_view[%row_index, %column_index] from dst[%zero] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}
