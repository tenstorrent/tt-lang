// FPU path (default): add uses add_tiles (reads from CB), no copy_tile for add.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse,lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=FPU

// SFPU path: all binary ops use copy_tile + SFPU binary ops.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst{enable-fpu-binary-ops=0},ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse,lower-affine)' \
// RUN:   -o %t.sfpu.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.sfpu.ttkernel.mlir -o %t.sfpu.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.sfpu.cpp %t.sfpu.emitc.mlir
// RUN: FileCheck %s --input-file=%t.sfpu.cpp --check-prefix=SFPU

// Purpose: Complete example with reader, compute, and writer threads.
// Pattern: reader (NOC) -> CBs -> compute (MATH) -> CB -> writer (NOC)
// Operation: f(A + B) where f is exp, matching the C++ example pattern.

#layout = #ttl.layout<shape = [2, 2], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// FPU path: reader kernel (same for both paths)
// =============================================================================
// FPU-LABEL: // reader_binary
// FPU: void kernel_main() {
// FPU-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// FPU-DAG:   size_t [[ONE:v[0-9]+]] = 1
// FPU-DAG:   size_t [[PAGE_SIZE:v[0-9]+]] = 4096
// FPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// Read tensor A into CB0
// FPU:   int32_t [[RT_ARG_A:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// FPU-NEXT:   auto [[ARGS_A:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>();
// FPU-NEXT:   TensorAccessor [[ACC_A:.*]] = TensorAccessor([[ARGS_A]], [[RT_ARG_A]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// FPU:   int32_t [[CB0_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(0));
// FPU-NEXT:   ptrdiff_t [[CB0_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB0_PTR]];
// FPU-NEXT:   size_t [[CB0_PTR_IDX:v[0-9]+]] = (size_t) [[CB0_PTR_PTRDIFF]];
// FPU:   for (size_t [[I_A:.*]] = [[ZERO]]; [[I_A]] < [[BOUND]]; [[I_A]] += [[ONE]]) {
// FPU-NEXT:     for (size_t [[J_A:.*]] = [[ZERO]]; [[J_A]] < [[BOUND]]; [[J_A]] += [[ONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// FPU:       size_t [[TILE_OFF_A_Y:v[0-9]+]] = [[I_A]] * [[BOUND]];
// FPU-NEXT:  size_t [[TILE_OFF_A:v[0-9]+]] = [[TILE_OFF_A_Y]] + [[J_A]];
// Byte offset: tile_offset * page_size + cb_base
// FPU-NEXT:  size_t [[BYTE_OFF_A:v[0-9]+]] = [[TILE_OFF_A]] * [[PAGE_SIZE]];
// FPU-NEXT:  size_t [[CB_ADDR_A_IDX:v[0-9]+]] = [[CB0_PTR_IDX]] + [[BYTE_OFF_A]];
// Cast tile offset and CB address to int32_t for noc_async_read_tile
// FPU-NEXT:  ptrdiff_t [[TILE_OFF_A_PD:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_A]];
// FPU-NEXT:  int32_t [[TILE_OFF_A_I32:v[0-9]+]] = (int32_t) [[TILE_OFF_A_PD]];
// FPU-NEXT:  ptrdiff_t [[CB_ADDR_A_PD:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_A_IDX]];
// FPU-NEXT:  int32_t [[CB_ADDR_A:v[0-9]+]] = (int32_t) [[CB_ADDR_A_PD]];
// FPU-NEXT:  noc_async_read_tile([[TILE_OFF_A_I32]], [[ACC_A]], [[CB_ADDR_A]]);
// FPU:     }
// FPU-NEXT:   }
// FPU-NEXT:   noc_async_read_barrier();

// Read tensor B into CB1
// FPU:   int32_t [[RT_ARG_B:.*]] = get_common_arg_val<uint32_t>([[ONE]]);
// FPU-NEXT:   auto [[ARGS_B:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 2>(), 1>();
// FPU-NEXT:   TensorAccessor [[ACC_B:.*]] = TensorAccessor([[ARGS_B]], [[RT_ARG_B]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// FPU:   int32_t [[CB1_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(1));
// FPU-NEXT:   ptrdiff_t [[CB1_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB1_PTR]];
// FPU-NEXT:   size_t [[CB1_PTR_IDX:v[0-9]+]] = (size_t) [[CB1_PTR_PTRDIFF]];
// FPU:   for (size_t [[I_B:.*]] = [[ZERO]]; [[I_B]] < [[BOUND]]; [[I_B]] += [[ONE]]) {
// FPU-NEXT:     for (size_t [[J_B:.*]] = [[ZERO]]; [[J_B]] < [[BOUND]]; [[J_B]] += [[ONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// FPU:       size_t [[TILE_OFF_B_Y:v[0-9]+]] = [[I_B]] * [[BOUND]];
// FPU-NEXT:  size_t [[TILE_OFF_B:v[0-9]+]] = [[TILE_OFF_B_Y]] + [[J_B]];
// Byte offset: tile_offset * page_size + cb_base
// FPU-NEXT:  size_t [[BYTE_OFF_B:v[0-9]+]] = [[TILE_OFF_B]] * [[PAGE_SIZE]];
// FPU-NEXT:  size_t [[CB_ADDR_B_IDX:v[0-9]+]] = [[CB1_PTR_IDX]] + [[BYTE_OFF_B]];
// Cast tile offset and CB address to int32_t for noc_async_read_tile
// FPU-NEXT:  ptrdiff_t [[TILE_OFF_B_PD:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_B]];
// FPU-NEXT:  int32_t [[TILE_OFF_B_I32:v[0-9]+]] = (int32_t) [[TILE_OFF_B_PD]];
// FPU-NEXT:  ptrdiff_t [[CB_ADDR_B_PD:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_B_IDX]];
// FPU-NEXT:  int32_t [[CB_ADDR_B:v[0-9]+]] = (int32_t) [[CB_ADDR_B_PD]];
// FPU-NEXT:  noc_async_read_tile([[TILE_OFF_B_I32]], [[ACC_B]], [[CB_ADDR_B]]);
// FPU:     }
// FPU-NEXT:   }
// FPU-NEXT:   noc_async_read_barrier();
// FPU-NEXT:   return;

// =============================================================================
// FPU path: compute kernel -- binary_op_init_common, add_tiles, exp
// =============================================================================
// FPU-LABEL: // compute_fused
// FPU: void kernel_main() {
// FPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// FPU-DAG:   size_t [[STEP:v[0-9]+]] = 1
// FPU-DAG:   size_t [[CBOUND:v[0-9]+]] = 2
// FPU-DAG:   size_t [[CZERO:v[0-9]+]] = 0

// FPU:       cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// FPU-NEXT:  cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// FPU-NEXT:  cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// FPU-NEXT:  binary_op_init_common(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));

// FPU:       for (size_t [[CI:.*]] = [[CZERO]]; [[CI]] < [[CBOUND]]; [[CI]] += [[STEP]]) {
// FPU-NEXT:    for (size_t [[CJ:.*]] = [[CZERO]]; [[CJ]] < [[CBOUND]]; [[CJ]] += [[STEP]]) {
// FPU:           tile_regs_acquire();
// Linearized CB index for add_tiles: i * 2 + j (2 cols per row)
// FPU:           size_t [[CSTRIDE:v[0-9]+]] = 2;
// FPU-NEXT:      size_t [[CTILE_Y:v[0-9]+]] = [[CI]] * [[CSTRIDE]];
// FPU-NEXT:      size_t [[CTILE_IDX:v[0-9]+]] = [[CTILE_Y]] + [[CJ]];
// No copy_tile for FPU add -- operands read directly from CB
// FPU-NOT:       copy_tile
// FPU:           add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// FPU-NEXT:      add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[CTILE_IDX]], [[CTILE_IDX]], [[CZERO]]);
// FPU-NEXT:      exp_tile_init();
// FPU-NEXT:      exp_tile([[CZERO]]);
// FPU-NEXT:      tile_regs_commit();
// FPU-NEXT:      tile_regs_wait();
// pack_tile reuses the same linearized CB index as add_tiles.
// FPU:           pack_tile<true>([[CZERO]], get_compile_time_arg_val(2), [[CTILE_IDX]]);
// FPU-NEXT:      cb_push_back(get_compile_time_arg_val(2), [[TILES]]);
// FPU-NEXT:      tile_regs_release();

// FPU-NOT:   init_sfpu
// FPU-NOT:   add_binary_tile

// =============================================================================
// FPU path: writer kernel
// =============================================================================
// FPU-LABEL: // writer_unary
// FPU: void kernel_main() {
// FPU-DAG:   size_t [[WBOUND:v[0-9]+]] = 2
// FPU-DAG:   size_t [[WONE:v[0-9]+]] = 1
// FPU-DAG:   size_t [[WPAGE:v[0-9]+]] = 4096
// FPU-DAG:   size_t [[WZERO:v[0-9]+]] = 0
// FPU:   int32_t [[WRT_ARG:.*]] = get_common_arg_val<uint32_t>([[WZERO]]);
// FPU-NEXT:   auto [[WARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 1>(), 0>();
// FPU-NEXT:   TensorAccessor [[WACC:.*]] = TensorAccessor([[WARGS]], [[WRT_ARG]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// FPU:   int32_t [[WR_PTR:.*]] = get_read_ptr(get_compile_time_arg_val(2));
// FPU-NEXT:   ptrdiff_t [[WR_PTR_PD:v[0-9]+]] = (ptrdiff_t) [[WR_PTR]];
// FPU-NEXT:   size_t [[WR_PTR_IDX:v[0-9]+]] = (size_t) [[WR_PTR_PD]];
// FPU:   for (size_t [[WI:.*]] = [[WZERO]]; [[WI]] < [[WBOUND]]; [[WI]] += [[WONE]]) {
// FPU-NEXT:     for (size_t [[WJ:.*]] = [[WZERO]]; [[WJ]] < [[WBOUND]]; [[WJ]] += [[WONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// FPU:       size_t [[WTILE_Y:v[0-9]+]] = [[WI]] * [[WBOUND]];
// FPU-NEXT:  size_t [[WTILE_OFF:v[0-9]+]] = [[WTILE_Y]] + [[WJ]];
// Byte offset: tile_offset * page_size + cb_base
// FPU-NEXT:  size_t [[WBYTE_OFF:v[0-9]+]] = [[WTILE_OFF]] * [[WPAGE]];
// FPU-NEXT:  size_t [[WCB_ADDR_IDX:v[0-9]+]] = [[WR_PTR_IDX]] + [[WBYTE_OFF]];
// Cast tile offset and CB address to int32_t for noc_async_write_tile
// FPU-NEXT:  ptrdiff_t [[WTILE_PD:v[0-9]+]] = (ptrdiff_t) [[WTILE_OFF]];
// FPU-NEXT:  int32_t [[WTILE_I32:v[0-9]+]] = (int32_t) [[WTILE_PD]];
// FPU-NEXT:  ptrdiff_t [[WCB_ADDR_PD:v[0-9]+]] = (ptrdiff_t) [[WCB_ADDR_IDX]];
// FPU-NEXT:  int32_t [[WCB_ADDR:v[0-9]+]] = (int32_t) [[WCB_ADDR_PD]];
// FPU-NEXT:  noc_async_write_tile([[WTILE_I32]], [[WACC]], [[WCB_ADDR]]);
// FPU:     }
// FPU-NEXT:   }
// FPU-NEXT:   noc_async_write_barrier();

// =============================================================================
// SFPU path: reader kernel (same for both paths)
// =============================================================================
// SFPU-LABEL: // reader_binary
// SFPU: void kernel_main() {
// SFPU-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// SFPU-DAG:   size_t [[ONE:v[0-9]+]] = 1
// SFPU-DAG:   size_t [[PAGE_SIZE:v[0-9]+]] = 4096
// SFPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// Read tensor A into CB0
// SFPU:   int32_t [[RT_ARG_A:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// SFPU-NEXT:   auto [[ARGS_A:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 2>(), 0>();
// SFPU-NEXT:   TensorAccessor [[ACC_A:.*]] = TensorAccessor([[ARGS_A]], [[RT_ARG_A]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// SFPU:   int32_t [[CB0_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(0));
// SFPU-NEXT:   ptrdiff_t [[CB0_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB0_PTR]];
// SFPU-NEXT:   size_t [[CB0_PTR_IDX:v[0-9]+]] = (size_t) [[CB0_PTR_PTRDIFF]];
// SFPU:   for (size_t [[I_A:.*]] = [[ZERO]]; [[I_A]] < [[BOUND]]; [[I_A]] += [[ONE]]) {
// SFPU-NEXT:     for (size_t [[J_A:.*]] = [[ZERO]]; [[J_A]] < [[BOUND]]; [[J_A]] += [[ONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// SFPU:       size_t [[TILE_OFF_A_Y:v[0-9]+]] = [[I_A]] * [[BOUND]];
// SFPU-NEXT:  size_t [[TILE_OFF_A:v[0-9]+]] = [[TILE_OFF_A_Y]] + [[J_A]];
// Byte offset: tile_offset * page_size + cb_base
// SFPU-NEXT:  size_t [[BYTE_OFF_A:v[0-9]+]] = [[TILE_OFF_A]] * [[PAGE_SIZE]];
// SFPU-NEXT:  size_t [[CB_ADDR_A_IDX:v[0-9]+]] = [[CB0_PTR_IDX]] + [[BYTE_OFF_A]];
// Cast tile offset and CB address to int32_t for noc_async_read_tile
// SFPU-NEXT:  ptrdiff_t [[TILE_OFF_A_PD:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_A]];
// SFPU-NEXT:  int32_t [[TILE_OFF_A_I32:v[0-9]+]] = (int32_t) [[TILE_OFF_A_PD]];
// SFPU-NEXT:  ptrdiff_t [[CB_ADDR_A_PD:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_A_IDX]];
// SFPU-NEXT:  int32_t [[CB_ADDR_A:v[0-9]+]] = (int32_t) [[CB_ADDR_A_PD]];
// SFPU-NEXT:  noc_async_read_tile([[TILE_OFF_A_I32]], [[ACC_A]], [[CB_ADDR_A]]);
// SFPU:     }
// SFPU-NEXT:   }
// SFPU-NEXT:   noc_async_read_barrier();

// Read tensor B into CB1
// SFPU:   int32_t [[RT_ARG_B:.*]] = get_common_arg_val<uint32_t>([[ONE]]);
// SFPU-NEXT:   auto [[ARGS_B:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 2>(), 1>();
// SFPU-NEXT:   TensorAccessor [[ACC_B:.*]] = TensorAccessor([[ARGS_B]], [[RT_ARG_B]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// SFPU:   int32_t [[CB1_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(1));
// SFPU-NEXT:   ptrdiff_t [[CB1_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB1_PTR]];
// SFPU-NEXT:   size_t [[CB1_PTR_IDX:v[0-9]+]] = (size_t) [[CB1_PTR_PTRDIFF]];
// SFPU:   for (size_t [[I_B:.*]] = [[ZERO]]; [[I_B]] < [[BOUND]]; [[I_B]] += [[ONE]]) {
// SFPU-NEXT:     for (size_t [[J_B:.*]] = [[ZERO]]; [[J_B]] < [[BOUND]]; [[J_B]] += [[ONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// SFPU:       size_t [[TILE_OFF_B_Y:v[0-9]+]] = [[I_B]] * [[BOUND]];
// SFPU-NEXT:  size_t [[TILE_OFF_B:v[0-9]+]] = [[TILE_OFF_B_Y]] + [[J_B]];
// Byte offset: tile_offset * page_size + cb_base
// SFPU-NEXT:  size_t [[BYTE_OFF_B:v[0-9]+]] = [[TILE_OFF_B]] * [[PAGE_SIZE]];
// SFPU-NEXT:  size_t [[CB_ADDR_B_IDX:v[0-9]+]] = [[CB1_PTR_IDX]] + [[BYTE_OFF_B]];
// Cast tile offset and CB address to int32_t for noc_async_read_tile
// SFPU-NEXT:  ptrdiff_t [[TILE_OFF_B_PD:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_B]];
// SFPU-NEXT:  int32_t [[TILE_OFF_B_I32:v[0-9]+]] = (int32_t) [[TILE_OFF_B_PD]];
// SFPU-NEXT:  ptrdiff_t [[CB_ADDR_B_PD:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_B_IDX]];
// SFPU-NEXT:  int32_t [[CB_ADDR_B:v[0-9]+]] = (int32_t) [[CB_ADDR_B_PD]];
// SFPU-NEXT:  noc_async_read_tile([[TILE_OFF_B_I32]], [[ACC_B]], [[CB_ADDR_B]]);
// SFPU:     }
// SFPU-NEXT:   }
// SFPU-NEXT:   noc_async_read_barrier();
// SFPU-NEXT:   return;

// =============================================================================
// SFPU path: compute kernel -- init_sfpu, copy_tile, add_binary_tile, exp
// =============================================================================
// SFPU-LABEL: // compute_fused
// SFPU: void kernel_main() {
// SFPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// SFPU-DAG:   size_t [[STEP:v[0-9]+]] = 1
// SFPU-DAG:   size_t [[CBOUND:v[0-9]+]] = 2
// SFPU-DAG:   size_t [[CZERO:v[0-9]+]] = 0

// SFPU:       cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// SFPU-NEXT:  cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// SFPU-NEXT:  cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// SFPU-NEXT:  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));

// SFPU:       for (size_t [[CI:.*]] = [[CZERO]]; [[CI]] < [[CBOUND]]; [[CI]] += [[STEP]]) {
// SFPU-NEXT:    for (size_t [[CJ:.*]] = [[CZERO]]; [[CJ]] < [[CBOUND]]; [[CJ]] += [[STEP]]) {
// SFPU:           tile_regs_acquire();
// Linearized index for copy_tile CB index (from affine.linearize_index, lowered)
// SFPU:           size_t [[CTILE_Y:v[0-9]+]] = [[CI]] * {{.*}};
// SFPU-NEXT:      size_t [[CTILE_IDX:v[0-9]+]] = [[CTILE_Y]] + [[CJ]];
// SFPU-NEXT:      copy_tile_init(get_compile_time_arg_val(0));
// SFPU-NEXT:      copy_tile(get_compile_time_arg_val(0), [[CTILE_IDX]], [[CZERO]]);
// SFPU-NEXT:      copy_tile_init(get_compile_time_arg_val(1));
// SFPU-NEXT:      copy_tile(get_compile_time_arg_val(1), [[CTILE_IDX]], [[STEP]]);
// SFPU-NEXT:      add_binary_tile_init();
// SFPU-NEXT:      add_binary_tile([[CZERO]], [[STEP]], [[CZERO]]);
// SFPU-NEXT:      exp_tile_init();
// SFPU-NEXT:      exp_tile([[CZERO]]);
// SFPU-NEXT:      tile_regs_commit();
// SFPU-NEXT:      tile_regs_wait();
// SFPU-NEXT:      pack_tile<true>([[CZERO]], get_compile_time_arg_val(2), [[CTILE_IDX]]);
// SFPU-NEXT:      cb_push_back(get_compile_time_arg_val(2), [[TILES]]);
// SFPU-NEXT:      tile_regs_release();

// SFPU-NOT:   binary_op_init_common
// SFPU-NOT:   add_tiles

// =============================================================================
// SFPU path: writer kernel
// =============================================================================
// SFPU-LABEL: // writer_unary
// SFPU: void kernel_main() {
// SFPU-DAG:   size_t [[WBOUND:v[0-9]+]] = 2
// SFPU-DAG:   size_t [[WONE:v[0-9]+]] = 1
// SFPU-DAG:   size_t [[WPAGE:v[0-9]+]] = 4096
// SFPU-DAG:   size_t [[WZERO:v[0-9]+]] = 0
// SFPU:   int32_t [[WRT_ARG:.*]] = get_common_arg_val<uint32_t>([[WZERO]]);
// SFPU-NEXT:   auto [[WARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 1>(), 0>();
// SFPU-NEXT:   TensorAccessor [[WACC:.*]] = TensorAccessor([[WARGS]], [[WRT_ARG]],
// CB pointer casting chain: int32_t -> ptrdiff_t -> size_t
// SFPU:   int32_t [[WR_PTR:.*]] = get_read_ptr(get_compile_time_arg_val(2));
// SFPU-NEXT:   ptrdiff_t [[WR_PTR_PD:v[0-9]+]] = (ptrdiff_t) [[WR_PTR]];
// SFPU-NEXT:   size_t [[WR_PTR_IDX:v[0-9]+]] = (size_t) [[WR_PTR_PD]];
// SFPU:   for (size_t [[WI:.*]] = [[WZERO]]; [[WI]] < [[WBOUND]]; [[WI]] += [[WONE]]) {
// SFPU-NEXT:     for (size_t [[WJ:.*]] = [[WZERO]]; [[WJ]] < [[WBOUND]]; [[WJ]] += [[WONE]]) {
// Tile offset: linearize 2D index (i * bound + j)
// SFPU:       size_t [[WTILE_Y:v[0-9]+]] = [[WI]] * [[WBOUND]];
// SFPU-NEXT:  size_t [[WTILE_OFF:v[0-9]+]] = [[WTILE_Y]] + [[WJ]];
// Byte offset: tile_offset * page_size + cb_base
// SFPU-NEXT:  size_t [[WBYTE_OFF:v[0-9]+]] = [[WTILE_OFF]] * [[WPAGE]];
// SFPU-NEXT:  size_t [[WCB_ADDR_IDX:v[0-9]+]] = [[WR_PTR_IDX]] + [[WBYTE_OFF]];
// Cast tile offset and CB address to int32_t for noc_async_write_tile
// SFPU-NEXT:  ptrdiff_t [[WTILE_PD:v[0-9]+]] = (ptrdiff_t) [[WTILE_OFF]];
// SFPU-NEXT:  int32_t [[WTILE_I32:v[0-9]+]] = (int32_t) [[WTILE_PD]];
// SFPU-NEXT:  ptrdiff_t [[WCB_ADDR_PD:v[0-9]+]] = (ptrdiff_t) [[WCB_ADDR_IDX]];
// SFPU-NEXT:  int32_t [[WCB_ADDR:v[0-9]+]] = (int32_t) [[WCB_ADDR_PD]];
// SFPU-NEXT:  noc_async_write_tile([[WTILE_I32]], [[WACC]], [[WCB_ADDR]]);
// SFPU:     }
// SFPU-NEXT:   }
// SFPU-NEXT:   noc_async_write_barrier();

// Reader kernel: reads A and B from DRAM, pushes to CB0 and CB1
func.func @reader_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, %b: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Copy A to CB0
  %slice_a = ttl.tensor_slice %a[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_a = ttl.copy %slice_a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  // Copy B to CB1
  %slice_b = ttl.tensor_slice %b[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_b = ttl.copy %slice_b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_b : !ttl.transfer_handle<read>

  func.return
}

// Compute kernel: reads from CB0, CB1, computes f(A+B), writes to CB2
func.func @compute_fused(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                         %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  // Wait for inputs from reader thread
  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Fused computation: f(A + B) where f is exp
  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// Writer kernel: pops from CB2, writes to DRAM
func.func @writer_unary(%out: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Wait for data from compute thread (must match CB shape)
  %cb2_view = ttl.cb_wait %cb2 : <[2, 2], f32, 2> -> tensor<2x2xf32>

  // Copy from CB2 to output tensor
  %slice_out = ttl.tensor_slice %out[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_out = ttl.copy %cb2, %slice_out : (!ttl.cb<[2, 2], f32, 2>, tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
  ttl.wait %xf_out : !ttl.transfer_handle<write>

  func.return
}
