// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores,canonicalize,cse,ttkernel-analyze-dfb-resources)' | FileCheck %s

// Surviving DFB uses are unioned across specialized RISC functions. Omitted
// scope keeps conservative legacy placement, while explicitly local and
// remote-uniform DFBs use their surviving participant sets.

// CHECK: ttl.per_core_dfb_configs =
// CHECK-SAME: address_scope = "legacy", dfb_index = 0 : i32, num_pages = 2 : i32
// CHECK-SAME: address_scope = "local", dfb_index = 1 : i32, num_pages = 1 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 2 : i32, num_pages = 2 : i32
// CHECK-SAME: address_scope = "local", dfb_index = 3 : i32, num_pages = 3 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 4 : i32, num_pages = 1 : i32
// CHECK-SAME: address_scope = "legacy", dfb_index = 0 : i32, num_pages = 2 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 2 : i32, num_pages = 4 : i32
// CHECK-SAME: address_scope = "local", dfb_index = 3 : i32, num_pages = 5 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 4 : i32, num_pages = 1 : i32
// CHECK-SAME: address_scope = "legacy", dfb_index = 0 : i32, num_pages = 2 : i32
// CHECK-SAME: address_scope = "local", dfb_index = 3 : i32, num_pages = 5 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 4 : i32, num_pages = 1 : i32

module attributes {
  ttl.launch_grid = [3 : i64, 2 : i64],
  ttl.logical_dfb_configs = [
    {element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 1 : i32, num_pages = 1 : i32, physical_index = 1 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "remote_uniform", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 2 : i32, num_pages = 2 : i32, physical_index = 2 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "remote_uniform", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 3 : i32, num_pages = 4 : i32, physical_index = 2 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 4 : i32, num_pages = 3 : i32, physical_index = 3 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 5 : i32, num_pages = 5 : i32, physical_index = 3 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "remote_uniform", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 6 : i32, num_pages = 1 : i32, physical_index = 4 : i32, unpack_to_dest_fp32 = false}
  ]
} {
  func.func @legacy_and_local() {
    %c0 = arith.constant 0 : index
    %x = ttkernel.my_logical_x_ : () -> index
    %active = arith.cmpi eq, %x, %c0 : index
    scf.if %active {
      %legacy = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 0 : i64} : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
      %local = ttkernel.get_compile_time_arg_val(1) {ttl.dfb_logical_index = 1 : i64} : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%legacy, %local) {header = "use.hpp"} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
    }
    return
  }

  func.func @remote_uniform_capacity() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %x = ttkernel.my_logical_x_ : () -> index
    %small_core = arith.cmpi eq, %x, %c0 : index
    %large_core = arith.cmpi eq, %x, %c1 : index
    scf.if %small_core {
      %small = ttkernel.get_compile_time_arg_val(2) {ttl.dfb_logical_index = 2 : i64} : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%small) {header = "use.hpp"} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    }
    scf.if %large_core {
      %large = ttkernel.get_compile_time_arg_val(2) {ttl.dfb_logical_index = 3 : i64} : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%large) {header = "use.hpp"} : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
    }
    return
  }

  func.func @local_capacity() {
    %c0 = arith.constant 0 : index
    %x = ttkernel.my_logical_x_ : () -> index
    %active = arith.cmpi eq, %x, %c0 : index
    scf.if %active {
      %small = ttkernel.get_compile_time_arg_val(3) {ttl.dfb_logical_index = 4 : i64} : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%small) {header = "use.hpp"} : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>) -> ()
    } else {
      %large = ttkernel.get_compile_time_arg_val(3) {ttl.dfb_logical_index = 5 : i64} : () -> !ttkernel.cb<5, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%large) {header = "use.hpp"} : (!ttkernel.cb<5, !ttcore.tile<32x32, bf16>>) -> ()
    }
    return
  }

  func.func @whole_grid() {
    %cb = ttkernel.get_compile_time_arg_val(4) {ttl.dfb_logical_index = 6 : i64} : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    ttkernel.opaque_call "use"(%cb) {header = "use.hpp"} : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// A pinned DFB keeps its backing and descriptor across a preserving seam but
// is absent from that seam's reset table. A later ordinary reset includes it.
// The metadata-only reset operand on core 1 does not create backing there.

// CHECK: ttl.per_core_dfb_configs =
// CHECK-SAME: configs = [{{.*}}address_scope = "local", dfb_index = 0 : i32, num_pages = 2 : i32{{.*}}address_scope = "local", dfb_index = 1 : i32, num_pages = 1 : i32{{.*}}core_coords = {{\[}}[0, 0]]
// CHECK-SAME: configs = [{{.*}}address_scope = "local", dfb_index = 1 : i32, num_pages = 1 : i32{{.*}}core_coords = {{\[}}[1, 0]]
// CHECK-LABEL: func.func @preserved_boundary_c0_0
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"({{.*}}) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 1, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5]}
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5]}
// CHECK-LABEL: func.func @preserved_boundary_c1_0
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"({{.*}}) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 1, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5]}
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0]}

module attributes {
  ttl.launch_grid = [2 : i64, 1 : i64],
  ttl.logical_dfb_configs = [
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 1 : i32, logical_index = 1 : i32, num_pages = 1 : i32, physical_index = 1 : i32, unpack_to_dest_fp32 = false}
  ]
} {
  func.func @preserved_boundary() {
    %c0 = arith.constant 0 : index
    %x = ttkernel.my_logical_x_ : () -> index
    %active = arith.cmpi eq, %x, %c0 : index
    scf.if %active {
      %held = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 0 : i64} : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
      ttkernel.opaque_call "use"(%held) {header = "use.hpp"} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    }
    %next = ttkernel.get_compile_time_arg_val(1) {ttl.dfb_logical_index = 1 : i64} : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    ttkernel.opaque_call "use"(%next) {header = "use.hpp"} : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
    %metadata = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 0 : i64} : () -> i32
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"(%metadata) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 1, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = [0 : i64]} : (i32) -> ()
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []} : () -> ()
    return
  }
}

// -----

// The preserved DFB stays on physical slot 0 in epochs 0 and 1. Once it is
// absent in epoch 2, the larger epoch-local DFB reuses that backing. Slot 1
// therefore remains one page instead of growing to seven pages.

// CHECK: ttl.dfb_epoch_physical_configs = [
// CHECK-SAME: dfb_index = 0 : i32, {{[^}]*}}total_size = 14336 : i64
// CHECK-SAME: dfb_index = 1 : i32, {{[^}]*}}total_size = 2048 : i64
// CHECK: ttl.logical_dfb_configs = [
// CHECK-SAME: epoch = 0 : i32, logical_index = 0 : i32, num_pages = 1 : i32, physical_index = 1 : i64
// CHECK-SAME: epoch = 1 : i32, logical_index = 1 : i32, num_pages = 1 : i32, physical_index = 1 : i64
// CHECK-SAME: epoch = 0 : i32, logical_index = 2 : i32, num_pages = 7 : i32, physical_index = 0 : i64
// CHECK-SAME: epoch = 2 : i32, logical_index = 3 : i32, num_pages = 7 : i32, physical_index = 0 : i64
// CHECK: ttl.per_core_dfb_configs =
// CHECK-SAME: dfb_index = 0 : i32, num_pages = 7 : i32
// CHECK-SAME: dfb_index = 1 : i32, num_pages = 1 : i32
// CHECK-LABEL: func.func @reuse_preserved_backing
// Epoch 0 configures the local slot and the preserved slot.
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {{.*}}template_args = [2, 1, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5, 0, 14336, 7, 2048, 5, 32, 32, 16, 4, 5, 5]
// Epoch 1 keeps physical slot 0 intact and reconfigures only slot 1.
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"({{.*}}) {{.*}}template_args = [1, 1, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5]
// Epoch 2 configures the large local DFB on the now-available slot 0 once.
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {{.*}}template_args = [1, 0, 14336, 7, 2048, 5, 32, 32, 16, 4, 5, 5]

module attributes {
  ttl.launch_grid = [1 : i64, 1 : i64],
  ttl.logical_dfb_configs = [
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 1 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 1 : i32, logical_index = 1 : i32, num_pages = 1 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 2 : i32, num_pages = 7 : i32, physical_index = 1 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "local", element_type = !ttcore.tile<32x32, bf16>, epoch = 2 : i32, logical_index = 3 : i32, num_pages = 7 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}
  ],
  ttl.dfb_epoch_physical_configs = [
    {dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, tile_height = 32 : i32, tile_width = 32 : i32, total_size = 14336 : i64},
    {dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, tile_height = 32 : i32, tile_width = 32 : i32, total_size = 14336 : i64}
  ]
} {
  func.func @reuse_preserved_backing() {
    %local0 = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 0 : i64} : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %local1 = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 1 : i64} : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %held = ttkernel.get_compile_time_arg_val(1) {ttl.dfb_logical_index = 2 : i64} : () -> !ttkernel.cb<7, !ttcore.tile<32x32, bf16>>
    %local2 = ttkernel.get_compile_time_arg_val(0) {ttl.dfb_logical_index = 3 : i64} : () -> !ttkernel.cb<7, !ttcore.tile<32x32, bf16>>
    ttkernel.opaque_call "use"(%local0, %local1, %held, %local2) {header = "use.hpp"} : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<7, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<7, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [2, 0, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5, 1, 14336, 7, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []} : () -> ()
    %metadata = ttkernel.get_compile_time_arg_val(1) {ttl.dfb_logical_index = 2 : i64} : () -> i32
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"(%metadata) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 0, 2048, 1, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = [1 : i64]} : (i32) -> ()
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [2, 0, 14336, 7, 2048, 5, 32, 32, 16, 4, 5, 5, 1, 14336, 7, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 2 : i32, ttl.dfb_reset_preserved_indices = []} : () -> ()
    return
  }
}

// -----

// Capacity is measured in bytes across reset epochs, then rounded up to pages
// of the physical descriptor's initial format. A legacy logical DFB contributes
// its own capacity everywhere without defeating a remote-uniform logical DFB
// that reuses the same physical slot in a later epoch.

// CHECK: ttl.per_core_dfb_configs =
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 3 : i32, num_pages = 4 : i32
// CHECK-SAME: address_scope = "remote_uniform", dfb_index = 3 : i32, num_pages = 3 : i32
// CHECK-LABEL: func.func @later_epoch_c0_0
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 3, 7168, 7, 1024, 5, 16, 32, 16, 2, 5, 5]}
// CHECK-LABEL: func.func @later_epoch_c1_0
// CHECK: ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0]}

module attributes {
  ttl.launch_grid = [2 : i64, 1 : i64],
  ttl.logical_dfb_configs = [
    {element_type = !ttcore.tile<32x32, bf16>, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 3 : i32, physical_index = 3 : i32, unpack_to_dest_fp32 = false},
    {address_scope = "remote_uniform", element_type = !ttcore.tile<16x32, bf16>, epoch = 1 : i32, logical_index = 1 : i32, num_pages = 7 : i32, physical_index = 3 : i32, unpack_to_dest_fp32 = false}
  ]
} {
  func.func @initial_epoch() {
    %cb = ttkernel.get_compile_time_arg_val(3) {ttl.dfb_logical_index = 0 : i64} : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
    ttkernel.opaque_call "use"(%cb) {header = "use.hpp"} : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }

  func.func @later_epoch() {
    %c0 = arith.constant 0 : index
    %x = ttkernel.my_logical_x_ : () -> index
    %active = arith.cmpi eq, %x, %c0 : index
    scf.if %active {
      %cb = ttkernel.get_compile_time_arg_val(3) {ttl.dfb_logical_index = 1 : i64} : () -> !ttkernel.cb<7, !ttcore.tile<16x32, bf16>>
      ttkernel.opaque_call "use"(%cb) {header = "use.hpp"} : (!ttkernel.cb<7, !ttcore.tile<16x32, bf16>>) -> ()
    }
    ttkernel.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [1, 3, 7168, 7, 1024, 5, 16, 32, 16, 2, 5, 5], ttl.dfb_reset_epoch = 1 : i32} : () -> ()
    return
  }
}
