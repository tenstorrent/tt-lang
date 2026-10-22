// Verify global qualification of TT-Lang helpers, types, and selected definitions.
// RUN: ttlang-opt --convert-ttkernel-to-emitc --split-input-file %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp --split-input-file %t.emitc.mlir -o %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Generic calls and the SFPI verbatim wrapper retain their helper definitions.
// EMITC-LABEL: func.func @compute_helpers
// EMITC: emitc.call_opaque "::experimental::unpack_stall_on_pack"
// EMITC: emitc.call_opaque "::experimental::tilize_block"
// EMITC: emitc.call_opaque "::experimental::untilize_block"
// EMITC: emitc.call_opaque "::experimental::pack_untilize_block"
// EMITC: emitc.call_opaque "::experimental::matmul_block"
// EMITC: emitc.verbatim "::experimental::invoke_sfpi([=]() {"
// EMITC-NEXT: emitc.verbatim "sfpi_body();"
// EMITC-NEXT: emitc.verbatim "});"
// CPP-DAG: #include "api/compute/pack_untilize.h"
// CPP-DAG: #include "api/compute/tilize.h"
// CPP-DAG: #include "api/compute/untilize.h"
// CPP-DAG: ALWI void unpack_stall_on_pack()
// CPP-DAG: ALWI void tilize_block(
// CPP-DAG: ALWI void untilize_block(
// CPP-DAG: ALWI void pack_untilize_block(
// CPP-DAG: ALWI void matmul_block(
// CPP-DAG: inline void invoke_sfpi(
// CPP-LABEL: void kernel_main()
// CPP: ::experimental::unpack_stall_on_pack();
// CPP-NEXT: ::experimental::tilize_block(
// CPP-NEXT: ::experimental::untilize_block(
// CPP-NEXT: ::experimental::pack_untilize_block<1, 1>(
// CPP-NEXT: ::experimental::matmul_block(
// CPP-NEXT: ::experimental::invoke_sfpi([=]() {
// CPP-NEXT: sfpi_body();
// CPP-NEXT: });
func.func @compute_helpers() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %scalar = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1024, bf16>
  %tile = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  ttkernel.experimental.unpack_stall_on_pack
  ttkernel.experimental.tilize_block(%scalar, %tile, %one, %one)
      : (!ttkernel.cb<1024, bf16>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32, i32) -> ()
  ttkernel.experimental.untilize_block(%tile, %scalar, %one, %one)
      : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1024, bf16>, i32, i32) -> ()
  ttkernel.experimental.pack_untilize_block(%tile, %scalar, %one, %one)
      {colsPerDstPass = 1 : i32, totalColTiles = 1 : i32}
      : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1024, bf16>, i32, i32) -> ()
  "ttkernel.experimental.matmul_block"(%tile, %tile, %zero, %zero, %zero, %zero, %one, %one, %one, %one)
      : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  ttkernel.invoke_sfpi {
    emitc.verbatim "sfpi_body();"
  }
  return
}

// -----

// The fabric manager type and its generic helper calls use the same namespace.
// EMITC-LABEL: func.func @fabric_helpers
// EMITC: emitc.verbatim "::experimental::FabricConnectionManager fabric_connection_manager_0;"
// EMITC: !emitc.opaque<"::experimental::FabricConnectionManager">
// EMITC: emitc.call_opaque "::experimental::setup_fabric_connections"
// EMITC: emitc.call_opaque "::experimental::close_fabric_connections"
// CPP: struct FabricConnectionManager {
// CPP: setup_fabric_connections(FabricConnectionManager &fabric_connection_manager)
// CPP: close_fabric_connections(FabricConnectionManager &fabric_connection_manager)
// CPP-LABEL: void kernel_main()
// CPP-NEXT: ::experimental::FabricConnectionManager [[MANAGER:fabric_connection_manager_[0-9]+]];
// CPP-NEXT: ::experimental::setup_fabric_connections([[MANAGER]]);
// CPP-NEXT: ::experimental::close_fabric_connections([[MANAGER]]);
func.func @fabric_helpers() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %manager = ttkernel.experimental.create_fabric_connection_manager()
      : () -> !ttkernel.fabric_connection_manager
  ttkernel.experimental.setup_fabric_connections(%manager)
      : (!ttkernel.fabric_connection_manager) -> ()
  ttkernel.experimental.close_fabric_connections(%manager)
      : (!ttkernel.fabric_connection_manager) -> ()
  return
}

// -----

// Coordinate translation and padding calls select their dataflow definitions.
// EMITC-LABEL: func.func @dataflow_helpers
// EMITC: %[[X:.*]] = emitc.call_opaque "::experimental::convert_logical_x_to_translated"
// EMITC: %[[Y:.*]] = emitc.call_opaque "::experimental::convert_logical_y_to_translated"
// EMITC: emitc.call_opaque "::experimental::write_row_mask_tile"(%[[X]],
// EMITC: emitc.call_opaque "::experimental::write_col_mask_tile"(%[[Y]],
// EMITC: emitc.call_opaque "::experimental::fill_arange_tile"
// CPP-DAG: std::uint32_t convert_logical_x_to_translated(std::uint32_t logical_x)
// CPP-DAG: std::uint32_t convert_logical_y_to_translated(std::uint32_t logical_y)
// CPP-DAG: ALWI void write_row_mask_tile(
// CPP-DAG: ALWI void write_col_mask_tile(
// CPP-DAG: ALWI void fill_arange_tile(
// CPP-LABEL: void kernel_main()
// CPP: int32_t [[X:v[0-9]+]] = ::experimental::convert_logical_x_to_translated(
// CPP-NEXT: int32_t [[Y:v[0-9]+]] = ::experimental::convert_logical_y_to_translated(
// CPP-NEXT: ::experimental::write_row_mask_tile<DataFormat::Float16_b>([[X]],
// CPP-NEXT: ::experimental::write_col_mask_tile<DataFormat::Float16_b>([[Y]],
// CPP-NEXT: ::experimental::fill_arange_tile<DataFormat::Float16_b>(
func.func @dataflow_helpers() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %tile = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %one = arith.constant 1 : i32
  %x = ttkernel.experimental.convert_logical_x_to_translated(%one) : (i32) -> i32
  %y = ttkernel.experimental.convert_logical_y_to_translated(%one) : (i32) -> i32
  ttkernel.experimental.write_row_mask_tile(%x, %tile)
      : (i32, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
  ttkernel.experimental.write_col_mask_tile(%y, %tile)
      : (i32, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
  ttkernel.experimental.fill_arange_tile(%tile)
      : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
  return
}
