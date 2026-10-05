// RUN: ttlang-opt %s --split-input-file --ttkernel-verify-hardware-config --verify-diagnostics
// Summary: ttkernel-verify-hardware-config diagnostics for operations outside
// the generated matrix: calls and regions the analysis does not model.

// A call without declared effects resets the MATH configuration.
func.func private @helper()

func.func @call_resets_configuration() {
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile_init() : () -> ()
  // expected-note @below {{MATH configuration reset here}}
  func.call @helper() : () -> ()
  // expected-error @below {{requires MATH configuration from 'ttkernel.exp_tile_init' but it is not established on every incoming path}}
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// An unmodeled region starts from and exits with an unknown configuration.
func.func @unmodeled_region_resets_configuration() {
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile_init() : () -> ()
  scf.execute_region {
    // expected-error @below {{requires MATH configuration from 'ttkernel.exp_tile_init' but it is not established on every incoming path}}
    ttkernel.exp_tile(%c0) : (index) -> ()
    scf.yield
  }
  // expected-error @below {{requires MATH configuration from 'ttkernel.exp_tile_init' but it is not established on every incoming path}}
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// Inits that share an operation but differ in key operands are distinct.
func.func @transpose_requires_matching_input() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-note @below {{configured here}}
  ttkernel.transpose_wh_init(%cb1, %cb0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
  // expected-error @below {{requires MATH configuration from 'ttkernel.transpose_wh_init' but it is configured with different operands or attributes}}
  ttkernel.transpose_wh_tile(%cb0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  func.return
}

// -----

// The block K dimension is part of the matmul init key.
func.func @matmul_block_kt_mismatch() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %transpose = arith.constant 0 : i32
  %ct = arith.constant 1 : i32
  %rt = arith.constant 1 : i32
  %kt0 = arith.constant 1 : i32
  %kt1 = arith.constant 2 : i32
  // expected-note @below {{configured here}}
  "ttkernel.mm_block_init_short"(%cb0, %cb1, %transpose, %ct, %rt, %kt0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, i32, i32, i32, i32) -> ()
  // expected-error @below {{requires MATH configuration from 'ttkernel.mm_block_init_short' but it is configured with different operands or attributes}}
  ttkernel.matmul_block(%cb0, %cb1, %c0, %c0, %c0, %transpose, %ct, %rt, %kt1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index, i32, i32, i32, i32) -> ()
  func.return
}

// -----

// Row normalization programs MATH inside its LLK, so a prior init does not
// survive it.
func.func @row_normalization_resets_configuration() {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile_init() : () -> ()
  // expected-note @below {{MATH configuration reset here}}
  ttkernel.experimental_row_normalization_block(%cb, %cb, %cb) num_tiles = 1 scale = 1.000000e+00 epsilon = 1.000000e-05 has_gamma = false dtype = <bf16> : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
  // expected-error @below {{requires MATH configuration from 'ttkernel.exp_tile_init' but it is not established on every incoming path}}
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// invoke_sfpi has a non-scf region, so the analysis exits unknown without a
// writer note. The reset effect is what init insertion consults.
func.func @invoke_sfpi_resets_configuration() {
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile_init() : () -> ()
  ttkernel.invoke_sfpi {
  }
  // expected-error @below {{requires MATH configuration from 'ttkernel.exp_tile_init' but it is not established on every incoming path}}
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}
