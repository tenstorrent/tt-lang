// Verify invalid targets, policy constraints, and execution strategies are
// diagnosed before kernel configuration mutates the IR.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{matmul-full-fp32=0 reduce-full-fp32=0})' --split-input-file --verify-diagnostics

// expected-error @below {{'builtin.module' op ttl.target_arch must be a #ttcore.arch attribute}}
module attributes {ttl.target_arch = "blackhole"} {
  func.func @malformed_target() {
    return
  }
}

// -----

// Quasar requires a Gen2 configuration descriptor and kernel launch API.
// expected-error @below {{'builtin.module' op Quasar compute kernels require the Gen2 configuration and launch APIs, which are not supported by the current TT-Lang runtime}}
module attributes {ttl.target_arch = #ttcore.arch<quasar>} {
  func.func @unsupported_quasar_target() {
    return
  }
}

// -----

func.func @disabled_required_f32() attributes {fp32_dest_acc_en = false} {
  %zero = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_fill' op requires 32-bit destination elements, but fp32 destination accumulation is explicitly disabled}}
  %value = ttl.tile_fill 1.0 into dst[%zero]
      : !ttcore.tile<32x32, f32>
  return
}

// -----

// expected-error @below {{'func.func' op fp32_dest_acc_en must be a boolean attribute}}
func.func @malformed_boolean_policy() attributes {fp32_dest_acc_en = "true"} {
  return
}

// -----

// expected-error @below {{'func.func' op ttl.unpack_to_dest_fp32 must be a dense i32 array attribute}}
func.func @malformed_unpack_policy()
    attributes {ttl.unpack_to_dest_fp32 = [0 : i32]} {
  return
}

// -----

// expected-error @below {{'func.func' op ttl.unpack_to_dest_fp32 must contain dataflow buffer indices in range [0, 31] for the conservative 32-DFB-index capacity used when target metadata is absent}}
func.func @out_of_range_unpack_policy()
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 32>} {
  return
}

// -----

func.func @explicit_unpack_excludes_required_dfb(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %input_tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.tile_abs' op dataflow buffer 0 requires unpack-to-DST-f32 mode, but ttl.unpack_to_dest_fp32 excludes this index}}
  %absolute = ttl.tile_abs %input_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return
}

// -----

func.func @strategy_on_fixed_operation() {
  %zero = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_fill' op ttl.tile_execution_strategy is only valid on tile operations with execution-strategy alternatives}}
  %value = ttl.tile_fill 1.0 into dst[%zero]
      {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>}
      : !ttcore.tile<32x32, f32>
  return
}

// -----

func.func @malformed_strategy(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_add' op ttl.tile_execution_strategy must be a #ttl.tile_execution_strategy attribute}}
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      {ttl.tile_execution_strategy = "fpu"}
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

func.func @explicit_fpu_conflicts_with_policy(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.enable_fpu_binary_ops = false} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_add' op explicit FPU strategy conflicts with disabled FPU binary policy}}
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>}
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// The requested target must agree with the device selected from the system
// description.
// expected-error @below {{'builtin.module' op ttl.target_arch does not match the selected device arch}}
module attributes {
  ttl.target_arch = #ttcore.arch<blackhole>,
  ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 204800, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>
} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @target_mismatch() {
    return
  }
}

// -----

// An explicit FPU strategy requires operands that address the same tile.
func.func @explicit_strategy_illegal_for_operands(
    %lhs: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x2x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %lhs_tile = tensor.extract %lhs_attached[%zero, %zero]
      : tensor<1x2x!ttcore.tile<32x32, bf16>>
  %rhs_tile = tensor.extract %rhs_attached[%zero, %one]
      : tensor<1x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_add' op explicit ttl.tile_execution_strategy is not legal for its operands}}
  %sum = ttl.tile_add %lhs_tile, %rhs_tile into dst[%zero]
      {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>}
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Attached DFB handles must resolve to a physical index before analysis.
func.func @dataflow_buffer_without_finalized_index(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %input_dfb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) {
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %input_tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.tile_exp' op uses a dataflow buffer without a finalized index}}
  %exp = ttl.tile_exp %input_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Physical DFB indices outside the kernel configuration domain are invalid.
func.func @dataflow_buffer_index_out_of_range(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %input_dfb = ttl.bind_cb {cb_index = 32, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %input_tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.tile_exp' op uses dataflow buffer index 32 outside the supported range [0, 31] for the conservative 32-DFB-index capacity used when target metadata is absent}}
  %exp = ttl.tile_exp %input_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Kernel configuration rejects malformed tile-operation operands without
// aborting the compiler.
func.func @non_tile_operand(%input: f32, %output: f32) {
  %zero = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_bcast' op expected a tile or tensor-of-tiles operand, got 'f32'}}
  %broadcast = ttl.tile_bcast %input, %output 2 : i32 into dst[%zero]
      : (f32, f32) -> f32
  return
}

// -----

// Device chip IDs must refer to entries in the system description.
// expected-error @below {{'builtin.module' op default device selects chip 1 outside the system description}}
module attributes {
  ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 204800, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>
} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [1]>
  func.func @invalid_device_chip() {
    return
  }
}
