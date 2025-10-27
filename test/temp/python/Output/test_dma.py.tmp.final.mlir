#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073185472, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1], [1 : i32, 0 : i32], [ 0x0x0x0]>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      ttcore.global @lhs = memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1> [0]
      func.func @test_dma_ops(%arg0: memref<32x32xf32> {d2m.stream = true}, %arg1: memref<32x32xf32> {d2m.stream = false}, %arg2: memref<32x32xf32> {d2m.stream = false}) -> memref<32x32xf32> {
        %0 = "ttmetal.create_buffer"() <{address = 101152 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %1 = "ttmetal.create_buffer"() <{address = 105248 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg0, %1) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%1, %0, %1, %0) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel1, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel2, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%1) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %2 = "ttmetal.create_buffer"() <{address = 113440 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %3 = "ttmetal.create_buffer"() <{address = 105248 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %4 = "ttmetal.create_buffer"() <{address = 109344 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg1, %4) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%4, %3, %4, %3) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel3, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel4, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel5, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%4) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %5 = "ttmetal.create_buffer"() <{address = 109344 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        "ttmetal.enqueue_program"(%0, %3, %5, %2, %3, %5) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel6, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <buffer_address[0]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel7, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc1>, #ttmetal.compute_config<@compute_kernel8, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 3, 3>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%3) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%0) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %6 = "ttmetal.create_buffer"() <{address = 101152 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_program"(%5, %6, %5, %6) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel9, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel10, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc1>, #ttmetal.compute_config<@compute_kernel11, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%5) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.enqueue_read_buffer"(%6, %arg2) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<32x32xf32>) -> ()
        "ttmetal.finish"() : () -> ()
        "ttmetal.deallocate_buffer"(%6) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        return %arg2 : memref<32x32xf32>
      }
      func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel1() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel2() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %3 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %3 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%1, %2) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel3() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel4() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel5() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %3 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %3 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%1, %2) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel6() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = buffer_address, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %7 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %7 : i32
        }
        %1 = emitc.expression  : () -> i32 {
          %7 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
          yield %7 : i32
        }
        %2 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 18 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %3 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        %4 = emitc.literal "get_compile_time_arg_val(3)" : i32
        %5 = emitc.expression %2, %2, %4 : (!emitc.size_t, !emitc.size_t, i32) -> i64 {
          %7 = call_opaque "get_noc_addr"(%2, %2, %4) : (!emitc.size_t, !emitc.size_t, i32) -> i64
          yield %7 : i64
        }
        %6 = emitc.expression %3 : (!emitc.opaque<"::tt::CB">) -> i32 {
          %7 = call_opaque "get_write_ptr"(%3) : (!emitc.opaque<"::tt::CB">) -> i32
          yield %7 : i32
        }
        emitc.call_opaque "noc_async_read"(%5, %6, %1) : (i64, i32, i32) -> ()
        emitc.call_opaque "noc_async_read_barrier"() : () -> ()
        emitc.call_opaque "cb_push_back"(%3, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel7() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        return
      }
      func.func private @compute_kernel8() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %1 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %2 = emitc.expression  : () -> !emitc.size_t {
          %7 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
          yield %7 : !emitc.size_t
        }
        %3 = emitc.expression  : () -> i32 {
          %7 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %7 : i32
        }
        %4 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %5 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %6 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "init_sfpu"(%4, %6) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "cb_wait_front"(%4, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "tile_regs_acquire"() : () -> ()
        emitc.call_opaque "copy_tile_init"(%4) : (!emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "copy_tile"(%4, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "copy_tile_init"(%5) : (!emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "copy_tile"(%5, %0, %1) : (!emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "add_binary_tile_init"() : () -> ()
        emitc.call_opaque "add_binary_tile"(%0, %1, %2) : (!emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_commit"() : () -> ()
        emitc.call_opaque "tile_regs_wait"() : () -> ()
        emitc.call_opaque "pack_tile"(%2, %6, %0) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_release"() : () -> ()
        emitc.call_opaque "cb_wait_front"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%4, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%5, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%6, %3) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel9() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel10() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = emitc.expression  : () -> i32 {
          %2 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %2 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel11() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = emitc.expression  : () -> i32 {
          %3 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
          yield %3 : i32
        }
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "compute_kernel_hw_startup"(%1, %2) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "untilize_init"(%1) : (!emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::untilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
    }
  }
}

