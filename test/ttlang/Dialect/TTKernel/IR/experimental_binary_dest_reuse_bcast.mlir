// Verifies EmitC lowering for column-broadcast multiplication with DST reuse.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc | FileCheck %s

// The input DFB tile is column-broadcast and multiplied into the selected DST
// slot after an initialization with the same operation configuration.
module {
  func.func @multiply_column()
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %input_index = arith.constant 0 : index
    %dst_index = arith.constant 3 : index
    // CHECK-LABEL: func.func @multiply_column
    // CHECK:       emitc.call_opaque "experimental::binary_dest_reuse_bcast_tiles_init"
    ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(
        %input, <mul>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    // CHECK-NEXT:  emitc.call_opaque "experimental::binary_dest_reuse_bcast_tiles"
    ttkernel.experimental.binary_dest_reuse_bcast_tiles(
        %input, %input_index, %dst_index, <mul>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    return
  }
}
