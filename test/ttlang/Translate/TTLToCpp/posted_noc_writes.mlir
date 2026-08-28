// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: Verifies translation of posted payload, stateful one-packet,
// inline, and flush operations to their tt-metal NoC APIs.

func.func @posted_noc_writes() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %noc = arith.constant 0 : i8
  %dst_addr = arith.constant 4096 : i32
  %src_addr = arith.constant 8192 : i32
  %size = arith.constant 896 : i32
  %value = arith.constant 1 : i32
  %byte_enable = arith.constant 15 : i8
  %dst_noc_addr = ttkernel.get_noc_addr(%c0, %c0, %dst_addr, %noc) : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write %src_addr, core[%c0, %c0], %dst_addr, %size, noc %noc posted true : (i32, index, index, i32, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_set_state(%dst_noc_addr, %size, noc %noc) posted true : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(%src_addr, %dst_addr, noc %noc) posted true : (i32, i32, i8) -> ()
  ttkernel.noc_inline_dw_write(core[%c0, %c0], %dst_addr, %value, %byte_enable, noc %noc) posted true : (index, index, i32, i32, i8, i8) -> ()
  ttkernel.noc_async_writes_flushed(%noc) posted true : (i8) -> ()
  func.return
}

// CPP-LABEL: void kernel_main() {
// CPP: noc0.async_write<NocOptions::POSTED>
// CPP: noc_async_write_one_packet_set_state<true>
// CPP: noc_async_write_one_packet_with_state<true>
// CPP: noc0.inline_dw_write<NocOptions::INLINE_L1 | NocOptions::POSTED>
// CPP: noc0.async_writes_flushed<NocOptions::POSTED>();
