// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: Verifies posted and non-posted payload, stateful one-packet,
// inline, and departure-wait operations lower to the corresponding NoC APIs.

func.func @posted_noc_writes() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %coordinate = arith.constant 0 : index
  %noc = arith.constant 0 : i8
  %bank = arith.constant 1 : i32
  %destination_address = arith.constant 4096 : i32
  %source_address = arith.constant 8192 : i32
  %size = arith.constant 896 : i32
  %value = arith.constant 1 : i32
  %byte_enable = arith.constant 15 : i8
  %local_semaphore = ttkernel.get_semaphore(%coordinate)
      : (index) -> !ttkernel.local_semaphore
  %typed_address = ttkernel.cast_to_l1_addr %local_semaphore
      : !ttkernel.local_semaphore to !ttkernel.l1_addr
  %destination_noc_address = ttkernel.get_noc_addr(
      %coordinate, %coordinate, %destination_address, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr

  ttkernel.noc_async_write %source_address,
      core[%coordinate, %coordinate], %destination_address, %size, noc %noc
      posted true : (i32, index, index, i32, i32, i8) -> ()
  ttkernel.noc_async_write %source_address, bank[%bank],
      %destination_address, %size, noc %noc posted true
      : (i32, i32, i32, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_set_state(
      %destination_noc_address, %size, noc %noc) posted true
      : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) posted true
      : (i32, i32, i8) -> ()
  ttkernel.noc_inline_dw_write(
      core[%coordinate, %coordinate], %destination_address, %value,
      %byte_enable, noc %noc) posted true
      : (index, index, i32, i32, i8, i8) -> ()
  ttkernel.noc_inline_dw_write(
      core[%coordinate, %coordinate], %local_semaphore, %value,
      %byte_enable, noc %noc) posted true
      : (index, index, !ttkernel.local_semaphore, i32, i8, i8) -> ()
  ttkernel.noc_inline_dw_write(
      core[%coordinate, %coordinate], %typed_address, %value,
      %byte_enable, noc %noc) posted true
      : (index, index, !ttkernel.l1_addr, i32, i8, i8) -> ()
  ttkernel.noc_async_writes_flushed(%noc) posted true : (i8) -> ()
  ttkernel.noc_async_writes_flushed(%noc) : (i8) -> ()
  func.return
}

// CPP-LABEL: void kernel_main() {
// CPP-COUNT-2: .async_write<NocOptions::POSTED>
// CPP: noc_async_write_one_packet_set_state<true>
// CPP-NEXT: noc_async_write_one_packet_with_state<true>
// CPP-COUNT-3: .inline_dw_write<NocOptions::INLINE_L1 | NocOptions::POSTED>
// CPP-NEXT: noc0.async_writes_flushed<NocOptions::POSTED>();
// CPP-NEXT: noc0.async_writes_flushed();
