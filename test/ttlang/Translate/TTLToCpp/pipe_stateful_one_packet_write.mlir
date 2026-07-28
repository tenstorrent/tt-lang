// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: Verifies TTKernel-to-C++ translation for stateful one-packet NoC
// writes used by PipeNet loop lowering.

func.func @stateful_one_packet_write() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %noc = arith.constant 0 : i8
  %dst_addr = arith.constant 4096 : i32
  %src_addr = arith.constant 8192 : i32
  %size = arith.constant 2048 : i32
  %dst_noc_addr = ttkernel.get_noc_addr(%c0, %c0, %dst_addr, %noc) : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(%dst_noc_addr, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(%src_addr, %dst_addr, noc %noc) : (i32, i32, i8) -> ()
  func.return
}

// EMITC: emitc.call_opaque "noc_async_write_one_packet_set_state"({{.*}}) : (i64, i32, i8) -> ()
// EMITC: emitc.call_opaque "noc_async_write_one_packet_with_state"({{.*}}) : (i32, i32, i8) -> ()

// CPP: noc_async_write_one_packet_set_state
// CPP: noc_async_write_one_packet_with_state
