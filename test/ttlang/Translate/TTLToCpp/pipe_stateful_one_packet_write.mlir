// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC --implicit-check-not='emitc.call_opaque "noc_async_write"'
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP --implicit-check-not='noc_async_write('

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

// EMITC-LABEL: func.func @stateful_one_packet_write
// EMITC-DAG: %[[COORD:.*]] = "emitc.constant"() <{value = 0 : index}>
// EMITC-DAG: %[[NOC:.*]] = "emitc.constant"() <{value = 0 : i8}>
// EMITC-DAG: %[[DST:.*]] = "emitc.constant"() <{value = 4096 : i32}>
// EMITC-DAG: %[[SRC:.*]] = "emitc.constant"() <{value = 8192 : i32}>
// EMITC-DAG: %[[SIZE:.*]] = "emitc.constant"() <{value = 2048 : i32}>
// EMITC: emitc.verbatim {{.*}} args %[[COORD]], %[[COORD]], %[[DST]]
// EMITC-NEXT: %[[NOC_ADDR:.*]] = emitc.literal
// EMITC-NEXT: emitc.call_opaque "noc_async_write_one_packet_set_state"(%[[NOC_ADDR]], %[[SIZE]], %[[NOC]])
// EMITC-NEXT: emitc.call_opaque "noc_async_write_one_packet_with_state"(%[[SRC]], %[[DST]], %[[NOC]])
// EMITC-NEXT: return

// CPP-LABEL: void kernel_main() {
// CPP: size_t [[COORD:v[0-9]+]] = 0;
// CPP-NEXT: int8_t [[NOC:v[0-9]+]] = 0;
// CPP-NEXT: int32_t [[DST:v[0-9]+]] = 4096;
// CPP-NEXT: int32_t [[SRC:v[0-9]+]] = 8192;
// CPP-NEXT: int32_t [[SIZE:v[0-9]+]] = 2048;
// CPP-NEXT: uint64_t [[NOC_ADDR:noc_addr_[0-9]+]] = unicast_ep.get_noc_unicast_addr(static_cast<uint32_t>([[COORD]]), static_cast<uint32_t>([[COORD]]), static_cast<uint32_t>([[DST]]), noc0.get_noc_id());
// CPP-NEXT: noc_async_write_one_packet_set_state([[NOC_ADDR]], [[SIZE]], [[NOC]]);
// CPP-NEXT: noc_async_write_one_packet_with_state([[SRC]], [[DST]], [[NOC]]);
// CPP-NEXT: return;
