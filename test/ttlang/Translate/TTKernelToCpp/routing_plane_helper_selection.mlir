// Verify that direct-DRAM routing helpers are emitted only for their callees.
// RUN: ttlang-translate --ttkernel-to-cpp --split-input-file --output-split-marker=SPLIT-OUTPUT %s | FileCheck %s

// The write-only kernel must not include the larger scatter implementation.
// CHECK: static __attribute__((noinline)) void
// CHECK-NEXT: routing_plane_write(
// CHECK-NOT: routing_plane_scatter_write(
// CHECK: void kernel_main() {
// CHECK-NEXT: experimental::routing_plane_write();
// CHECK-NOT: routing_plane_scatter_write(
// CHECK: SPLIT-OUTPUT

module {
  func.func @write_only() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    emitc.call_opaque "experimental::routing_plane_write"() : () -> ()
    return
  }
}

// -----

// The scatter-only kernel must not include the unicast-only implementation.
// CHECK: FORCE_INLINE void routing_plane_scatter_write(
// CHECK-NOT: routing_plane_write(
// CHECK: void kernel_main() {
// CHECK-NEXT: experimental::routing_plane_scatter_write();

module {
  func.func @scatter_only() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    emitc.call_opaque "experimental::routing_plane_scatter_write"() : () -> ()
    return
  }
}
