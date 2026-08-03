// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline 2>&1 | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet)' --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefix=SUBPIPELINE

// Verify PipeNet schedule semantics before later transformations modify the
// high-level pipe and DFB operations or reuse provisional DFB indices.

// CHECK-LABEL: Pass Manager with
// CHECK:        ttl-create-producer-compute
// CHECK-NEXT: ),
// CHECK-NEXT: func.func(
// CHECK-NEXT:   ttl-insert-intermediate-dfbs{enable=true}
// CHECK-NEXT: ),
// CHECK-NEXT: func.func(
// CHECK-NEXT:   convert-ttl-to-compute
// CHECK-NEXT: ),
// CHECK-NEXT: func.func(
// CHECK-NEXT:   ttl-insert-cb-sync
// CHECK-NEXT: ),
// CHECK-NEXT: ttl-verify-pipenet-guards,
// CHECK-NEXT: ttl-verify-pipenet-schedule,
// CHECK-NEXT: func.func(
// CHECK-NEXT:   ttl-coalesce-dfb-acquires
// CHECK-NEXT: ),
// CHECK-NEXT: ttl-finalize-dfb-indices,
// CHECK-NEXT: func.func(
// CHECK-NOT:    ttl-verify-pipenet-guards
// CHECK-NOT:    ttl-verify-pipenet-schedule
// CHECK:        ttl-annotate-cb-associations
// CHECK-NEXT: ),
// CHECK-NOT:  ttl-verify-pipenet-guards
// CHECK-NOT:  ttl-verify-pipenet-schedule
// CHECK:      convert-ttl-to-ttkernel

// Verify the registered subpipeline preserves the required verifier order.

// SUBPIPELINE-LABEL: Pass Manager with
// SUBPIPELINE-NEXT: builtin.module(
// SUBPIPELINE-NEXT: ttl-verify-pipenet-guards,
// SUBPIPELINE-NEXT: ttl-verify-pipenet-schedule
// SUBPIPELINE-NOT:  ttl-verify-pipenet-guards
// SUBPIPELINE-NOT:  ttl-verify-pipenet-schedule

module {}
