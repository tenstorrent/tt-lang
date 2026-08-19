// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline 2>&1 | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet)' --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefix=SUBPIPELINE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-to-ttkernel-pipeline{matmul-full-fp32=false})' --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefix=MATMUL-DISABLED
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-to-ttkernel-pipeline{matmul-full-fp32=true})' --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefix=MATMUL-ENABLED

// Verify tensor recurrence lowering runs before DFB materialization and
// synchronization, and PipeNet verification runs before DFB index reuse.

// CHECK-LABEL: Pass Manager with
// CHECK:      ttl-form-accumulation-scopes
// CHECK:      ttl-lower-accumulation-scopes
// CHECK:      ttl-materialize-loop-state
// CHECK:      ttl-insert-copy-wait
// CHECK:      ttl-create-producer-compute
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
// CHECK-NEXT: ttl-form-pipe-transports{{.*}},
// CHECK-NEXT: func.func(
// CHECK-NEXT:   ttl-coalesce-dfb-acquires
// CHECK-NEXT: ),
// CHECK-NEXT: ttl-finalize-dfb-indices{exact-coloring-search-limit=1000000 reuse-user-dfbs=true},
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

// Verify the public pipeline option reaches kernel configuration resolution.

// MATMUL-DISABLED: ttl-set-compute-kernel-config{{.*}}matmul-full-fp32=false
// MATMUL-ENABLED: ttl-set-compute-kernel-config{{.*}}matmul-full-fp32=true

module {}
