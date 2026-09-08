// Summary: Full lowering, standalone specialization, and the cleanup pipeline
// used by Python share receive batching, record expansion, write-state cleanup,
// and runtime-argument finalization in the same order.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-cleanup-and-finalize-runtime-args)' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=CLEANUP
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUBPIPELINE
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='specialize-cores=true' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not=ttkernel-specialize-cores --implicit-check-not=ttkernel-annotate-dfb-use

// SUBPIPELINE-LABEL: Pass Manager with
// SUBPIPELINE-NEXT: builtin.module(
// SUBPIPELINE-NEXT: ttkernel-specialize-cores,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: cse,
// SUBPIPELINE-NEXT: func.func(
// SUBPIPELINE-NEXT:   ttkernel-batch-static-pipenet-receives,
// SUBPIPELINE-NEXT:   ttkernel-unroll-static-pipenet-record-loops
// SUBPIPELINE-NEXT: ),
// SUBPIPELINE-NEXT: lower-affine,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: cse,
// SUBPIPELINE-NEXT: ttkernel-cleanup,
// SUBPIPELINE-NEXT: ttkernel-finalize-tensor-runtime-args,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: ttkernel-annotate-dfb-use
// SUBPIPELINE-NOT:  ttkernel-specialize-cores
// SUBPIPELINE-NOT:  ttkernel-annotate-dfb-use

// CLEANUP-LABEL: Pass Manager with
// CLEANUP-NEXT: builtin.module(
// CLEANUP-NEXT: func.func(
// CLEANUP-NEXT: ttkernel-batch-static-pipenet-receives,
// CLEANUP-NEXT: ttkernel-unroll-static-pipenet-record-loops
// CLEANUP-NEXT: ),
// CLEANUP-NEXT: lower-affine,
// CLEANUP-NEXT: canonicalize{{.*}},
// CLEANUP-NEXT: cse,
// CLEANUP-NEXT: ttkernel-cleanup,
// CLEANUP-NEXT: ttkernel-finalize-tensor-runtime-args,
// CLEANUP-NEXT: canonicalize{{.*}}

// ENABLED: ttkernel-insert-l1-accumulation
// ENABLED: canonicalize{{.*}}
// ENABLED: cse,
// ENABLED-NEXT: ttkernel-specialize-cores,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: cse,
// ENABLED-NEXT: func.func(
// ENABLED-NEXT:   ttkernel-batch-static-pipenet-receives,
// ENABLED-NEXT:   ttkernel-unroll-static-pipenet-record-loops
// ENABLED-NEXT: ),
// ENABLED-NEXT: lower-affine,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: cse,
// ENABLED-NEXT: ttkernel-cleanup,
// ENABLED-NEXT: ttkernel-finalize-tensor-runtime-args,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: ttkernel-annotate-dfb-use

// DISABLED: ttkernel-insert-l1-accumulation
// DISABLED: func.func(
// DISABLED-NEXT: ttkernel-combine-pack-tiles
// DISABLED-NEXT: ),
// DISABLED-NEXT: canonicalize{{.*}},
// DISABLED-NEXT: cse,
// DISABLED-NEXT: func.func(
// DISABLED-NEXT: ttkernel-batch-static-pipenet-receives,
// DISABLED-NEXT: ttkernel-unroll-static-pipenet-record-loops
// DISABLED-NEXT: ),
// DISABLED-NEXT: lower-affine,
// DISABLED-NEXT: canonicalize{{.*}},
// DISABLED-NEXT: cse
// DISABLED-NEXT: ttkernel-cleanup
// DISABLED-NEXT: ttkernel-finalize-tensor-runtime-args
// DISABLED-NEXT: canonicalize{{.*}}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {}
