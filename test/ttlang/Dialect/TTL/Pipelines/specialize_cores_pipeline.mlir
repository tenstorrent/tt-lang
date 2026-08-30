// Summary: The specialize-and-annotate-dfb-use subpipeline owns the
// per-core clone, fold, local record-loop unroll, tensor-argument finalization,
// and DFB-use annotation sequence. The full pipeline also runs record-loop
// unrolling and argument finalization without specialization.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUBPIPELINE
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='specialize-cores=true' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED

// SUBPIPELINE-LABEL: Pass Manager with
// SUBPIPELINE-NEXT: builtin.module(
// SUBPIPELINE-NEXT: ttkernel-specialize-cores,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: cse,
// SUBPIPELINE-NEXT: func.func(
// SUBPIPELINE-NEXT:   ttkernel-unroll-static-pipenet-record-loops
// SUBPIPELINE-NEXT: ),
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: cse,
// SUBPIPELINE-NEXT: ttkernel-finalize-tensor-runtime-args,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: ttkernel-annotate-dfb-use
// SUBPIPELINE-NOT:  ttkernel-specialize-cores
// SUBPIPELINE-NOT:  ttkernel-annotate-dfb-use

// ENABLED: ttkernel-insert-l1-accumulation
// ENABLED: canonicalize{{.*}}
// ENABLED: cse,
// ENABLED-NEXT: ttkernel-specialize-cores,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: cse,
// ENABLED-NEXT: func.func(
// ENABLED-NEXT:   ttkernel-unroll-static-pipenet-record-loops
// ENABLED-NEXT: ),
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: cse,
// ENABLED-NEXT: ttkernel-finalize-tensor-runtime-args,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: ttkernel-annotate-dfb-use

// DISABLED: ttkernel-insert-l1-accumulation
// DISABLED-NOT: ttkernel-specialize-cores
// DISABLED-NOT: ttkernel-annotate-dfb-use
// DISABLED: func.func(
// DISABLED-NEXT: ttkernel-unroll-static-pipenet-record-loops
// DISABLED-NEXT: ),
// DISABLED-NEXT: canonicalize{{.*}},
// DISABLED-NEXT: cse
// DISABLED-NEXT: ttkernel-finalize-tensor-runtime-args
// DISABLED-NEXT: canonicalize{{.*}}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {}
