// Summary: The specialize-and-annotate-dfb-use subpipeline owns the
// per-core clone, fold, and DFB-use annotation sequence.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUBPIPELINE
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='specialize-cores=true' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED

// SUBPIPELINE-LABEL: Pass Manager with
// SUBPIPELINE-NEXT: builtin.module(
// SUBPIPELINE-NEXT: ttkernel-specialize-cores,
// SUBPIPELINE-NEXT: canonicalize{{.*}},
// SUBPIPELINE-NEXT: cse,
// SUBPIPELINE-NEXT: ttkernel-annotate-dfb-use
// SUBPIPELINE-NOT:  ttkernel-specialize-cores
// SUBPIPELINE-NOT:  ttkernel-annotate-dfb-use

// ENABLED: ttkernel-insert-l1-accumulation
// ENABLED: canonicalize{{.*}}
// ENABLED: cse,
// ENABLED-NEXT: ttkernel-specialize-cores,
// ENABLED-NEXT: canonicalize{{.*}},
// ENABLED-NEXT: cse,
// ENABLED-NEXT: ttkernel-annotate-dfb-use

// DISABLED: ttkernel-insert-l1-accumulation
// DISABLED-NOT: ttkernel-specialize-cores
// DISABLED-NOT: ttkernel-annotate-dfb-use

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {}
