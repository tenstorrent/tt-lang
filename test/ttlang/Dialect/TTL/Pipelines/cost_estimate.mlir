// Summary: Verify the TTL pipeline carries the cost estimate and its options.
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline=cost-estimate=1 --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ON
// RUN: ttlang-opt %s '--ttl-to-ttkernel-pipeline=cost-estimate=1 cost-estimate-detail=1 cost-estimate-math-fidelity=HiFi4' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DETAIL
// RUN: not ttlang-opt %s --ttl-to-ttkernel-pipeline=cost-estimate-detail=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOENABLE

// The pass is in the pipeline either way, gated on its own `enable`, so the
// pipeline has one shape rather than two. Disabled it returns immediately.
// OFF: ttkernel-cost-estimate{detail=false enable=false
// OFF-SAME: math-fidelity=

// Enabling it through the pipeline is what _lower_program_to_kernel does for
// --ttl-cost-estimate, which is the equivalence this pipeline claims.
// ON: ttkernel-cost-estimate{detail=false enable=true

// Both extra options reach the pass: the detail view, and the math fidelity the
// IR cannot carry.
// DETAIL: ttkernel-cost-estimate{detail=true enable=true math-fidelity=HiFi4

// Asking for detail alone is refused by the pass, through the pipeline as
// directly.
// NOENABLE: error: cost estimate detail was requested with the estimate disabled

// It runs last in the TTKernel stage: after the passes that create the
// operations it reads, and before EmitC conversion turns circular-buffer calls
// into opaque verbatim strings.
// RUN: ttlang-opt %s '--ttl-to-ttkernel-pipeline=cost-estimate=1 lower-to-emitc=1' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER
// ORDER: ttkernel-insert-inits
// ORDER: ttkernel-cost-estimate
// ORDER: convert-ttkernel-to-emitc

module {}
