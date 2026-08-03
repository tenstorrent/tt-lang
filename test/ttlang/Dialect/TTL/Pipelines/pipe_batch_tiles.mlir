// Summary: Verify the TTL pipeline propagates its PipeTransport group bound.
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='pipe-batch-tiles=4' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=BOUND
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='l1-budget-override=98304' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=BUDGET

// The default leaves group selection to ttl-form-pipe-transports.
// AUTO: ttl-form-pipe-transports{group-size=0 l1-budget-override=0}
// AUTO: ttl-validate-cb-budget{l1-budget-override=0}

// A pipeline bound is forwarded to ttl-form-pipe-transports.
// BOUND: ttl-form-pipe-transports{group-size=4 l1-budget-override=0}

// The L1 override applies to both transport selection and final DFB
// validation.
// BUDGET: ttl-form-pipe-transports{group-size=0 l1-budget-override=98304}
// BUDGET: ttl-validate-cb-budget{l1-budget-override=98304}

module {}
