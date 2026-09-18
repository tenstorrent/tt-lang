// Summary: Verify invalid PipeTransport formation options fail the pass.
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=-1' \
// RUN:   --verify-diagnostics

// Purpose: Negative group sizes do not define automatic, disabled, or bounded
// grouping semantics.
// expected-error @below {{'builtin.module' op pipe transport group size must be non-negative}}
module {
}
