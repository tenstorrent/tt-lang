// Compiler-owned lowering requires the finalized arena size.
// RUN: ttlang-opt %s --verify-diagnostics --convert-ttl-to-ttkernel

// expected-error @below {{missing validated compiler-l1 arena size}}
module attributes {ttl.memory_model = "compiler-l1"} {
}
