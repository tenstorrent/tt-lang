// Compiler-owned budget validation requires the finalized arena size.
// RUN: ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget)'

// expected-error @below {{requires a validated compiler-sram arena size}}
module attributes {ttl.memory_model = "compiler-sram"} {
}
