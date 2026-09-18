// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verify that dataflow buffer types reject non-positive capacities.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// A zero dimension would make every block empty.
// expected-error @below {{shape dimensions must be positive, got 0}}
func.func @zero_dimension(%dfb: !ttl.cb<[1, 0], f32, 1>) {
  return
}

// -----

// A negative dimension cannot describe dataflow buffer storage.
// expected-error @below {{shape dimensions must be positive, got -1}}
func.func @negative_dimension(%dfb: !ttl.cb<[-1, 1], i32, 1>) {
  return
}

// -----

// A zero block count would give the dataflow buffer no capacity.
// expected-error @below {{block_count must be positive, got 0}}
func.func @zero_block_count(%dfb: !ttl.cb<[1], f32, 0>) {
  return
}

// -----

// A negative block count cannot describe dataflow buffer storage.
// expected-error @below {{block_count must be positive, got -1}}
func.func @negative_block_count(%dfb: !ttl.cb<[1], i32, -1>) {
  return
}
