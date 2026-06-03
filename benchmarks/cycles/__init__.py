# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core Tracy cycle-count benchmarks.

Each benchmark runs one kernel once with the device profiler on and reports its
device kernel duration. They pair a tt-lang op against a low-level metal
primitive doing the same single-core work (e.g. the flash shard vs
``compute_sdpa_chunk``), to see how the tt-lang codegen compares cycle-for-cycle.
Run with ``TT_METAL_DEVICE_PROFILER=1``.
"""
