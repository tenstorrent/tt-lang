# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock timing for device benchmarks.

The runs are enqueued back-to-back with no sleeps and the device is synced
*once* after the loop, then averaged per iteration. Syncing per run measures
dispatch + execute latency for a single call (with a host round-trip bubble
between every run); enqueueing all of them and syncing once measures
steady-state throughput, which is what we want.
"""

import time

import ttnn


def time_runs(thunk, cleanup, device, *, warmup=3, runs=5):
    """Warmup, then time ``runs`` back-to-back invocations of ``thunk``.

    ``thunk()`` returns a value passed to ``cleanup`` (e.g. to deallocate an
    output). ``cleanup`` stays inside the loop, but since ttnn enqueues work in
    order it only *enqueues* the dealloc -- it neither blocks nor holds ``runs``
    live buffers. Returns the mean wall seconds per run.
    """
    for _ in range(warmup):
        cleanup(thunk())
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(runs):
        cleanup(thunk())
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / runs
