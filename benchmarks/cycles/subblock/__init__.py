# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core subblock cycle benchmark (fixed block, vary subblock).

One core does one FIXED block with the data-movement threads gutted (no
DRAM/NoC), so the device cycles are purely the compute; only the forced subblock
shape changes (``--ttl-force-subblock``). Two kinds share the harness via the
``kinds.py`` registry:

  add    -> single_block_add.py     (y = a + b)
  matmul -> single_block_matmul.py  (y = a @ b, K reduced in DST)

``sweep.py`` sweeps a kind's subblocks and ranks by device cycles (and exposes
``panel`` for benchmarks/driver.py); ``emit.py`` dumps the generated kernels per
subblock. ``__main__`` runs the sweep:

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.subblock --kind add
    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.subblock --kind matmul
"""
