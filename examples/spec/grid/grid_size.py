# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, the @ttl.operation wrapper, the
# no-op kernels, and the correctness check) exists so the file can run
# standalone; it is not copied into the specification. The marked lines are
# nested inside @ttl.operation and dedented on render, so the rendered spec is
# unchanged. ttl.grid_size() resolves against the operation's grid, so the
# marked lines run per node during operation setup.

import ttl
import ttnn


@ttl.operation(grid=(8, 8))
def grid_size_example() -> None:
    # spec:begin
    # for (8, 8) single chip or SPMD grid gets x_size = 64
    x_size = ttl.grid_size(dims=1)

    # for (8, 8, 8) multi-chip grid gets x_size = 8, y_size = 64
    x_size, y_size = ttl.grid_size(dims=2)

    # for (8, 8) single-chip or SPMD grid gets x_size = 8, y_size = 8, z_size = 1
    x_size, y_size, z_size = ttl.grid_size(dims=3)
    # spec:end

    # Grid size is node-independent; on a (8, 8) grid the successive forms yield
    # 64, then (8, 8), then (8, 8, 1). (Scaffolding assertion, not rendered.)
    assert (x_size, y_size, z_size) == (8, 8, 1)

    @ttl.compute()
    def _noop_compute() -> None:
        # grid_size() is a pure query with no data movement; three no-op kernels
        # satisfy the simulator's 3-kernel (compute + 2 DM) operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm0() -> None:
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


grid_size_example()
