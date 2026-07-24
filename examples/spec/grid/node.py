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
# unchanged. ttl.node() resolves against the current node, so the marked lines
# run per node during operation setup.

import ttl
import ttnn


@ttl.operation(grid=(8, 8))
def node_example() -> None:
    # spec:begin
    # for (8, 8) single chip or SPMD grid gets x = [0, 64)
    x = ttl.node(dims=1)

    # for (8, 8, 8) multi-chip grid gets x = [0, 8), y = [0, 64)
    x, y = ttl.node(dims=2)

    # for (8, 8) single-chip or SPMD grid gets x = [0, 8), y = [0, 8), z = 0
    x, y, z = ttl.node(dims=3)
    # spec:end

    # On a (8, 8) grid the dims=3 form yields the current node's coordinates:
    # x in [0, 8), y in [0, 8), z == 0. (Scaffolding assertion, not rendered.)
    assert 0 <= x < 8 and 0 <= y < 8 and z == 0

    @ttl.compute()
    def _noop_compute() -> None:
        # node() is a pure query with no data movement; three no-op kernels
        # satisfy the simulator's 3-kernel (compute + 2 DM) operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm0() -> None:
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


node_example()
