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
# no-op kernels) exists so the file can run standalone; it is not copied into
# the specification. The marked lines are nested inside @ttl.operation and
# dedented on render, so the rendered spec is unchanged.
#
# NOTE: the user-facing ttl.Semaphore barrier API is not implemented in the
# simulator (or the compiler) yet, so this example is expected to fail at the
# `ttl.Semaphore()` call. The simulator test asserts exactly that; if semaphore
# support lands, that test will start failing and this example should be
# promoted to a real, golden-checked test.

import ttl
import ttnn

# Concrete grid for a standalone run.
GRID_X, GRID_Y = 2, 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def many_to_one_barrier() -> None:
    # The rendered spec spells the grid helper unqualified; bind it to the ttl
    # helper so the marked lines below stay verbatim.
    grid_size = ttl.grid_size
    # spec:begin
    node_num = ttl.node(dims=1)
    my_barrier = ttl.Semaphore()
    node_0_barrier = my_barrier.get_remote((0, 0))
    non_0_node_count = grid_size(dims=1) - 1

    @ttl.datamovement()
    def dm():
        if node_num != 0:

            # do something on non-0 nodes while node 0 waits...

            node_0_barrier.inc(1)
        else:
            my_barrier.wait_eq(non_0_node_count)

            # non-0 nodes are done

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


many_to_one_barrier()
