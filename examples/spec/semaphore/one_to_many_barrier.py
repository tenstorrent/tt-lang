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
# Everything outside the markers (imports, the @ttl.operation wrapper, the no-
# op kernels) exists so the file can run standalone; it is not copied into the
# specification. The marked lines are nested inside @ttl.operation and dedented
# on render, so these mechanics add nothing to the rendered text.
#
# NOTE: the user-facing ttl.Semaphore barrier API is not implemented in the
# simulator (or the compiler) yet, so this example is expected to fail at the
# `ttl.Semaphore()` call. It is tracked by tt-lang issues #176 (simulator),
# #182 (compiler) and #177 (multi-chip). The simulator test asserts exactly
# that failure; if semaphore support lands, that test will start failing and
# this example should be promoted to a real, golden-checked test.

import ttl

# Concrete grid for a standalone run.
GRID_X, GRID_Y = 2, 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def one_to_many_barrier() -> None:
    # spec:begin
    node_num = ttl.node(dims=1)
    my_barrier = ttl.Semaphore()
    all_barrier = my_barrier.get_remote_multicast()

    @ttl.datamovement()
    def dm():
        if node_num == 0:

            # do something on node 0 while non-0 nodes wait...

            all_barrier.set(1)
        else:
            my_barrier.wait_eq(1)

            # node 0 is done

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


one_to_many_barrier()
