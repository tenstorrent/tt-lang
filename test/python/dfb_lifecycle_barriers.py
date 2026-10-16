# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s crossing-install 2>&1 | FileCheck %s --check-prefix=CROSSING-INSTALL
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s filled 2>&1 | FileCheck %s --check-prefix=FILLED
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s overfilled 2>&1 | FileCheck %s --check-prefix=OVERFILLED
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s overfilled-keep-state 2>&1 | FileCheck %s --check-prefix=OVERFILLED
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s wait-first 2>&1 | FileCheck %s --check-prefix=WAIT-FIRST
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s single-publish 2>&1 | FileCheck %s --check-prefix=SINGLE-PUBLISH
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s repeated-publish 2>&1 | FileCheck %s --check-prefix=REPEATED-PUBLISH
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s unrestored-publish 2>&1 | FileCheck %s --check-prefix=UNRESTORED-PUBLISH
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s generations 2>&1 | FileCheck %s --check-prefix=GENERATIONS
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s growing-generations 2>&1 | FileCheck %s --check-prefix=GROWING-GENERATIONS

# CROSSING-INSTALL: error: logical DFB {{[0-9]+}} has capacity-unsafe producer and consumer transactions on core_x=1, core_y=0
# CROSSING-INSTALL: note: the consumer pops 1 block(s) per launch, but the producer pushes 0 block(s) per launch
# CROSSING-INSTALL: note: in the interval that starts at this synchronized reset or reconfiguration, which restores the DFB
# FILLED-NOT: {{error|warning}}:
# FILLED: COMPILED
# OVERFILLED: error: logical DFB {{[0-9]+}} has transactions that cannot complete before a synchronized reset or reconfiguration on core_x=0, core_y=0
# OVERFILLED: note: before it, the producer holds 3 reserved block(s) that are not popped, exceeding capacity 2
# OVERFILLED: note: every kernel on the node waits here until all of them arrive
# WAIT-FIRST: error: logical DFB {{[0-9]+}} has transactions that cannot complete before a synchronized reset or reconfiguration on core_x=0, core_y=0
# WAIT-FIRST: note: before it, the consumer waits for 1 block(s) but the producer pushes 0
# SINGLE-PUBLISH-NOT: {{error|warning}}:
# SINGLE-PUBLISH: COMPILED
# REPEATED-PUBLISH: warning: logical DFB {{[0-9]+}} is never popped on core_x=0, core_y=0, but its producer can push 4 block(s) into capacity 1 before a synchronized reset or reconfiguration restores it
# REPEATED-PUBLISH: note: published blocks stay in the DFB until a pop or a reset or reconfiguration that restores it
# REPEATED-PUBLISH: note: this external call may perform protocol actions on the DFB that it does not declare
# REPEATED-PUBLISH-NOT: a reconfiguration restores a DFB
# REPEATED-PUBLISH: COMPILED
# UNRESTORED-PUBLISH: warning: logical DFB {{[0-9]+}} is never popped on core_x=0, core_y=0, but its producer can push 4 block(s) into capacity 1 during the launch
# UNRESTORED-PUBLISH: note: published blocks stay in the DFB until a pop, so the producer blocks once the DFB is full
# UNRESTORED-PUBLISH: note: this external call may perform protocol actions on the DFB that it does not declare
# UNRESTORED-PUBLISH-NOT: a reconfiguration restores a DFB
# UNRESTORED-PUBLISH: COMPILED
# GENERATIONS-NOT: {{error|warning}}:
# GENERATIONS: COMPILED
# GROWING-GENERATIONS: error: logical DFB {{[0-9]+}} has transactions that cannot complete before a synchronized reset or reconfiguration on core_x=0, core_y=0
# GROWING-GENERATIONS: note: before it, the producer holds 3 reserved block(s) that are not popped, exceeding capacity 2

"""Verify DFB lifecycles across synchronized resets and reconfigurations.

Every participant kernel waits at a reset or reconfiguration until all of them
arrive, so the transactions before it must complete among themselves. A
reconfiguration restores a DFB only on the nodes where the finalized plan
installs its descriptor; elsewhere the DFB keeps its state across it.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api

MODE = sys.argv[1]
INCLUDE = os.path.join(os.path.dirname(__file__), "include")
RECONFIGURATION_HEADER = os.path.join(INCLUDE, "dfb_reconfiguration_test_helpers.hpp")
LIVENESS_HEADER = os.path.join(INCLUDE, "dfb_liveness_test_helpers.hpp")


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def _participants():
    return (
        ttl.Kernel(ttl.KernelKind.COMPUTE),
        ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
        ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
    )


def make_crossing_install():
    # Node 0 completes a lifecycle on each side of the boundary. Node 1 pushes a
    # block before it and pops it after it, but the plan installs the next
    # descriptor on both nodes, so the reconfiguration restores the DFB on
    # node 1 and discards that block.
    compute_kernel, reader_kernel, writer_kernel = _participants()
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(2, 1))
    def crossing_install(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            with dfb.reserve() as blk:
                ttl.copy(inp[0, 0], blk).wait()
            ttl.reconfigure_dfbs(boundary)
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, 1], blk).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with dfb.wait() as blk:
                    ttl.copy(blk, out[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 1]).wait()

    return crossing_install


def make_fill(pushes, discard_dfb_state):
    # The reader pushes `pushes` blocks before the boundary and the writer pops
    # them after it.
    compute_kernel, reader_kernel, writer_kernel = _participants()
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=discard_dfb_state,
    )

    @ttl.operation(grid=(1, 1))
    def fill(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            for index in range(pushes):
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, index], blk).wait()
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            ttl.reconfigure_dfbs(boundary)
            for index in range(pushes):
                with dfb.wait() as blk:
                    ttl.copy(blk, out[0, index]).wait()

    return fill


def make_wait_first():
    # The writer waits before the boundary for a block the reader pushes
    # after it.
    compute_kernel, reader_kernel, writer_kernel = _participants()
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(1, 1))
    def wait_first(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            ttl.reconfigure_dfbs(boundary)
            with dfb.reserve() as blk:
                ttl.copy(inp[0, 0], blk).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)

    return wait_first


def make_waited_publications(publications):
    # The reader publishes `publications` blocks and compute waits for them
    # without popping; an external call without an effect contract excludes
    # the node from exact verification. One publication fits the capacity; four
    # block the reader before the reset restores the DFB.
    compute_kernel, reader_kernel, writer_kernel = _participants()
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation(grid=(1, 1))
    def waited_publications(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for _ in range(4):
                ttl.call_extern_func(
                    RECONFIGURATION_HEADER,
                    "wait_without_pop",
                    template_args=[ttl.dfb_descriptor(dfb)],
                    dfb_effects=[ttl.DFBEffect.wait(dfb, tiles=1)],
                )
            ttl.call_extern_func(
                LIVENESS_HEADER, "retain_dfb_liveness", dfb_dependencies=[dfb]
            )
            ttl.reset_dfbs(reset, dfbs=[dfb])

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            for index in range(publications):
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, index], blk).wait()
            ttl.reset_dfbs(reset, dfbs=[dfb])

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            ttl.reset_dfbs(reset, dfbs=[dfb])

    return waited_publications


def make_unrestored_publications():
    # The waited publications without a reset: nothing restores the DFB, so
    # four publications block the reader during the launch.
    compute_kernel, reader_kernel, writer_kernel = _participants()

    @ttl.operation(grid=(1, 1))
    def unrestored_publications(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for _ in range(4):
                ttl.call_extern_func(
                    RECONFIGURATION_HEADER,
                    "wait_without_pop",
                    template_args=[ttl.dfb_descriptor(dfb)],
                    dfb_effects=[ttl.DFBEffect.wait(dfb, tiles=1)],
                )
            ttl.call_extern_func(
                LIVENESS_HEADER, "retain_dfb_liveness", dfb_dependencies=[dfb]
            )

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            for index in range(4):
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, index], blk).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            pass

    return unrestored_publications


def make_generations(pushes_per_generation):
    # A generation loop with one reset per generation. The reader pushes into
    # `handoff` before the reset and the writer pops after it, so the block
    # crosses the reset, which restores only `scratch`; generation 0 also
    # passes one extra block. With two pushes per generation, the writer pops
    # the second block of every generation after the loop, and `handoff` holds
    # one more block at each reset than at the one before.
    compute_kernel, reader_kernel, writer_kernel = _participants()
    reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))

    @ttl.operation(grid=(1, 1))
    def generations(inp, out):
        handoff = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            for generation in range(64):
                ttl.reset_dfbs(reset, dfbs=[scratch])

        @ttl.datamovement(kernel=reader_kernel)
        def reader():
            for generation in range(64):
                if generation == 0:
                    with handoff.reserve() as blk:
                        ttl.copy(inp[0, 0], blk).wait()
                for index in range(pushes_per_generation):
                    with handoff.reserve() as blk:
                        ttl.copy(inp[0, index], blk).wait()
                ttl.reset_dfbs(reset, dfbs=[scratch])

        @ttl.datamovement(kernel=writer_kernel)
        def writer():
            for generation in range(64):
                if generation == 0:
                    with handoff.wait() as blk:
                        ttl.copy(blk, out[0, 0]).wait()
                ttl.reset_dfbs(reset, dfbs=[scratch])
                with handoff.wait() as blk:
                    ttl.copy(blk, out[0, 1]).wait()
            for generation in range(64 * (pushes_per_generation - 1)):
                with handoff.wait() as blk:
                    ttl.copy(blk, out[0, 2]).wait()

    return generations


FACTORIES = {
    "crossing-install": make_crossing_install,
    "filled": lambda: make_fill(pushes=2, discard_dfb_state=True),
    "overfilled": lambda: make_fill(pushes=3, discard_dfb_state=True),
    "overfilled-keep-state": lambda: make_fill(pushes=3, discard_dfb_state=False),
    "wait-first": make_wait_first,
    "single-publish": lambda: make_waited_publications(publications=1),
    "repeated-publish": lambda: make_waited_publications(publications=4),
    "unrestored-publish": make_unrestored_publications,
    "generations": lambda: make_generations(pushes_per_generation=1),
    "growing-generations": lambda: make_generations(pushes_per_generation=2),
}
operation = FACTORIES[MODE]()

tensors = [
    ttnn.from_torch(
        torch.zeros((32, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
operation(*tensors)
print("COMPILED")
