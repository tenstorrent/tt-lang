# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for simulator selection among completed receives."""

from collections.abc import Iterable

import pytest

from test_utils import make_ones_tile

from sim import ttl
from sim.analysis import analyze_kernel_function, install_copy_wait_hooks
from sim.context import set_current_kernel_type
from sim.copy import ReceiveRequest, copy, wait_any
from sim.dfb import Block, DataflowBuffer
from sim.kernel import KernelKind
from sim.pipe import Pipe


@pytest.fixture(autouse=True)
def setup_scheduler_context(dm_kernel_context):
    """Run receive-selection operations in a data-movement kernel context."""
    pass


def make_receive_endpoints(count: int) -> tuple[tuple[Pipe, ...], tuple[Block, ...]]:
    """Create distinct PipeNet endpoints and reserved landing blocks."""
    set_current_kernel_type(KernelKind.DATA_MOVEMENT)
    pipes = tuple(Pipe(0, pipe_index + 1) for pipe_index in range(count))
    landing_dfbs = tuple(
        DataflowBuffer(likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2)
        for _ in range(count)
    )
    landing_blocks = tuple(landing.reserve() for landing in landing_dfbs)
    return pipes, landing_blocks


def publish_ready_messages(
    pipes: tuple[Pipe, ...], ready_indices: Iterable[int]
) -> None:
    """Send one message to each selected receive endpoint."""
    set_current_kernel_type(KernelKind.DATA_MOVEMENT)

    source_dfb = DataflowBuffer(
        likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
    )
    with source_dfb.reserve() as source:
        copy(make_ones_tile(), source).wait()
    with source_dfb.wait() as source:
        for pipe_index in ready_indices:
            copy(source, pipes[pipe_index]).wait()


def make_ready_requests(
    count: int, ready_indices: tuple[int, ...] | None = None
) -> tuple[tuple[ReceiveRequest, ...], tuple[Block, ...]]:
    """Create distinct receives and send one message to selected pipes."""
    pipes, landing_blocks = make_receive_endpoints(count)
    requests = tuple(
        copy(pipe, landing) for pipe, landing in zip(pipes, landing_blocks, strict=True)
    )
    publish_ready_messages(
        pipes, range(count) if ready_indices is None else ready_indices
    )

    return requests, landing_blocks


def complete_requests(
    requests: tuple[ReceiveRequest, ...], landing_blocks: tuple[Block, ...]
) -> None:
    for request, landing in zip(requests, landing_blocks, strict=True):
        request.wait()
        landing.push()


def test_wait_any_rotates_from_start() -> None:
    """An all-ready tuple selects the first entry at or after start."""
    requests, landing_blocks = make_ready_requests(4)

    ready = wait_any(requests, start=3)

    assert ready.index() == 3
    assert requests[3].is_completed
    assert all(not request.is_completed for request in requests[:3])
    complete_requests(requests, landing_blocks)


@pytest.mark.parametrize("start, expected", [(5, 1), (-1, 3)])
def test_wait_any_wraps_rotating_order(start: int, expected: int) -> None:
    """A start value is normalized modulo the request count."""
    requests, landing_blocks = make_ready_requests(4)

    ready = wait_any(requests, start=start)

    assert ready.index() == expected
    complete_requests(requests, landing_blocks)


def test_wait_any_skips_pending_requests() -> None:
    """Selection completes the ready request and leaves others pending."""
    requests, landing_blocks = make_ready_requests(4, ready_indices=(1,))

    ready = wait_any(requests, start=3)

    assert ready.index() == 1
    assert requests[1].is_completed
    assert all(
        not request.is_completed
        for request_index, request in enumerate(requests)
        if request_index != 1
    )
    landing_blocks[1].push()


def test_wait_any_scans_in_ascending_cyclic_order() -> None:
    """The scan visits candidates in ascending cyclic order."""
    requests, landing_blocks = make_ready_requests(4, ready_indices=(1, 3))

    ready = wait_any(requests, start=0)

    assert ready.index() == 1
    assert requests[1].is_completed
    assert not requests[3].is_completed
    landing_blocks[1].push()
    requests[3].wait()
    landing_blocks[3].push()


def test_wait_any_selects_the_first_ready_candidate() -> None:
    """Selection stops after the first completed request in scan order."""
    requests, landing_blocks = make_ready_requests(4, ready_indices=(2, 3))

    ready = wait_any(requests, start=2)

    assert ready.index() == 2
    assert not requests[3].is_completed
    landing_blocks[2].push()
    requests[3].wait()
    landing_blocks[3].push()


def test_deferred_waits_register_through_public_copy() -> None:
    """The public copy entry point registers each deferred request object."""
    pipes, landing_blocks = make_receive_endpoints(2)
    publish_ready_messages(pipes, range(2))

    def dm():
        pending = ttl.copy(pipes[0], landing_blocks[0])
        selected = ttl.copy(pipes[1], landing_blocks[1])
        ttl.wait_any((pending, selected), start=1)
        return pending, selected

    analysis = analyze_kernel_function(dm)
    install_copy_wait_hooks({dm.__code__: analysis})
    requests = dm()

    assert all(request.is_completed for request in requests)
    for landing_block in landing_blocks:
        landing_block.push()


def test_wait_any_rejects_invalid_request_sets() -> None:
    """The public API requires a nonempty tuple of distinct receive requests."""
    requests, landing_blocks = make_ready_requests(1)
    request = requests[0]

    with pytest.raises(TypeError, match="requires a tuple"):
        wait_any([request])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        wait_any(())
    with pytest.raises(TypeError, match="only PipeNet receive requests"):
        wait_any((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct receive requests"):
        wait_any((request, request))
    with pytest.raises(TypeError, match="start must be an integer"):
        wait_any((request,), start="0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="start must be an integer"):
        wait_any((request,), start=True)

    assert not request.is_completed
    complete_requests(requests, landing_blocks)
