# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate allocation-domain metadata before runtime resources are created."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ttl._sram_domains import (
    core_domains,
    validate_core_layouts,
    validate_receiver_targets,
)
from ttl.dataflow_buffer import PhysicalDFBConfig, SRAMCoreLayout, SRAMReceiverTarget


def make_config():
    return PhysicalDFBConfig(
        dfb_index=0,
        num_tiles=1,
        data_format="bfloat16",
        block_count=1,
        page_size=2048,
        tile=(32, 32),
        storage_index=0,
        l1_offset=0,
        l1_payload_offset=64,
        l1_allocation_bytes=2048,
        sram_core_layouts=(
            SRAMCoreLayout((0, 0), 64, True, 2112, 0),
            SRAMCoreLayout((1, 0), 0, False, 64, 1),
        ),
    )


def test_independent_core_layouts_and_inactive_payload():
    config = make_config()
    assert validate_core_layouts([config], [(0, 0), (1, 0)]) == {
        (0, 0): 2112,
        (1, 0): 64,
    }
    assert core_domains([config]) == [((0, 0),), ((1, 0),)]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"node": (0, 0)}, "exactly once"),
        ({"arena_bytes": 0}, "control records"),
        (
            {"payload_offset": 4, "payload_present": True, "arena_bytes": 2112},
            "outside",
        ),
        ({"payload_offset": 64}, "zero offset"),
        ({"domain": 0}, "share one layout"),
        ({"domain": -1}, "nonnegative"),
    ],
)
def test_invalid_core_layout(change, message):
    config = make_config()
    invalid = replace(
        config,
        sram_core_layouts=(
            config.sram_core_layouts[0],
            replace(config.sram_core_layouts[1], **change),
        ),
    )
    with pytest.raises(ValueError, match=message):
        validate_core_layouts([invalid], [(0, 0), (1, 0)])


def test_multicast_domain_requires_equal_layouts():
    config = make_config()
    config = replace(
        config,
        sram_core_layouts=(
            config.sram_core_layouts[0],
            replace(config.sram_core_layouts[0], node=(1, 0)),
        ),
    )
    assert validate_core_layouts([config], [(0, 0), (1, 0)]) == {
        (0, 0): 2112,
        (1, 0): 2112,
    }
    assert core_domains([config]) == [((0, 0), (1, 0))]


def test_domain_membership_must_agree_between_dfbs():
    config = make_config()
    other = replace(
        config,
        dfb_index=1,
        storage_index=1,
        sram_core_layouts=(
            replace(config.sram_core_layouts[0], domain=2),
            config.sram_core_layouts[1],
        ),
    )
    with pytest.raises(ValueError, match="inconsistent SRAM allocation domains"):
        validate_core_layouts([config, other], [(0, 0), (1, 0)])


def test_storage_aliases_must_agree_on_placement():
    config = make_config()
    other = replace(
        config,
        dfb_index=1,
        sram_core_layouts=(
            replace(config.sram_core_layouts[0], payload_offset=32),
            config.sram_core_layouts[1],
        ),
    )
    with pytest.raises(ValueError, match="share its core layouts"):
        validate_core_layouts([config, other], [(0, 0), (1, 0)])


@pytest.mark.parametrize(
    "targets,indices,message",
    [
        ([], [0], "complete destination identities"),
        ([SRAMReceiverTarget(1, (0, 0))], [0], "invalid DFB"),
        ([SRAMReceiverTarget(0, (1, 0))], [0], "no payload"),
        ([SRAMReceiverTarget(0, (2, 0))], [0], "no payload"),
    ],
)
def test_receiver_target_validation(targets, indices, message):
    spec = SimpleNamespace(
        sram_receiver_targets=targets, pipe_computed_address_dfb_indices=indices
    )
    with pytest.raises(ValueError, match=message):
        validate_receiver_targets([spec], [make_config()])


def test_receiver_target_retains_device_identity():
    spec = SimpleNamespace(
        sram_receiver_targets=[SRAMReceiverTarget(0, (0, 0), (0, 1))],
        pipe_computed_address_dfb_indices=[0],
    )
    validate_receiver_targets([spec], [make_config()])


def test_independent_address_requires_device_coordinate():
    from ttl._sram_domains import tensor_base

    tensor = SimpleNamespace(is_per_core_allocated=lambda: True)
    with pytest.raises(ValueError, match="requires a logical device coordinate"):
        tensor_base(object(), tensor, (0, 0), None)
