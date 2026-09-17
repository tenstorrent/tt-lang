# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Joint persistent and operation-arena SRAM placement tests."""

from dataclasses import replace

import pytest

from ttl._sram_placement import plan_joint_sram
from ttl._sram_requirements import (
    PreparedSRAMOperation,
    PreparedSRAMStorage,
    SRAMArenaBinding,
    SRAMAddressDomain,
    SRAMAddressing,
    SRAMLifetime,
    SRAMLocation,
    SRAMOwner,
    SRAMOwnerKind,
    SRAMOwnership,
    SRAMStorageRequirement,
)


def _allocate(
    region_bytes,
    conflicts,
    alignment_bytes,
    payload_base_offset,
    budget_bytes,
    strategy,
    exact_search_limit,
):
    del strategy, exact_search_limit
    conflict_set = {tuple(sorted(edge)) for edge in conflicts}
    offsets = []
    arena_bytes = payload_base_offset
    for region_index, size in enumerate(region_bytes):
        offset = payload_base_offset
        while True:
            end = offset + size
            overlaps = False
            for other_index, other_offset in enumerate(offsets):
                if tuple(sorted((region_index, other_index))) not in conflict_set:
                    continue
                other_end = other_offset + region_bytes[other_index]
                if end > other_offset and other_end > offset:
                    overlaps = True
                    break
            if not overlaps:
                break
            offset += alignment_bytes
        if end > budget_bytes:
            raise ValueError("placement exceeds SRAM budget")
        offsets.append(offset)
        arena_bytes = max(arena_bytes, end)
    return offsets, arena_bytes


def _requirement(
    owner_kind,
    owner_index,
    extent_bytes,
    cores,
    lifetime,
    addressing=SRAMAddressing.UNIFORM,
):
    locations = tuple(SRAMLocation((), core) for core in cores)
    domains = (
        (SRAMAddressDomain(locations),)
        if addressing is SRAMAddressing.UNIFORM
        else tuple(SRAMAddressDomain((location,)) for location in locations)
    )
    return SRAMStorageRequirement(
        owner=SRAMOwner(owner_kind, owner_index),
        extent_bytes=extent_bytes,
        alignment_bytes=16,
        address_domains=domains,
        ownership=SRAMOwnership.MOVABLE,
        lifetime=lifetime,
        addressing=addressing,
    )


def _storage(*requirements):
    return PreparedSRAMStorage(tuple(requirements))


def _operation(name, extent_bytes, cores, addressing=SRAMAddressing.UNIFORM):
    requirement = _requirement(
        SRAMOwnerKind.COMPILER_ARENA,
        0,
        extent_bytes,
        cores,
        SRAMLifetime.INVOCATION,
        addressing,
    )
    return PreparedSRAMOperation(
        name,
        (requirement,),
        (),
        (SRAMArenaBinding(0, tuple(cores)),),
    )


def test_serialized_operations_share_scratch_after_persistent_payload():
    cores = ((0, 0), (1, 0))
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            cores,
            SRAMLifetime.PERSISTENT,
        )
    )
    plan = plan_joint_sram(
        storage,
        (_operation("small", 32, cores), _operation("large", 48, cores)),
        budget_bytes=112,
        allocator=_allocate,
    )

    assert len(plan.pools) == 1
    placements = plan.pools[0].placements
    assert [placement.offset for placement in placements] == [0, 64, 64]
    assert plan.pools[0].reservation_bytes_per_location == 112
    assert plan.metrics.required_peak_bytes == 224
    assert plan.metrics.reservation_bytes == 224
    assert plan.metrics.fragmentation_bytes == 0
    assert plan.metrics.efficiency == 1.0


def test_joint_placement_uses_compiler_allocator_strategy():
    cores = ((0, 0),)
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            cores,
            SRAMLifetime.PERSISTENT,
        )
    )

    plan = plan_joint_sram(
        storage,
        (_operation("small", 32, cores), _operation("large", 48, cores)),
        budget_bytes=112,
    )

    assert [placement.offset for placement in plan.pools[0].placements] == [0, 64, 64]
    assert plan.pools[0].reservation_bytes_per_location == 112


def test_exact_data_boundary_and_independent_overflow():
    cores = ((0, 0),)
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            cores,
            SRAMLifetime.PERSISTENT,
        )
    )
    operation = _operation("operation", 48, cores)

    plan = plan_joint_sram(
        storage,
        (operation,),
        budget_bytes=112,
        allocator=_allocate,
    )
    assert plan.pools[0].reservation_bytes_per_location == 112
    with pytest.raises(ValueError, match="exceeds SRAM budget"):
        plan_joint_sram(
            storage,
            (operation,),
            budget_bytes=111,
            allocator=_allocate,
        )


def test_addressing_modes_use_separate_owned_reservations():
    cores = ((0, 0), (1, 0))
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            cores,
            SRAMLifetime.PERSISTENT,
        ),
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            1,
            32,
            (cores[0],),
            SRAMLifetime.PERSISTENT,
            SRAMAddressing.PER_CORE,
        ),
    )
    plan = plan_joint_sram(
        storage,
        (),
        budget_bytes=96,
        allocator=_allocate,
    )

    assert [pool.addressing for pool in plan.pools] == [
        SRAMAddressing.UNIFORM,
        SRAMAddressing.PER_CORE,
    ]
    assert plan.metrics.required_peak_bytes == 160
    assert plan.metrics.reservation_bytes == 160

    with pytest.raises(ValueError, match="joint SRAM reservations require 96 bytes"):
        plan_joint_sram(
            storage,
            (),
            budget_bytes=95,
            allocator=_allocate,
        )


def test_pooling_reports_overallocation_for_nonuniform_core_coverage():
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            ((0, 0), (1, 0)),
            SRAMLifetime.PERSISTENT,
        ),
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            1,
            32,
            ((1, 0),),
            SRAMLifetime.PERSISTENT,
        ),
    )
    plan = plan_joint_sram(
        storage,
        (),
        budget_bytes=96,
        allocator=_allocate,
    )

    assert plan.metrics.required_peak_bytes == 160
    assert plan.metrics.separate_planned_peak_bytes == 160
    assert plan.metrics.reservation_bytes == 192
    assert plan.metrics.fragmentation_bytes == 32
    assert plan.metrics.efficiency == pytest.approx(5 / 6)


def test_disjoint_cores_create_independent_pools():
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            64,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        ),
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            1,
            32,
            ((1, 0),),
            SRAMLifetime.PERSISTENT,
        ),
    )
    plan = plan_joint_sram(
        storage,
        (),
        budget_bytes=64,
        allocator=_allocate,
    )

    assert len(plan.pools) == 2
    assert [pool.reservation_bytes_per_location for pool in plan.pools] == [64, 32]


def test_operation_requirements_must_be_prepared():
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            16,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        )
    )
    with pytest.raises(TypeError, match="operations must be prepared"):
        plan_joint_sram(storage, (object(),), budget_bytes=16, allocator=_allocate)


def test_existing_tensor_remains_fixed_and_is_not_packed():
    persistent = _requirement(
        SRAMOwnerKind.PERSISTENT_DECLARATION,
        0,
        16,
        ((0, 0),),
        SRAMLifetime.PERSISTENT,
    )
    fixed_tensor = replace(
        _requirement(
            SRAMOwnerKind.TENSOR_ARGUMENT,
            0,
            64,
            ((0, 0),),
            SRAMLifetime.EXTERNAL,
        ),
        ownership=SRAMOwnership.FIXED,
        fixed_bases=(0x1000,),
    )
    operation = PreparedSRAMOperation("fixed", (fixed_tensor,), (), ())

    plan = plan_joint_sram(
        _storage(persistent),
        (operation,),
        budget_bytes=16,
        allocator=_allocate,
    )
    assert plan.metrics.reservation_bytes == 16

    invalid_tensor = replace(
        fixed_tensor, ownership=SRAMOwnership.MOVABLE, fixed_bases=()
    )
    invalid_operation = PreparedSRAMOperation("invalid", (invalid_tensor,), (), ())
    with pytest.raises(ValueError, match="must remain fixed and external"):
        plan_joint_sram(
            _storage(persistent),
            (invalid_operation,),
            budget_bytes=16,
            allocator=_allocate,
        )


def test_fixed_or_wrong_lifetime_requirement_is_rejected():
    persistent = _requirement(
        SRAMOwnerKind.PERSISTENT_DECLARATION,
        0,
        16,
        ((0, 0),),
        SRAMLifetime.PERSISTENT,
    )
    fixed = replace(
        persistent,
        ownership=SRAMOwnership.FIXED,
        fixed_bases=(0x1000,),
    )
    with pytest.raises(ValueError, match="must remain movable"):
        _storage(fixed)

    wrong_lifetime = replace(persistent, lifetime=SRAMLifetime.INVOCATION)
    with pytest.raises(ValueError, match="must remain movable"):
        _storage(wrong_lifetime)
