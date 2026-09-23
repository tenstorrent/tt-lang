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
    SRAMAddressing,
    SRAMEqualBaseGroup,
    SRAMLifetime,
    SRAMLocation,
    SRAMLocationRequirement,
    SRAMOwner,
    SRAMOwnerKind,
    SRAMOwnership,
    SRAMStorageRequirement,
)


@pytest.mark.parametrize(
    "strategy",
    [
        "first-fit-decreasing",
        "best-fit-decreasing",
        "multi-order-decreasing",
        "exact",
    ],
)
# All registered strategies enforce one base while retaining each location's
# actual extent.
def test_cpp_allocator_places_asymmetric_equal_base_regions(strategy):
    from ttl._mlir_libs._ttlang import ttl_ir

    offsets, high_water = ttl_ir.allocate_sram_location_regions(
        [(0, 256), (0, 256)],
        [10, 10],
        [0, 1],
        [64, 192],
        [None, None],
        [],
        [[0, 1]],
        [],
        64,
        strategy,
    )

    assert offsets == [0, 0]
    assert high_water == [64, 192]


# Equal-base groups cannot contain two owners at the same location.
def test_cpp_allocator_rejects_equal_base_regions_on_one_location():
    from ttl._mlir_libs._ttlang import ttl_ir

    with pytest.raises(
        ValueError,
        match="equal-offset group contains multiple regions at one location",
    ):
        ttl_ir.allocate_sram_location_regions(
            [(0, 256)],
            [10, 11],
            [0, 0],
            [64, 64],
            [None, None],
            [],
            [[0, 1]],
            [],
            64,
        )


# Persistent lifetime and multicast address equality remain independent
# constraints; neither reserves a common prefix on every location.
def test_cpp_allocator_combines_persistent_multicast_and_scratch_constraints():
    from ttl._mlir_libs._ttlang import ttl_ir

    # Regions 0-1 are per-core persistent storage, regions 2-3 are equal-base
    # multicast scratch, and region 4 is local scratch.
    offsets, high_water = ttl_ir.allocate_sram_location_regions(
        [(0, 256), (0, 256), (0, 256)],
        [0, 0, 1, 1, 2],
        [0, 1, 1, 2, 0],
        [64, 32, 48, 48, 32],
        [None, None, None, None, None],
        [(0, 4), (1, 2)],
        [[2, 3]],
        [],
        16,
        "exact",
    )

    assert offsets == [0, 48, 0, 0, 64]
    assert high_water == [96, 80, 48]


# The exact strategy minimizes the equal-sized physical reservation rather than
# the sum of location high-water marks.
def test_cpp_allocator_uses_equal_capacity_reservation_objective():
    from ttl._mlir_libs._ttlang import ttl_ir

    offsets, high_water = ttl_ir.allocate_sram_location_regions(
        [(0, 8), (0, 8)],
        [0, 1, 1, 2, 2],
        [0, 0, 1, 0, 1],
        [3, 1, 2, 3, 1],
        [None] * 5,
        [(0, 1), (0, 3), (1, 3), (2, 4)],
        [[1, 2], [3, 4]],
        [[0, 1]],
        1,
        "exact",
    )

    assert offsets == [4, 3, 3, 0, 0]
    assert high_water == [7, 5]


def _requirement(
    owner_kind,
    owner_index,
    extent_bytes,
    cores,
    lifetime,
    addressing=SRAMAddressing.UNIFORM,
):
    locations = tuple(SRAMLocation((), core) for core in cores)
    equal_base_groups = (
        (SRAMEqualBaseGroup(locations),)
        if addressing is SRAMAddressing.UNIFORM and len(locations) > 1
        else ()
    )
    return SRAMStorageRequirement(
        owner=SRAMOwner(owner_kind, owner_index),
        location_requirements=tuple(
            SRAMLocationRequirement(location, extent_bytes, payload_present=True)
            for location in locations
        ),
        alignment_bytes=16,
        equal_base_groups=equal_base_groups,
        ownership=SRAMOwnership.MOVABLE,
        lifetime=lifetime,
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
    )

    assert len(plan.pools) == 1
    placements = plan.pools[0].placements
    assert [placement.offset for placement in placements] == [0, 64, 64]
    assert plan.pools[0].reservation_bytes_per_location == 112
    assert plan.metrics.required_peak_bytes == 224
    assert plan.metrics.reservation_bytes == 224
    assert plan.metrics.fragmentation_bytes == 0
    assert plan.metrics.efficiency == 1.0


# Persistent payloads and every arena that can access the same core must occupy
# distinct bytes.
def test_persistent_declarations_conflict_with_each_other_and_scratch():
    cores = ((0, 0),)
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
            cores,
            SRAMLifetime.PERSISTENT,
        ),
    )

    plan = plan_joint_sram(
        storage,
        (_operation("operation", 48, cores),),
        budget_bytes=144,
    )

    assert [placement.offset for placement in plan.pools[0].placements] == [0, 112, 64]
    assert plan.pools[0].reservation_bytes_per_location == 144


# Equal-base and independent requirements remain separate physical pools until
# the runtime can reserve one selected address across different core subsets.
def test_transitive_core_overlap_does_not_merge_distinct_address_constraints():
    operations = (
        _operation("first", 64, ((0, 0), (1, 0))),
        _operation("second", 48, ((1, 0), (2, 0))),
        _operation("third", 32, ((2, 0),)),
    )

    plan = plan_joint_sram(
        PreparedSRAMStorage(()),
        operations,
        budget_bytes=96,
    )

    assert len(plan.pools) == 2
    assert [placement.offset for placement in plan.pools[0].placements] == [0, 0]
    assert [placement.offset for placement in plan.pools[1].placements] == [0]
    assert plan.pools[0].locations == tuple(
        SRAMLocation((), core) for core in ((0, 0), (1, 0), (2, 0))
    )
    assert plan.metrics.required_peak_bytes == 176
    assert plan.metrics.reservation_bytes == 224
    assert plan.metrics.fragmentation_bytes == 48


# A pool uses its strictest alignment and reports the resulting unused bytes.
def test_mixed_alignment_reports_padding_and_pool_fragmentation():
    cores = ((0, 0),)
    persistent = replace(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            17,
            cores,
            SRAMLifetime.PERSISTENT,
        ),
        alignment_bytes=64,
    )

    plan = plan_joint_sram(
        _storage(persistent),
        (_operation("operation", 33, cores),),
        budget_bytes=128,
    )

    assert [placement.offset for placement in plan.pools[0].placements] == [0, 64]
    assert plan.metrics.required_peak_bytes == 50
    assert plan.metrics.separate_planned_peak_bytes == 112
    assert plan.metrics.reservation_bytes == 128
    assert plan.metrics.fragmentation_bytes == 78


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


# The current retained view allocates one capacity on every participating core.
def test_joint_placement_reports_equal_capacity_pool_to_allocator():
    from ttl._mlir_libs._ttlang import ttl_ir

    observed_capacity_groups = None

    def recording_allocator(
        locations,
        owner_indices,
        location_indices,
        region_bytes,
        fixed_offsets,
        conflicts,
        equal_offset_groups,
        equal_capacity_groups,
        alignment_bytes,
        strategy,
        exact_search_limit,
    ):
        nonlocal observed_capacity_groups
        observed_capacity_groups = equal_capacity_groups
        return ttl_ir.allocate_sram_location_regions(
            locations,
            owner_indices,
            location_indices,
            region_bytes,
            fixed_offsets,
            conflicts,
            equal_offset_groups,
            equal_capacity_groups,
            alignment_bytes,
            strategy,
            exact_search_limit,
        )

    plan_joint_sram(
        PreparedSRAMStorage(()),
        (_operation("operation", 32, ((0, 0), (1, 0))),),
        budget_bytes=64,
        allocator=recording_allocator,
    )

    assert observed_capacity_groups == [[0, 1]]


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
    )
    assert plan.pools[0].reservation_bytes_per_location == 112
    with pytest.raises(
        ValueError, match="no placement offset fits every participating location"
    ):
        plan_joint_sram(
            storage,
            (operation,),
            budget_bytes=111,
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
    )

    assert plan.metrics.required_peak_bytes == 160
    assert plan.metrics.separate_planned_peak_bytes == 160
    assert plan.metrics.reservation_bytes == 160
    assert plan.metrics.fragmentation_bytes == 0
    assert plan.metrics.efficiency == 1.0


# Per-location extents avoid reserving every core for the maximum extent of
# every equal-base requirement.
def test_per_location_extents_reduce_equal_base_pool_capacity():
    first_location = SRAMLocation((), (0, 0))
    shared_location = SRAMLocation((), (1, 0))
    last_location = SRAMLocation((), (2, 0))
    first = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.PERSISTENT_DECLARATION, 0),
        location_requirements=(
            SRAMLocationRequirement(first_location, 96, True),
            SRAMLocationRequirement(shared_location, 16, True),
        ),
        alignment_bytes=16,
        equal_base_groups=(SRAMEqualBaseGroup((first_location, shared_location)),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.PERSISTENT,
    )
    second = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.PERSISTENT_DECLARATION, 1),
        location_requirements=(
            SRAMLocationRequirement(shared_location, 16, True),
            SRAMLocationRequirement(last_location, 96, True),
        ),
        alignment_bytes=16,
        equal_base_groups=(SRAMEqualBaseGroup((shared_location, last_location)),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.PERSISTENT,
    )

    plan = plan_joint_sram(_storage(first, second), (), budget_bytes=112)

    assert [placement.offset for placement in plan.pools[0].placements] == [0, 16]
    assert plan.pools[0].reservation_bytes_per_location == 112
    assert plan.metrics.required_peak_bytes == 224
    assert plan.metrics.reservation_bytes == 336


# The common model accepts partial groups, but retained views cannot realize
# them as one owned pool yet.
def test_partial_equal_base_group_requires_runtime_realization_support():
    locations = tuple(SRAMLocation((), (core, 0)) for core in range(3))
    requirement = SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.PERSISTENT_DECLARATION, 0),
        location_requirements=tuple(
            SRAMLocationRequirement(location, 16, True) for location in locations
        ),
        alignment_bytes=16,
        equal_base_groups=(SRAMEqualBaseGroup(locations[:2]),),
        ownership=SRAMOwnership.MOVABLE,
        lifetime=SRAMLifetime.PERSISTENT,
    )

    with pytest.raises(
        ValueError,
        match="requires one all-location equal-base group or independent locations",
    ):
        plan_joint_sram(_storage(requirement), (), budget_bytes=16)


# A one-location requirement has no semantic equality constraint. The runtime
# still selects the allocation mode required by the retained view.
def test_backend_selects_one_location_pool_addressing():
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            16,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        )
    )

    plan = plan_joint_sram(
        storage,
        (),
        budget_bytes=16,
        pool_addressing=lambda region: SRAMAddressing.UNIFORM,
    )

    assert plan.pools[0].addressing is SRAMAddressing.UNIFORM


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
        plan_joint_sram(storage, (object(),), budget_bytes=16)


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
        location_requirements=(
            SRAMLocationRequirement(
                SRAMLocation((), (0, 0)), 64, True, fixed_base=0x1000
            ),
        ),
    )
    operation = PreparedSRAMOperation("fixed", (fixed_tensor,), (), ())

    plan = plan_joint_sram(
        _storage(persistent),
        (operation,),
        budget_bytes=16,
    )
    assert plan.metrics.reservation_bytes == 16

    invalid_tensor = replace(
        fixed_tensor,
        ownership=SRAMOwnership.MOVABLE,
        location_requirements=(
            SRAMLocationRequirement(SRAMLocation((), (0, 0)), 64, True),
        ),
    )
    invalid_operation = PreparedSRAMOperation("invalid", (invalid_tensor,), (), ())
    with pytest.raises(ValueError, match="must remain fixed and external"):
        plan_joint_sram(
            _storage(persistent),
            (invalid_operation,),
            budget_bytes=16,
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
        location_requirements=(
            SRAMLocationRequirement(
                SRAMLocation((), (0, 0)), 16, True, fixed_base=0x1000
            ),
        ),
    )
    with pytest.raises(ValueError, match="must remain movable"):
        _storage(fixed)

    wrong_lifetime = replace(persistent, lifetime=SRAMLifetime.INVOCATION)
    with pytest.raises(ValueError, match="must remain movable"):
        _storage(wrong_lifetime)


# Invalid numeric options fail before invoking an allocator implementation.
@pytest.mark.parametrize("budget_bytes", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget_bytes):
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            16,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        )
    )

    with pytest.raises(ValueError, match="budget must be a positive integer"):
        plan_joint_sram(storage, (), budget_bytes=budget_bytes)


# The exact-search bound is part of the swappable allocator contract.
@pytest.mark.parametrize("exact_search_limit", [0, -1, True, 1.5])
def test_invalid_exact_search_limit_is_rejected(exact_search_limit):
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            16,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        )
    )

    with pytest.raises(ValueError, match="search limit must be a positive integer"):
        plan_joint_sram(
            storage,
            (),
            budget_bytes=16,
            exact_search_limit=exact_search_limit,
        )


# A strategy implementation must return one offset for every submitted region.
def test_allocator_result_must_cover_every_region():
    storage = _storage(
        _requirement(
            SRAMOwnerKind.PERSISTENT_DECLARATION,
            0,
            16,
            ((0, 0),),
            SRAMLifetime.PERSISTENT,
        )
    )

    def incomplete_allocator(*arguments):
        del arguments
        return (), ()

    with pytest.raises(RuntimeError, match="wrong placement count"):
        plan_joint_sram(
            storage,
            (),
            budget_bytes=16,
            allocator=incomplete_allocator,
        )
