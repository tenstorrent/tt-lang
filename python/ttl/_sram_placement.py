# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Joint placement of persistent tensors and prepared operation arenas."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from ._sram_requirements import (
    PreparedSRAMOperation,
    PreparedSRAMStorage,
    SRAMAddressing,
    SRAMLifetime,
    SRAMLocation,
    SRAMOwnerKind,
    SRAMOwnership,
    SRAMStorageRequirement,
)


@dataclass(frozen=True)
class JointSRAMRegion:
    """Identify one requirement participating in joint placement."""

    operation_index: Optional[int]
    requirement_index: int
    requirement: SRAMStorageRequirement

    @property
    def is_persistent(self) -> bool:
        return self.operation_index is None


@dataclass(frozen=True)
class JointSRAMPlacement:
    """Bind one participating requirement to a pool-relative byte offset."""

    region: JointSRAMRegion
    offset: int


@dataclass(frozen=True)
class JointSRAMPool:
    """Describe one owned reservation with a common allocation mode."""

    addressing: SRAMAddressing
    locations: Tuple[SRAMLocation, ...]
    alignment_bytes: int
    reservation_bytes_per_location: int
    placements: Tuple[JointSRAMPlacement, ...]


@dataclass(frozen=True)
class JointSRAMMetrics:
    """Report capacity and placement efficiency in physical shard bytes."""

    required_peak_bytes: int
    reservation_bytes: int
    fragmentation_bytes: int
    efficiency: float
    separate_planned_peak_bytes: int


@dataclass(frozen=True)
class JointSRAMPlan:
    """Contain an all-or-nothing placement for one storage owner."""

    pools: Tuple[JointSRAMPool, ...]
    metrics: JointSRAMMetrics


LocationAllocator = Callable[
    [
        Sequence[Tuple[int, int]],
        Sequence[int],
        Sequence[int],
        Sequence[int],
        Sequence[Optional[int]],
        Sequence[Tuple[int, int]],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
        int,
        str,
        int,
    ],
    Tuple[Sequence[int], Sequence[int]],
]
PoolAddressing = Callable[[JointSRAMRegion], SRAMAddressing]


def _locations(requirement: SRAMStorageRequirement) -> frozenset[SRAMLocation]:
    return frozenset(requirement.locations)


def _default_pool_addressing(region: JointSRAMRegion) -> SRAMAddressing:
    requirement = region.requirement
    if not requirement.equal_base_groups:
        return SRAMAddressing.PER_CORE
    if (
        len(requirement.equal_base_groups) == 1
        and requirement.equal_base_groups[0].locations == requirement.locations
    ):
        return SRAMAddressing.UNIFORM
    raise ValueError(
        "current SRAM pool realization requires one all-location equal-base "
        "group or independent locations"
    )


def _validate_pool_addressing(
    requirement: SRAMStorageRequirement, addressing: SRAMAddressing
):
    if not requirement.equal_base_groups:
        return
    if (
        len(requirement.equal_base_groups) != 1
        or requirement.equal_base_groups[0].locations != requirement.locations
    ):
        raise ValueError(
            "current SRAM pool realization requires one all-location equal-base "
            "group or independent locations"
        )
    if addressing is not SRAMAddressing.UNIFORM:
        raise ValueError("equal-base storage requires a uniform owned pool")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _regions_conflict(left: JointSRAMRegion, right: JointSRAMRegion) -> bool:
    if not _locations(left.requirement).intersection(_locations(right.requirement)):
        return False
    if left.is_persistent or right.is_persistent:
        return True
    return left.operation_index == right.operation_index


def _connected_components(regions: Sequence[JointSRAMRegion]):
    pending = set(range(len(regions)))
    components = []
    while pending:
        root = min(pending)
        pending.remove(root)
        component = [root]
        worklist = [root]
        while worklist:
            current = worklist.pop()
            current_locations = _locations(regions[current].requirement)
            adjacent = [
                candidate
                for candidate in sorted(pending)
                if current_locations.intersection(
                    _locations(regions[candidate].requirement)
                )
            ]
            for candidate in adjacent:
                pending.remove(candidate)
                component.append(candidate)
                worklist.append(candidate)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _default_location_allocator(
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
    from ttl._mlir_libs._ttlang import ttl_ir

    return ttl_ir.allocate_sram_location_regions(
        list(locations),
        list(owner_indices),
        list(location_indices),
        list(region_bytes),
        list(fixed_offsets),
        [tuple(edge) for edge in conflicts],
        [list(group) for group in equal_offset_groups],
        [list(group) for group in equal_capacity_groups],
        alignment_bytes,
        strategy,
        exact_search_limit,
    )


def _peak_bytes(
    regions: Sequence[JointSRAMRegion],
    region_bytes: Callable[[JointSRAMRegion, SRAMLocation], int],
) -> int:
    locations = sorted(
        {location for region in regions for location in _locations(region.requirement)}
    )
    operation_indices = sorted(
        {
            region.operation_index
            for region in regions
            if region.operation_index is not None
        }
    )
    total = 0
    for location in locations:
        persistent_bytes = sum(
            region_bytes(region, location)
            for region in regions
            if region.is_persistent and location in _locations(region.requirement)
        )
        maximum_scratch_bytes = max(
            (
                sum(
                    region_bytes(region, location)
                    for region in regions
                    if region.operation_index == operation_index
                    and location in _locations(region.requirement)
                )
                for operation_index in operation_indices
            ),
            default=0,
        )
        total += persistent_bytes + maximum_scratch_bytes
    return total


def plan_joint_sram(
    storage: PreparedSRAMStorage,
    operations: Sequence[PreparedSRAMOperation],
    *,
    budget_bytes: int,
    strategy: str = "multi-order-decreasing",
    exact_search_limit: int = 1_000_000,
    allocator: LocationAllocator = _default_location_allocator,
    pool_addressing: PoolAddressing = _default_pool_addressing,
) -> JointSRAMPlan:
    """Place persistent declarations and serialized operation arenas.

    Operation arenas may share offsets because the storage owner orders every
    participating launch after the previous launch's device completion.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("joint SRAM budget must be a positive integer")
    if type(exact_search_limit) is not int or exact_search_limit <= 0:
        raise ValueError("exact search limit must be a positive integer")
    if not isinstance(storage, PreparedSRAMStorage):
        raise TypeError("storage must be prepared before joint placement")
    if any(
        not isinstance(operation, PreparedSRAMOperation) for operation in operations
    ):
        raise TypeError("operations must be prepared before joint placement")

    for operation in operations:
        for requirement in operation.requirements:
            if requirement.owner.kind is SRAMOwnerKind.COMPILER_ARENA:
                if (
                    requirement.ownership is not SRAMOwnership.MOVABLE
                    or requirement.lifetime is not SRAMLifetime.INVOCATION
                ):
                    raise ValueError(
                        "compiler arena must remain movable for one invocation"
                    )
                continue
            if requirement.owner.kind is SRAMOwnerKind.TENSOR_ARGUMENT:
                if (
                    requirement.ownership is not SRAMOwnership.FIXED
                    or requirement.lifetime is not SRAMLifetime.EXTERNAL
                ):
                    raise ValueError(
                        "existing tensor requirement must remain fixed and external"
                    )
                continue
            raise ValueError("prepared operation contains an unsupported SRAM owner")

    regions = [
        JointSRAMRegion(None, requirement_index, requirement)
        for requirement_index, requirement in enumerate(storage.requirements)
    ]
    for operation_index, operation in enumerate(operations):
        for requirement_index, requirement in enumerate(operation.requirements):
            if requirement.owner.kind is not SRAMOwnerKind.COMPILER_ARENA:
                continue
            regions.append(
                JointSRAMRegion(operation_index, requirement_index, requirement)
            )
    if not regions:
        raise ValueError("joint SRAM placement has no movable requirements")
    for region in regions:
        if region.requirement.ownership is not SRAMOwnership.MOVABLE:
            raise ValueError("joint SRAM placement accepts only movable requirements")
        expected_lifetime = (
            SRAMLifetime.PERSISTENT if region.is_persistent else SRAMLifetime.INVOCATION
        )
        if region.requirement.lifetime is not expected_lifetime:
            raise ValueError("joint SRAM requirement has an incompatible lifetime")

    classified_regions = []
    for region in regions:
        addressing = pool_addressing(region)
        if not isinstance(addressing, SRAMAddressing):
            raise TypeError("pool addressing must be an SRAMAddressing value")
        _validate_pool_addressing(region.requirement, addressing)
        classified_regions.append((region, addressing))

    pools = []
    for addressing in SRAMAddressing:
        mode_regions = [
            region
            for region, region_addressing in classified_regions
            if region_addressing is addressing
        ]
        for component in _connected_components(mode_regions):
            component_regions = [mode_regions[index] for index in component]
            alignment_bytes = max(
                region.requirement.alignment_bytes for region in component_regions
            )
            locations = tuple(
                sorted(
                    {
                        location
                        for region in component_regions
                        for location in _locations(region.requirement)
                    }
                )
            )
            location_indices = {
                location: index for index, location in enumerate(locations)
            }
            interval_regions = []
            interval_locations = []
            region_bytes = []
            interval_by_region_location = {}
            for region_index, region in enumerate(component_regions):
                for location in region.requirement.locations:
                    interval_index = len(region_bytes)
                    interval_regions.append(region_index)
                    interval_locations.append(location_indices[location])
                    region_bytes.append(
                        _align_up(
                            region.requirement.at(location).extent_bytes,
                            alignment_bytes,
                        )
                    )
                    interval_by_region_location[(region_index, location)] = (
                        interval_index
                    )
            conflicts = []
            for left_index, left_region in enumerate(component_regions):
                for right_index in range(left_index + 1, len(component_regions)):
                    right_region = component_regions[right_index]
                    if not _regions_conflict(left_region, right_region):
                        continue
                    for location in sorted(
                        _locations(left_region.requirement).intersection(
                            _locations(right_region.requirement)
                        )
                    ):
                        conflicts.append(
                            (
                                interval_by_region_location[(left_index, location)],
                                interval_by_region_location[(right_index, location)],
                            )
                        )
            # A retained view has one offset for all of its shards. This is a
            # backend constraint; semantic equal-base groups were validated
            # before this realization is selected.
            equal_offset_groups = [
                [
                    interval_by_region_location[(region_index, location)]
                    for location in region.requirement.locations
                ]
                for region_index, region in enumerate(component_regions)
                if len(region.requirement.locations) > 1
            ]
            interval_offsets, high_water_bytes = allocator(
                [(0, budget_bytes) for _ in locations],
                interval_regions,
                interval_locations,
                region_bytes,
                [None] * len(region_bytes),
                conflicts,
                equal_offset_groups,
                [list(range(len(locations)))] if len(locations) > 1 else [],
                alignment_bytes,
                strategy,
                exact_search_limit,
            )
            if len(interval_offsets) != len(region_bytes) or len(
                high_water_bytes
            ) != len(locations):
                raise RuntimeError("SRAM allocator returned the wrong placement count")
            offsets = []
            for region_index, region in enumerate(component_regions):
                region_offsets = {
                    int(
                        interval_offsets[
                            interval_by_region_location[(region_index, location)]
                        ]
                    )
                    for location in region.requirement.locations
                }
                if len(region_offsets) != 1:
                    raise RuntimeError(
                        "current SRAM view realization requires one owner offset"
                    )
                offsets.append(region_offsets.pop())
            arena_bytes = max((int(value) for value in high_water_bytes), default=0)
            pools.append(
                JointSRAMPool(
                    addressing=addressing,
                    locations=locations,
                    alignment_bytes=alignment_bytes,
                    reservation_bytes_per_location=arena_bytes,
                    placements=tuple(
                        JointSRAMPlacement(region, int(offset))
                        for region, offset in zip(component_regions, offsets)
                    ),
                )
            )

    reservation_bytes = sum(
        pool.reservation_bytes_per_location * len(pool.locations) for pool in pools
    )
    reservation_bytes_by_location = {}
    for pool in pools:
        for location in pool.locations:
            reservation_bytes_by_location[location] = (
                reservation_bytes_by_location.get(location, 0)
                + pool.reservation_bytes_per_location
            )
    overflowing_location = next(
        (
            (location, reserved_bytes)
            for location, reserved_bytes in sorted(
                reservation_bytes_by_location.items()
            )
            if reserved_bytes > budget_bytes
        ),
        None,
    )
    if overflowing_location is not None:
        location, reserved_bytes = overflowing_location
        raise ValueError(
            f"joint SRAM reservations require {reserved_bytes} bytes at "
            f"device {location.device}, core {location.core}, exceeding the "
            f"{budget_bytes}-byte budget"
        )
    required_peak_bytes = _peak_bytes(
        regions,
        lambda region, location: region.requirement.at(location).extent_bytes,
    )
    fragmentation_bytes = reservation_bytes - required_peak_bytes
    if fragmentation_bytes < 0:
        raise RuntimeError("joint SRAM reservation is smaller than its live data")
    metrics = JointSRAMMetrics(
        required_peak_bytes=required_peak_bytes,
        reservation_bytes=reservation_bytes,
        fragmentation_bytes=fragmentation_bytes,
        efficiency=(
            required_peak_bytes / reservation_bytes if reservation_bytes else 1.0
        ),
        separate_planned_peak_bytes=_peak_bytes(
            regions,
            lambda region, location: _align_up(
                region.requirement.at(location).extent_bytes,
                region.requirement.alignment_bytes,
            ),
        ),
    )
    return JointSRAMPlan(tuple(pools), metrics)


__all__ = [
    "JointSRAMMetrics",
    "JointSRAMPlacement",
    "JointSRAMPlan",
    "JointSRAMPool",
    "JointSRAMRegion",
    "plan_joint_sram",
]
