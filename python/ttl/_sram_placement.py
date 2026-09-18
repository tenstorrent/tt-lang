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


Allocator = Callable[
    [Sequence[int], Sequence[Tuple[int, int]], int, int, int, str, int],
    Tuple[Sequence[int], int],
]


def _locations(requirement: SRAMStorageRequirement) -> frozenset[SRAMLocation]:
    return frozenset(
        location
        for domain in requirement.address_domains
        for location in domain.locations
    )


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


def _default_allocator(
    region_bytes,
    conflicts,
    alignment_bytes,
    payload_base_offset,
    budget_bytes,
    strategy,
    exact_search_limit,
):
    from ttl._mlir_libs._ttlang import ttl_ir

    return ttl_ir.allocate_sram_regions(
        list(region_bytes),
        [tuple(edge) for edge in conflicts],
        alignment_bytes,
        payload_base_offset,
        budget_bytes,
        strategy,
        exact_search_limit,
    )


def _peak_bytes(
    regions: Sequence[JointSRAMRegion], region_bytes: Callable[[JointSRAMRegion], int]
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
            region_bytes(region)
            for region in regions
            if region.is_persistent and location in _locations(region.requirement)
        )
        maximum_scratch_bytes = max(
            (
                sum(
                    region_bytes(region)
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
    allocator: Allocator = _default_allocator,
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

    pools = []
    for addressing in SRAMAddressing:
        mode_regions = [
            region for region in regions if region.requirement.addressing is addressing
        ]
        for component in _connected_components(mode_regions):
            component_regions = [mode_regions[index] for index in component]
            alignment_bytes = max(
                region.requirement.alignment_bytes for region in component_regions
            )
            region_bytes = [
                _align_up(region.requirement.extent_bytes, alignment_bytes)
                for region in component_regions
            ]
            conflicts = [
                (left_index, right_index)
                for left_index in range(len(component_regions))
                for right_index in range(left_index + 1, len(component_regions))
                if _regions_conflict(
                    component_regions[left_index], component_regions[right_index]
                )
            ]
            offsets, arena_bytes = allocator(
                region_bytes,
                conflicts,
                alignment_bytes,
                0,
                budget_bytes,
                strategy,
                exact_search_limit,
            )
            if len(offsets) != len(component_regions):
                raise RuntimeError("SRAM allocator returned the wrong placement count")
            locations = tuple(
                sorted(
                    {
                        location
                        for region in component_regions
                        for location in _locations(region.requirement)
                    }
                )
            )
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
        regions, lambda region: region.requirement.extent_bytes
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
            lambda region: _align_up(
                region.requirement.extent_bytes,
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
