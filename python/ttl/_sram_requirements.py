# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable storage requirements shared by SRAM runtime allocators."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Sequence, Tuple

_SRAM_ADDRESS_LIMIT = 1 << 32
_DFB_CONTROL_BYTES = 8


class SRAMOwnership(Enum):
    """State who selects an owner's physical placement.

    FIXED preserves the supplied base at every location. MOVABLE permits the
    allocator to select an offset subject to conflicts and address constraints.
    """

    FIXED = auto()
    MOVABLE = auto()


class SRAMAddressing(Enum):
    """Select the retained pool's physical allocation mode.

    UNIFORM uses one physical base across participating locations. PER_CORE
    permits independent bases. Equal-base groups carry semantic address
    constraints in the common requirement model.
    """

    UNIFORM = auto()
    PER_CORE = auto()


class SRAMLifetime(Enum):
    """State when an owner's storage may be released or reused.

    INVOCATION lasts through one completed operation launch. EXTERNAL follows
    the caller-owned tensor lifetime. PERSISTENT lasts until SRAMStorage closes.
    """

    INVOCATION = auto()
    EXTERNAL = auto()
    PERSISTENT = auto()


class SRAMOwnerKind(Enum):
    """Identify the subsystem that owns one physical storage requirement."""

    COMPILER_ARENA = auto()
    TENSOR_ARGUMENT = auto()
    PERSISTENT_DECLARATION = auto()


class SRAMUseKind(Enum):
    """Identify a logical use of a physical storage requirement."""

    DFB_CONTROL = auto()
    DFB_PAYLOAD = auto()


@dataclass(frozen=True, order=True)
class SRAMLocation:
    """Identify one logical device and worker core.

    An empty device coordinate denotes the device domain selected at binding.
    """

    device: Tuple[int, ...]
    core: Tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.device, tuple) or not isinstance(self.core, tuple):
            raise TypeError("SRAM location coordinates must be tuples")
        if any(
            type(coordinate) is not int or coordinate < 0 for coordinate in self.device
        ):
            raise ValueError("SRAM device coordinates must be nonnegative integers")
        if len(self.core) != 2 or any(
            type(coordinate) is not int or coordinate < 0 for coordinate in self.core
        ):
            raise ValueError(
                "SRAM core coordinates must contain two nonnegative integers"
            )


@dataclass(frozen=True)
class SRAMEqualBaseGroup:
    """Locations of one owner that require one physical base address."""

    locations: Tuple[SRAMLocation, ...]

    def __post_init__(self):
        if not isinstance(self.locations, tuple) or any(
            not isinstance(location, SRAMLocation) for location in self.locations
        ):
            raise TypeError("SRAM equal-base locations must be SRAMLocation values")
        normalized = tuple(sorted(self.locations))
        if len(normalized) < 2:
            raise ValueError(
                "SRAM equal-base group must contain at least two locations"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("SRAM equal-base group contains duplicate locations")
        object.__setattr__(self, "locations", normalized)


@dataclass(frozen=True)
class SRAMLocationRequirement:
    """Describe one owner's storage interval at one device and worker core."""

    location: SRAMLocation
    extent_bytes: int
    payload_present: bool
    fixed_base: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.location, SRAMLocation):
            raise TypeError("SRAM location requirement needs an SRAMLocation")
        if type(self.extent_bytes) is not int or self.extent_bytes <= 0:
            raise ValueError("SRAM location extent must be a positive integer")
        if type(self.payload_present) is not bool:
            raise TypeError("SRAM payload presence must be a bool")
        if self.fixed_base is not None and (
            type(self.fixed_base) is not int or self.fixed_base < 0
        ):
            raise ValueError("fixed SRAM base must be a nonnegative integer")
        if (
            self.fixed_base is not None
            and self.fixed_base > _SRAM_ADDRESS_LIMIT - self.extent_bytes
        ):
            raise ValueError("fixed SRAM interval exceeds the 32-bit address space")


@dataclass(frozen=True)
class SRAMOwner:
    """Identify one owner within an immutable requirement set."""

    kind: SRAMOwnerKind
    index: int

    def __post_init__(self):
        if not isinstance(self.kind, SRAMOwnerKind):
            raise TypeError("SRAM owner kind must be an SRAMOwnerKind")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("SRAM owner index must be a nonnegative integer")


@dataclass(frozen=True)
class SRAMStorageRequirement:
    """Describe one physical owner across location-specific intervals."""

    owner: SRAMOwner
    location_requirements: Tuple[SRAMLocationRequirement, ...]
    alignment_bytes: int
    equal_base_groups: Tuple[SRAMEqualBaseGroup, ...]
    ownership: SRAMOwnership
    lifetime: SRAMLifetime

    def __post_init__(self):
        if not isinstance(self.owner, SRAMOwner):
            raise TypeError("SRAM requirement owner must be an SRAMOwner")
        if not isinstance(self.location_requirements, tuple) or any(
            not isinstance(requirement, SRAMLocationRequirement)
            for requirement in self.location_requirements
        ):
            raise TypeError(
                "SRAM location requirements must be SRAMLocationRequirement values"
            )
        if not isinstance(self.equal_base_groups, tuple) or any(
            not isinstance(group, SRAMEqualBaseGroup)
            for group in self.equal_base_groups
        ):
            raise TypeError("SRAM equal-base groups must be SRAMEqualBaseGroup values")
        normalized_groups = tuple(
            sorted(self.equal_base_groups, key=lambda group: group.locations)
        )
        object.__setattr__(self, "equal_base_groups", normalized_groups)
        if not isinstance(self.ownership, SRAMOwnership):
            raise TypeError("SRAM requirement ownership must be an SRAMOwnership")
        if not isinstance(self.lifetime, SRAMLifetime):
            raise TypeError("SRAM requirement lifetime must be an SRAMLifetime")
        if (
            type(self.alignment_bytes) is not int
            or self.alignment_bytes <= 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise ValueError(
                "SRAM requirement alignment must be a positive power of two"
            )
        normalized = tuple(
            sorted(self.location_requirements, key=lambda value: value.location)
        )
        if not normalized:
            raise ValueError("SRAM requirement must contain a location")
        locations = tuple(value.location for value in normalized)
        if len(set(locations)) != len(locations):
            raise ValueError("SRAM requirement contains duplicate locations")
        object.__setattr__(self, "location_requirements", normalized)
        grouped_locations = []
        location_set = set(locations)
        for group in self.equal_base_groups:
            if not set(group.locations).issubset(location_set):
                raise ValueError(
                    "SRAM equal-base group contains a location outside its owner"
                )
            grouped_locations.extend(group.locations)
        if len(set(grouped_locations)) != len(grouped_locations):
            raise ValueError("SRAM equal-base groups must be disjoint")
        if self.ownership is SRAMOwnership.FIXED:
            if any(value.fixed_base is None for value in normalized):
                raise ValueError("fixed SRAM requirement needs one base per location")
        elif any(value.fixed_base is not None for value in normalized):
            raise ValueError("movable SRAM requirement cannot contain fixed bases")
        for value in normalized:
            if value.fixed_base is not None and value.fixed_base % self.alignment_bytes:
                raise ValueError("fixed SRAM base does not satisfy its alignment")
        requirements_by_location = {
            value.location: value for value in self.location_requirements
        }
        for group in self.equal_base_groups:
            fixed_bases = {
                requirements_by_location[location].fixed_base
                for location in group.locations
            }
            if len(fixed_bases) != 1:
                raise ValueError(
                    "fixed locations in one equal-base group need the same base"
                )

    @property
    def locations(self) -> Tuple[SRAMLocation, ...]:
        return tuple(value.location for value in self.location_requirements)

    @property
    def max_extent_bytes(self) -> int:
        return max(value.extent_bytes for value in self.location_requirements)

    def at(self, location: SRAMLocation) -> SRAMLocationRequirement:
        for requirement in self.location_requirements:
            if requirement.location == location:
                return requirement
        raise ValueError("SRAM location is outside its storage owner")


@dataclass(frozen=True)
class SRAMUse:
    """Bind one logical DFB byte range to a physical storage requirement."""

    kind: SRAMUseKind
    dfb_index: int
    segment_index: Optional[int]
    requirement_index: int
    byte_offset: int
    byte_size: int
    locations: Tuple[SRAMLocation, ...]

    def __post_init__(self):
        if not isinstance(self.kind, SRAMUseKind):
            raise TypeError("SRAM use kind must be an SRAMUseKind")
        if not isinstance(self.locations, tuple) or any(
            not isinstance(location, SRAMLocation) for location in self.locations
        ):
            raise TypeError("SRAM use locations must be SRAMLocation values")
        for name, value in (
            ("DFB index", self.dfb_index),
            ("requirement index", self.requirement_index),
            ("byte offset", self.byte_offset),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"SRAM use {name} must be a nonnegative integer")
        if self.kind is SRAMUseKind.DFB_CONTROL:
            if self.segment_index is not None:
                raise ValueError("DFB control use cannot reference a storage segment")
        elif type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError(
                "DFB payload use segment index must be a nonnegative integer"
            )
        if type(self.byte_size) is not int or self.byte_size <= 0:
            raise ValueError("SRAM use byte size must be a positive integer")
        normalized = tuple(sorted(self.locations))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("SRAM use locations must be nonempty and distinct")
        object.__setattr__(self, "locations", normalized)


@dataclass(frozen=True)
class SRAMArenaBinding:
    """Bind one invocation arena requirement to its worker cores."""

    requirement_index: int
    cores: Tuple[Tuple[int, int], ...]

    def __post_init__(self):
        if not isinstance(self.cores, tuple):
            raise TypeError("SRAM arena cores must be a tuple")
        if type(self.requirement_index) is not int or self.requirement_index < 0:
            raise ValueError("SRAM arena requirement index must be nonnegative")
        normalized = tuple(sorted(self.cores))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("SRAM arena cores must be nonempty and distinct")
        for core in normalized:
            SRAMLocation((), core)
        object.__setattr__(self, "cores", normalized)


@dataclass(frozen=True)
class PreparedSRAMOperation:
    """Contain validated storage requirements for one operation invocation."""

    name: str
    requirements: Tuple[SRAMStorageRequirement, ...]
    uses: Tuple[SRAMUse, ...]
    arenas: Tuple[SRAMArenaBinding, ...]

    def __post_init__(self):
        if (
            not isinstance(self.requirements, tuple)
            or not isinstance(self.uses, tuple)
            or not isinstance(self.arenas, tuple)
        ):
            raise TypeError("prepared SRAM operation records must be tuples")
        if any(
            not isinstance(requirement, SRAMStorageRequirement)
            for requirement in self.requirements
        ):
            raise TypeError("prepared SRAM requirements have the wrong type")
        if any(not isinstance(use, SRAMUse) for use in self.uses):
            raise TypeError("prepared SRAM uses have the wrong type")
        if any(not isinstance(arena, SRAMArenaBinding) for arena in self.arenas):
            raise TypeError("prepared SRAM arena bindings have the wrong type")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("prepared SRAM operation name must be nonempty")
        owners = [requirement.owner for requirement in self.requirements]
        if len(set(owners)) != len(owners):
            raise ValueError("prepared SRAM operation contains duplicate owners")
        arena_requirements = set()
        for arena in self.arenas:
            requirement = self._requirement(arena.requirement_index)
            if requirement.owner.kind is not SRAMOwnerKind.COMPILER_ARENA:
                raise ValueError("SRAM arena must reference a compiler arena owner")
            if arena.requirement_index in arena_requirements:
                raise ValueError("compiler arena requirement has multiple bindings")
            arena_requirements.add(arena.requirement_index)
            expected_locations = {SRAMLocation((), core) for core in arena.cores}
            actual_locations = set(requirement.locations)
            if actual_locations != expected_locations:
                raise ValueError("compiler arena cores do not match its locations")
        required_arena_indices = {
            requirement_index
            for requirement_index, requirement in enumerate(self.requirements)
            if requirement.owner.kind is SRAMOwnerKind.COMPILER_ARENA
        }
        if arena_requirements != required_arena_indices:
            raise ValueError("compiler arena requirement has no arena binding")
        self.arena_bytes_by_core()
        for use in self.uses:
            requirement = self._requirement(use.requirement_index)
            if not set(use.locations).issubset(requirement.locations):
                raise ValueError("SRAM use location is outside its requirement")
            if use.kind is SRAMUseKind.DFB_PAYLOAD and any(
                not requirement.at(location).payload_present
                for location in use.locations
            ):
                raise ValueError("SRAM payload use has no storage at its location")
            if any(
                use.byte_offset > requirement.at(location).extent_bytes - use.byte_size
                for location in use.locations
            ):
                raise ValueError("SRAM use byte range exceeds its requirement")

    def _requirement(self, index: int) -> SRAMStorageRequirement:
        if type(index) is not int or not 0 <= index < len(self.requirements):
            raise ValueError("SRAM record references an unknown requirement")
        return self.requirements[index]

    @property
    def uses_compiler_arena(self) -> bool:
        return bool(self.arenas)

    def arena_bytes_by_core(self) -> dict[Tuple[int, int], int]:
        result = {}
        for arena in self.arenas:
            requirement = self.requirements[arena.requirement_index]
            for core in arena.cores:
                if core in result:
                    raise ValueError("worker core belongs to multiple compiler arenas")
                result[core] = requirement.at(SRAMLocation((), core)).extent_bytes
        return result


@dataclass(frozen=True)
class PersistentSRAMDeclaration:
    """Describe one persistent tensor before physical reservation."""

    extent_bytes: int
    alignment_bytes: int
    cores: Tuple[Tuple[int, int], ...]
    addressing: SRAMAddressing

    def __post_init__(self):
        if not isinstance(self.addressing, SRAMAddressing):
            raise TypeError("persistent SRAM addressing must be an SRAMAddressing")
        if not isinstance(self.cores, tuple):
            raise TypeError("persistent SRAM cores must be a tuple")
        if type(self.extent_bytes) is not int or self.extent_bytes <= 0:
            raise ValueError("persistent SRAM extent must be a positive integer")
        if (
            type(self.alignment_bytes) is not int
            or self.alignment_bytes <= 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise ValueError(
                "persistent SRAM alignment must be a positive power of two"
            )
        normalized = tuple(sorted(self.cores))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("persistent SRAM cores must be nonempty and distinct")
        for core in normalized:
            SRAMLocation((), core)
        object.__setattr__(self, "cores", normalized)


@dataclass(frozen=True)
class PreparedSRAMStorage:
    """Contain movable persistent requirements prepared before reservation."""

    requirements: Tuple[SRAMStorageRequirement, ...]

    def __post_init__(self):
        if not isinstance(self.requirements, tuple) or any(
            not isinstance(requirement, SRAMStorageRequirement)
            for requirement in self.requirements
        ):
            raise TypeError("prepared persistent SRAM requirements have the wrong type")
        for requirement_index, requirement in enumerate(self.requirements):
            expected_owner = SRAMOwner(
                SRAMOwnerKind.PERSISTENT_DECLARATION, requirement_index
            )
            if requirement.owner != expected_owner:
                raise ValueError(
                    "persistent SRAM requirements must use dense declaration owners"
                )
            if (
                requirement.ownership is not SRAMOwnership.MOVABLE
                or requirement.lifetime is not SRAMLifetime.PERSISTENT
            ):
                raise ValueError(
                    "persistent SRAM declaration must remain movable until reservation"
                )


def prepare_persistent_storage(
    declarations: Sequence[PersistentSRAMDeclaration],
) -> PreparedSRAMStorage:
    """Convert persistent declarations to common movable requirements."""
    requirements = []
    for declaration_index, declaration in enumerate(declarations):
        locations = _locations(declaration.cores)
        equal_base_groups = (
            (SRAMEqualBaseGroup(locations),)
            if declaration.addressing is SRAMAddressing.UNIFORM and len(locations) > 1
            else ()
        )
        requirements.append(
            SRAMStorageRequirement(
                owner=SRAMOwner(
                    SRAMOwnerKind.PERSISTENT_DECLARATION, declaration_index
                ),
                location_requirements=tuple(
                    SRAMLocationRequirement(
                        location,
                        declaration.extent_bytes,
                        payload_present=True,
                    )
                    for location in locations
                ),
                alignment_bytes=declaration.alignment_bytes,
                equal_base_groups=equal_base_groups,
                ownership=SRAMOwnership.MOVABLE,
                lifetime=SRAMLifetime.PERSISTENT,
            )
        )
    return PreparedSRAMStorage(tuple(requirements))


def _locations(
    cores: Sequence[Tuple[int, int]],
    device_coordinates: Sequence[Tuple[int, ...]] = ((),),
) -> Tuple[SRAMLocation, ...]:
    return tuple(
        SRAMLocation(tuple(device), tuple(core))
        for device in device_coordinates
        for core in cores
    )


def _sram_alignment(ttnn_api: Any) -> int:
    get_alignment = getattr(ttnn_api, "get_l1_alignment", None)
    if not callable(get_alignment):
        raise RuntimeError("TTNN does not expose the target SRAM alignment")
    alignment = get_alignment()
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("TTNN returned an invalid target SRAM alignment")
    return alignment


def compiler_arena_bytes(configs) -> Optional[int]:
    has_metadata = any(
        config.l1_offset is not None
        or config.l1_payload_offset is not None
        or config.l1_allocation_bytes is not None
        for config in configs
    )
    if not has_metadata:
        return None
    if not all(config.l1_offset is not None for config in configs):
        raise ValueError("mixed compiler SRAM and Metal storage metadata")
    arena_ends = [config.l1_offset + _DFB_CONTROL_BYTES for config in configs]
    for config in configs:
        has_payload_offset = config.l1_payload_offset is not None
        has_allocation_bytes = config.l1_allocation_bytes is not None
        if has_payload_offset != has_allocation_bytes:
            raise ValueError("incomplete compiler SRAM payload allocation metadata")
        if config.sram_core_layouts:
            arena_ends.extend(layout.arena_bytes for layout in config.sram_core_layouts)
        if has_payload_offset:
            if not config.sram_core_layouts:
                arena_ends.append(config.l1_payload_offset + config.l1_allocation_bytes)
            continue
        if not config.storage_segments or any(
            not segment.is_tensor_backed for segment in config.storage_segments
        ):
            raise ValueError(
                "compiler SRAM storage without an arena payload requires "
                "tensor backing on every storage segment"
            )
    return max(arena_ends)


def _tensor_device_coordinates(tensor: Any) -> Tuple[Tuple[int, ...], ...]:
    get_coordinates = getattr(tensor, "device_coords", None)
    if not callable(get_coordinates):
        return ((),)
    coordinates = tuple(
        tuple(int(value) for value in coordinate) for coordinate in get_coordinates()
    )
    return coordinates or ((),)


def _tensor_requirement(
    ttnn_api: Any,
    tensor: Any,
    tensor_index: int,
    cores: Sequence[Tuple[int, int]],
    extent_bytes: int,
    alignment_bytes: int,
) -> SRAMStorageRequirement:
    device_coordinates = _tensor_device_coordinates(tensor)
    get_per_core_allocation = getattr(tensor, "is_per_core_allocated", None)
    if not callable(get_per_core_allocation):
        raise ValueError("SRAM tensor does not expose its address allocation mode")
    try:
        per_core = bool(get_per_core_allocation())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "failed to query SRAM tensor address allocation mode"
        ) from error
    if per_core and device_coordinates == ((),):
        raise ValueError("per-core SRAM tensor must expose logical device coordinates")
    if per_core:
        location_requirements = []
        for device_coordinate in device_coordinates:
            for core in sorted(cores):
                location = SRAMLocation(device_coordinate, core)
                location_requirements.append(
                    SRAMLocationRequirement(
                        location,
                        extent_bytes,
                        payload_present=True,
                        fixed_base=int(
                            tensor.experimental_per_core_buffer_address(
                                ttnn_api.MeshCoordinate(device_coordinate),
                                ttnn_api.CoreCoord(*core),
                            )
                        ),
                    )
                )
        equal_base_groups = ()
    else:
        locations = _locations(cores, device_coordinates)
        base = int(tensor.buffer_address())
        location_requirements = [
            SRAMLocationRequirement(
                location,
                extent_bytes,
                payload_present=True,
                fixed_base=base,
            )
            for location in locations
        ]
        equal_base_groups = (
            (SRAMEqualBaseGroup(locations),) if len(locations) > 1 else ()
        )
    return SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, tensor_index),
        location_requirements=tuple(location_requirements),
        alignment_bytes=alignment_bytes,
        equal_base_groups=equal_base_groups,
        ownership=SRAMOwnership.FIXED,
        lifetime=SRAMLifetime.EXTERNAL,
    )


def _tensor_cores(ttnn_api: Any, tensor: Any) -> Tuple[Tuple[int, int], ...]:
    try:
        grid = tensor.memory_config().shard_spec.grid
        cores = tuple(
            sorted(
                (int(core.x), int(core.y))
                for core in ttnn_api.corerange_to_cores(grid, row_wise=True)
            )
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("SRAM tensor does not expose a valid shard grid") from error
    if not cores or len(set(cores)) != len(cores):
        raise ValueError("SRAM tensor shard grid must contain distinct worker cores")
    return cores


def prepare_sram_operation(
    *,
    name: str,
    tensors: Sequence[Any],
    configs: Sequence[Any],
    cores: Sequence[Tuple[int, int]],
    ttnn_api: Any,
) -> PreparedSRAMOperation:
    """Build and validate one operation's physical owners and logical uses."""
    cores = tuple(sorted(tuple(core) for core in cores))
    if not cores or len(set(cores)) != len(cores):
        raise ValueError("prepared SRAM operation cores must be nonempty and distinct")
    for core in cores:
        SRAMLocation((), core)

    requirements = []
    uses = []
    arenas = []
    alignment_bytes = None

    def get_alignment_bytes():
        nonlocal alignment_bytes
        if alignment_bytes is None:
            alignment_bytes = _sram_alignment(ttnn_api)
        return alignment_bytes

    arena_bytes = compiler_arena_bytes(configs)
    per_core_layout = any(config.sram_core_layouts for config in configs)

    if per_core_layout:
        from ._sram_domains import core_domains, validate_core_layouts

        sizes = validate_core_layouts(configs, cores)
        for arena_index, arena_cores in enumerate(core_domains(configs)):
            requirement_index = len(requirements)
            domain_locations = _locations(arena_cores)
            location_requirements = []
            for core, location in zip(arena_cores, domain_locations):
                layouts = [
                    next(
                        layout
                        for layout in config.sram_core_layouts
                        if layout.node == core
                    )
                    for config in configs
                ]
                location_requirements.append(
                    SRAMLocationRequirement(
                        location,
                        sizes[core],
                        payload_present=any(
                            layout.payload_present for layout in layouts
                        ),
                    )
                )
            requirements.append(
                SRAMStorageRequirement(
                    owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, arena_index),
                    location_requirements=tuple(location_requirements),
                    alignment_bytes=get_alignment_bytes(),
                    equal_base_groups=(
                        (SRAMEqualBaseGroup(domain_locations),)
                        if len(domain_locations) > 1
                        else ()
                    ),
                    ownership=SRAMOwnership.MOVABLE,
                    lifetime=SRAMLifetime.INVOCATION,
                )
            )
            arenas.append(SRAMArenaBinding(requirement_index, arena_cores))
            for config in configs:
                uses.append(
                    SRAMUse(
                        SRAMUseKind.DFB_CONTROL,
                        config.dfb_index,
                        None,
                        requirement_index,
                        config.l1_offset,
                        _DFB_CONTROL_BYTES,
                        domain_locations,
                    )
                )
                payload_locations_by_offset = {}
                for layout in config.sram_core_layouts:
                    if layout.node in arena_cores and layout.payload_present:
                        payload_locations_by_offset.setdefault(
                            layout.payload_offset, []
                        ).append(SRAMLocation((), layout.node))
                for payload_offset, payload_locations in sorted(
                    payload_locations_by_offset.items()
                ):
                    uses.append(
                        SRAMUse(
                            SRAMUseKind.DFB_PAYLOAD,
                            config.dfb_index,
                            arena_index,
                            requirement_index,
                            payload_offset,
                            config.l1_allocation_bytes,
                            tuple(payload_locations),
                        )
                    )
    elif arena_bytes is not None:
        requirement_index = len(requirements)
        domain_locations = _locations(cores)
        requirements.append(
            SRAMStorageRequirement(
                owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
                location_requirements=tuple(
                    SRAMLocationRequirement(
                        location,
                        arena_bytes,
                        payload_present=any(
                            config.l1_payload_offset is not None for config in configs
                        ),
                    )
                    for location in domain_locations
                ),
                alignment_bytes=get_alignment_bytes(),
                equal_base_groups=(
                    (SRAMEqualBaseGroup(domain_locations),)
                    if len(domain_locations) > 1
                    else ()
                ),
                ownership=SRAMOwnership.MOVABLE,
                lifetime=SRAMLifetime.INVOCATION,
            )
        )
        arenas.append(SRAMArenaBinding(requirement_index, cores))
        for config in configs:
            uses.append(
                SRAMUse(
                    SRAMUseKind.DFB_CONTROL,
                    config.dfb_index,
                    None,
                    requirement_index,
                    config.l1_offset,
                    _DFB_CONTROL_BYTES,
                    domain_locations,
                )
            )
            if config.l1_payload_offset is not None:
                uses.append(
                    SRAMUse(
                        SRAMUseKind.DFB_PAYLOAD,
                        config.dfb_index,
                        0,
                        requirement_index,
                        config.l1_payload_offset,
                        config.l1_allocation_bytes,
                        domain_locations,
                    )
                )

    tensor_segments = {}
    for config in configs:
        for segment_index, segment in enumerate(config.storage_segments):
            if not segment.is_tensor_backed:
                continue
            if segment.tensor_index < 0 or segment.tensor_index >= len(tensors):
                raise ValueError(
                    f"DFB[{config.dfb_index}] references invalid tensor index "
                    f"{segment.tensor_index}"
                )
            if segment.byte_size is None or segment.byte_size <= 0:
                raise ValueError(
                    "tensor-backed DFB segment must have a positive byte size"
                )
            if not set(segment.nodes).issubset(cores):
                raise ValueError(
                    "tensor-backed DFB segment is outside the operation cores"
                )
            tensor_segments.setdefault(segment.tensor_index, []).append(
                (config.dfb_index, segment_index, segment)
            )

    for tensor_index, segments in sorted(tensor_segments.items()):
        from .dataflow_buffer import (
            _validate_tensor_backed_dfb_range,
            _validate_tensor_backed_dfb_tensor,
        )

        requirement_index = len(requirements)
        tensor = tensors[tensor_index]
        properties = _validate_tensor_backed_dfb_tensor(
            tensor, context=f"tensor argument {tensor_index}"
        )
        for dfb_index, segment_index, segment in segments:
            _validate_tensor_backed_dfb_range(
                properties,
                byte_offset=segment.byte_offset,
                byte_size=segment.byte_size,
                context=f"DFB[{dfb_index}] storage segment {segment_index}",
            )
        tensor_cores = _tensor_cores(ttnn_api, tensor)
        requirements.append(
            _tensor_requirement(
                ttnn_api,
                tensor,
                tensor_index,
                tensor_cores,
                properties.logical_shard_size_bytes,
                get_alignment_bytes(),
            )
        )
        device_coordinates = _tensor_device_coordinates(tensor)
        for dfb_index, segment_index, segment in segments:
            uses.append(
                SRAMUse(
                    SRAMUseKind.DFB_PAYLOAD,
                    dfb_index,
                    segment_index,
                    requirement_index,
                    segment.byte_offset,
                    segment.byte_size,
                    _locations(segment.nodes, device_coordinates),
                )
            )

    return PreparedSRAMOperation(
        name=name,
        requirements=tuple(requirements),
        uses=tuple(uses),
        arenas=tuple(arenas),
    )
