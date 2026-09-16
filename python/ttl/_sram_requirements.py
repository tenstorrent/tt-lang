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
    """State whether placement is supplied or remains allocator-selectable."""

    FIXED = auto()
    MOVABLE = auto()


class SRAMAddressing(Enum):
    """State whether participating locations share one base address."""

    UNIFORM = auto()
    PER_CORE = auto()


class SRAMLifetime(Enum):
    """State which owner controls the end of a requirement's lifetime."""

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
class SRAMAddressDomain:
    """Locations that require one equal physical base address."""

    locations: Tuple[SRAMLocation, ...]

    def __post_init__(self):
        if not isinstance(self.locations, tuple) or any(
            not isinstance(location, SRAMLocation) for location in self.locations
        ):
            raise TypeError("SRAM address-domain locations must be SRAMLocation values")
        normalized = tuple(sorted(self.locations))
        if not normalized:
            raise ValueError("SRAM address domain must contain at least one location")
        if len(set(normalized)) != len(normalized):
            raise ValueError("SRAM address domain contains duplicate locations")
        object.__setattr__(self, "locations", normalized)


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
    """Describe one physical allocation across its address-equality domains."""

    owner: SRAMOwner
    extent_bytes: int
    alignment_bytes: int
    address_domains: Tuple[SRAMAddressDomain, ...]
    ownership: SRAMOwnership
    lifetime: SRAMLifetime
    fixed_bases: Tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.owner, SRAMOwner):
            raise TypeError("SRAM requirement owner must be an SRAMOwner")
        if not isinstance(self.address_domains, tuple) or any(
            not isinstance(domain, SRAMAddressDomain) for domain in self.address_domains
        ):
            raise TypeError(
                "SRAM requirement address domains must be SRAMAddressDomain values"
            )
        if not isinstance(self.ownership, SRAMOwnership):
            raise TypeError("SRAM requirement ownership must be an SRAMOwnership")
        if not isinstance(self.lifetime, SRAMLifetime):
            raise TypeError("SRAM requirement lifetime must be an SRAMLifetime")
        if not isinstance(self.fixed_bases, tuple):
            raise TypeError("SRAM requirement fixed bases must be a tuple")
        if type(self.extent_bytes) is not int or self.extent_bytes <= 0:
            raise ValueError("SRAM requirement extent must be a positive integer")
        if (
            type(self.alignment_bytes) is not int
            or self.alignment_bytes <= 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise ValueError(
                "SRAM requirement alignment must be a positive power of two"
            )
        if not self.address_domains:
            raise ValueError("SRAM requirement must contain an address domain")
        locations = [
            location for domain in self.address_domains for location in domain.locations
        ]
        if len(set(locations)) != len(locations):
            raise ValueError("SRAM requirement address domains must be disjoint")
        if self.ownership is SRAMOwnership.FIXED:
            if len(self.fixed_bases) != len(self.address_domains):
                raise ValueError(
                    "fixed SRAM requirement needs one base per address domain"
                )
        elif self.fixed_bases:
            raise ValueError("movable SRAM requirement cannot contain fixed bases")
        for base in self.fixed_bases:
            if type(base) is not int or base < 0:
                raise ValueError("fixed SRAM base must be a nonnegative integer")
            if base % self.alignment_bytes:
                raise ValueError("fixed SRAM base does not satisfy its alignment")
            if base > _SRAM_ADDRESS_LIMIT - self.extent_bytes:
                raise ValueError("fixed SRAM interval exceeds the 32-bit address space")


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
            actual_locations = {
                location
                for domain in requirement.address_domains
                for location in domain.locations
            }
            if actual_locations != expected_locations:
                raise ValueError(
                    "compiler arena cores do not match its address domains"
                )
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
            if use.byte_offset > requirement.extent_bytes - use.byte_size:
                raise ValueError("SRAM use byte range exceeds its requirement")
            requirement_locations = {
                location
                for domain in requirement.address_domains
                for location in domain.locations
            }
            if not set(use.locations).issubset(requirement_locations):
                raise ValueError("SRAM use location is outside its requirement")

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
            extent = self.requirements[arena.requirement_index].extent_bytes
            for core in arena.cores:
                if core in result:
                    raise ValueError("worker core belongs to multiple compiler arenas")
                result[core] = extent
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
        if declaration.addressing is SRAMAddressing.UNIFORM:
            domains = (SRAMAddressDomain(locations),)
        else:
            domains = tuple(SRAMAddressDomain((location,)) for location in locations)
        requirements.append(
            SRAMStorageRequirement(
                owner=SRAMOwner(
                    SRAMOwnerKind.PERSISTENT_DECLARATION, declaration_index
                ),
                extent_bytes=declaration.extent_bytes,
                alignment_bytes=declaration.alignment_bytes,
                address_domains=domains,
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
        domains = []
        bases = []
        for device_coordinate in device_coordinates:
            for core in sorted(cores):
                location = SRAMLocation(device_coordinate, core)
                domains.append(SRAMAddressDomain((location,)))
                bases.append(
                    int(
                        tensor.experimental_per_core_buffer_address(
                            ttnn_api.MeshCoordinate(device_coordinate),
                            ttnn_api.CoreCoord(*core),
                        )
                    )
                )
    else:
        domains = [SRAMAddressDomain(_locations(cores, device_coordinates))]
        bases = [int(tensor.buffer_address())]
    return SRAMStorageRequirement(
        owner=SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, tensor_index),
        extent_bytes=extent_bytes,
        alignment_bytes=alignment_bytes,
        address_domains=tuple(domains),
        ownership=SRAMOwnership.FIXED,
        lifetime=SRAMLifetime.EXTERNAL,
        fixed_bases=tuple(bases),
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
            requirements.append(
                SRAMStorageRequirement(
                    owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, arena_index),
                    extent_bytes=sizes[arena_cores[0]],
                    alignment_bytes=get_alignment_bytes(),
                    address_domains=(SRAMAddressDomain(domain_locations),),
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
                layout = next(
                    layout
                    for layout in config.sram_core_layouts
                    if layout.node == arena_cores[0]
                )
                if layout.payload_present:
                    uses.append(
                        SRAMUse(
                            SRAMUseKind.DFB_PAYLOAD,
                            config.dfb_index,
                            arena_index,
                            requirement_index,
                            layout.payload_offset,
                            config.l1_allocation_bytes,
                            domain_locations,
                        )
                    )
    elif arena_bytes is not None:
        requirement_index = len(requirements)
        domain_locations = _locations(cores)
        requirements.append(
            SRAMStorageRequirement(
                owner=SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0),
                extent_bytes=arena_bytes,
                alignment_bytes=get_alignment_bytes(),
                address_domains=(SRAMAddressDomain(domain_locations),),
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
