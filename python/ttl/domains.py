# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Logical device domains and device-level transfer graphs.

This module contains architecture-neutral frontend metadata. Explicit graphs
store O(E) user edges. Structured graphs store O(1) descriptors. Neither form
allocates runtime host state or device-visible communication memory.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, Union

Coordinate = Tuple[int, ...]
ComponentCoordinates = Tuple[Coordinate, ...]


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer, got {value!r}")
    return value


def _normalize_extent(extent: Sequence[int], context: str) -> Coordinate:
    if isinstance(extent, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of integers, got {extent!r}")
    normalized = tuple(_require_int(dim, f"{context} dimension") for dim in extent)
    if not normalized:
        raise ValueError(f"{context} must have at least one dimension")
    for axis, dimension in enumerate(normalized):
        if dimension <= 0:
            raise ValueError(f"{context} axis {axis} must be positive, got {dimension}")
    return normalized


def _normalize_coordinate(value: Any, context: str) -> Coordinate:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{context} must be an integer or integer sequence, got {value!r}"
        )
    coordinate = tuple(
        _require_int(component, f"{context} component") for component in value
    )
    if not coordinate:
        raise ValueError(f"{context} must have at least one component")
    for axis, component in enumerate(coordinate):
        if component < 0:
            raise ValueError(
                f"{context} axis {axis} must be non-negative, got {component}"
            )
    return coordinate


@dataclass(frozen=True)
class DomainComponent:
    """Named rectangular component of a logical device domain."""

    name: str
    extent: Sequence[int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                f"domain component name must be a string, got {self.name!r}"
            )
        if not self.name:
            raise ValueError("domain component name must not be empty")
        object.__setattr__(
            self,
            "extent",
            _normalize_extent(self.extent, f"domain component {self.name!r} extent"),
        )


@dataclass(frozen=True, init=False)
class DeviceRef:
    """Coordinate of one member of a `DeviceDomain`."""

    coordinates: ComponentCoordinates
    component_names: Tuple[str, ...]

    def __init__(self, *coordinates: Any, **named_coordinates: Any) -> None:
        if coordinates and named_coordinates:
            raise ValueError(
                "DeviceRef accepts positional coordinates or named coordinates, not both"
            )
        if named_coordinates:
            component_names = tuple(named_coordinates.keys())
            normalized = tuple(
                _normalize_coordinate(coord, f"DeviceRef {name!r}")
                for name, coord in named_coordinates.items()
            )
        else:
            if not coordinates:
                raise ValueError("DeviceRef requires at least one coordinate")
            component_names = ()
            normalized = tuple(
                _normalize_coordinate(coord, "DeviceRef coordinate")
                for coord in coordinates
            )
        object.__setattr__(self, "coordinates", normalized)
        object.__setattr__(self, "component_names", component_names)

    @property
    def is_named(self) -> bool:
        return bool(self.component_names)


@dataclass(frozen=True)
class DeviceRange:
    """Half-open rectangular range in a `DeviceDomain`."""

    lo: DeviceRef
    hi: DeviceRef

    def __post_init__(self) -> None:
        if not isinstance(self.lo, DeviceRef) or not isinstance(self.hi, DeviceRef):
            raise TypeError("DeviceRange endpoints must be DeviceRef values")


@dataclass(frozen=True)
class TransferEdge:
    """Device-level transfer edge without core or dataflow-buffer binding."""

    source: DeviceRef
    destination: Union[DeviceRef, DeviceRange]


@dataclass(frozen=True)
class StructuredTransfer:
    """Base class for compact transfer relations."""

    component_name: str


@dataclass(frozen=True)
class AxisNeighborTransfer(StructuredTransfer):
    """Transfer relation between neighbors along one domain axis."""

    axis: int
    offset: int
    wrap: bool


@dataclass(frozen=True)
class GatherTransfer(StructuredTransfer):
    """Transfer relation from every non-root device to one root."""

    root: DeviceRef


@dataclass(frozen=True)
class MulticastTransfer(StructuredTransfer):
    """Transfer relation from one source to every other device."""

    source: DeviceRef


@dataclass(frozen=True)
class GraphMetadataCost:
    """Asymptotic storage summary for a `TransferGraph`."""

    storage_class: str
    compile_time: str
    runtime_host: str
    device_visible: str


@dataclass(frozen=True, init=False)
class DeviceDomain:
    """Logical index set of devices.

    A domain contains one or more named rectangular components. A single
    component represents a regular domain. Multiple components represent a
    product domain without implying any physical device hierarchy or topology.
    """

    components: Tuple[DomainComponent, ...]

    def __init__(self, extent: Sequence[int], *, name: str = "device") -> None:
        object.__setattr__(self, "components", (DomainComponent(name, extent),))

    @classmethod
    def product(
        cls, **components: Union["DeviceDomain", Sequence[int]]
    ) -> "DeviceDomain":
        if not components:
            raise ValueError("DeviceDomain.product requires at least one component")

        domain_components = []
        for name, component in components.items():
            if isinstance(component, DeviceDomain):
                if component.component_count != 1:
                    raise ValueError(
                        "nested product-domain components are not supported; "
                        "provide each component directly"
                    )
                extent = component.components[0].extent
            else:
                extent = component
            domain_components.append(DomainComponent(name, extent))

        domain = cls.__new__(cls)
        object.__setattr__(domain, "components", tuple(domain_components))
        return domain

    @property
    def component_names(self) -> Tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def shape(self) -> Coordinate:
        if self.component_count != 1:
            raise ValueError("product DeviceDomain does not have one rectangular shape")
        return tuple(self.components[0].extent)

    def component_index(self, component_name: str) -> int:
        for index, component in enumerate(self.components):
            if component.name == component_name:
                return index
        raise ValueError(f"unknown domain component {component_name!r}")

    def device_ref(self, value: Any) -> DeviceRef:
        if isinstance(value, DeviceRef):
            return self.resolve_device_ref(value)
        if self.component_count != 1:
            raise ValueError(
                "product domains require DeviceRef values or named coordinates"
            )
        return self.resolve_device_ref(DeviceRef(value))

    def resolve_device_ref(self, device_ref: DeviceRef) -> DeviceRef:
        coordinates = self._coordinates_in_component_order(device_ref)
        self._validate_coordinates(coordinates, allow_upper_bound=False)
        return DeviceRef(*coordinates)

    def resolve_device_range(self, device_range: DeviceRange) -> DeviceRange:
        lo_coordinates = self._coordinates_in_component_order(device_range.lo)
        hi_coordinates = self._coordinates_in_component_order(device_range.hi)
        self._validate_coordinates(lo_coordinates, allow_upper_bound=False)
        self._validate_coordinates(hi_coordinates, allow_upper_bound=True)
        for component, lo_coord, hi_coord in zip(
            self.components, lo_coordinates, hi_coordinates
        ):
            for axis, (lo_component, hi_component) in enumerate(
                zip(lo_coord, hi_coord)
            ):
                if lo_component >= hi_component:
                    raise ValueError(
                        f"DeviceRange component {component.name!r} axis {axis} "
                        f"requires lo < hi, got lo={lo_component}, hi={hi_component}"
                    )
        return DeviceRange(DeviceRef(*lo_coordinates), DeviceRef(*hi_coordinates))

    def _coordinates_in_component_order(
        self, device_ref: DeviceRef
    ) -> ComponentCoordinates:
        if device_ref.is_named:
            expected_names = self.component_names
            if set(device_ref.component_names) != set(expected_names):
                raise ValueError(
                    f"DeviceRef names {device_ref.component_names} do not match "
                    f"domain components {expected_names}"
                )
            by_name = dict(zip(device_ref.component_names, device_ref.coordinates))
            return tuple(by_name[name] for name in expected_names)

        if len(device_ref.coordinates) != self.component_count:
            raise ValueError(
                f"DeviceRef has {len(device_ref.coordinates)} component coordinates, "
                f"domain has {self.component_count}"
            )
        return device_ref.coordinates

    def _validate_coordinates(
        self, coordinates: ComponentCoordinates, *, allow_upper_bound: bool
    ) -> None:
        for component, coordinate in zip(self.components, coordinates):
            if len(coordinate) != len(component.extent):
                raise ValueError(
                    f"DeviceRef component {component.name!r} has rank "
                    f"{len(coordinate)}, expected {len(component.extent)}"
                )
            for axis, (value, extent) in enumerate(zip(coordinate, component.extent)):
                upper_ok = value <= extent if allow_upper_bound else value < extent
                if not upper_ok:
                    relation = "<=" if allow_upper_bound else "<"
                    raise ValueError(
                        f"DeviceRef component {component.name!r} axis {axis} "
                        f"requires 0 <= coord {relation} {extent}, got {value}"
                    )


@dataclass(frozen=True, init=False)
class TransferGraph:
    """Explicit or structured transfer relation over a `DeviceDomain`."""

    domain: DeviceDomain
    transfer_edges: Tuple[TransferEdge, ...]
    structured: Optional[StructuredTransfer]

    def __init__(
        self,
        domain: DeviceDomain,
        *,
        edges: Iterable[TransferEdge] = (),
        structured: Optional[StructuredTransfer] = None,
    ) -> None:
        transfer_edges = tuple(edges)
        if structured is not None and transfer_edges:
            raise ValueError("TransferGraph must be explicit or structured, not both")
        if structured is None and not transfer_edges:
            raise ValueError("TransferGraph.edges requires at least one edge")
        if structured is not None:
            structured = self._normalize_structured(domain, structured)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "transfer_edges", transfer_edges)
        object.__setattr__(self, "structured", structured)

    @classmethod
    def edges(
        cls,
        domain: DeviceDomain,
        edges: Iterable[Tuple[Any, Union[Any, DeviceRange]]],
    ) -> "TransferGraph":
        transfer_edges = []
        for source, destination in edges:
            source_ref = domain.device_ref(source)
            if isinstance(destination, DeviceRange):
                destination_ref = domain.resolve_device_range(destination)
                if cls._range_contains(destination_ref, source_ref):
                    raise ValueError("source-in-destination multicast is not supported")
            else:
                destination_ref = domain.device_ref(destination)
            transfer_edges.append(TransferEdge(source_ref, destination_ref))
        return cls(domain, edges=transfer_edges)

    @classmethod
    def axis_neighbor(
        cls,
        domain: DeviceDomain,
        *,
        component: Optional[str] = None,
        axis: int = 0,
        offset: int = 1,
        wrap: bool = False,
    ) -> "TransferGraph":
        component_name = cls._default_component(domain, component)
        component_info = domain.components[domain.component_index(component_name)]
        _require_int(axis, "axis")
        _require_int(offset, "offset")
        if axis < 0 or axis >= len(component_info.extent):
            raise ValueError(
                f"axis {axis} is out of bounds for component {component_name!r} "
                f"rank {len(component_info.extent)}"
            )
        if offset <= 0:
            raise ValueError(f"offset must be positive, got {offset}")
        if not isinstance(wrap, bool):
            raise TypeError(f"wrap must be a bool, got {wrap!r}")
        return cls(
            domain,
            structured=AxisNeighborTransfer(
                component_name=component_name,
                axis=axis,
                offset=offset,
                wrap=wrap,
            ),
        )

    @classmethod
    def gather(
        cls, domain: DeviceDomain, root: Any, *, component: Optional[str] = None
    ) -> "TransferGraph":
        return cls(
            domain,
            structured=GatherTransfer(
                component_name=cls._default_component(domain, component),
                root=domain.device_ref(root),
            ),
        )

    @classmethod
    def multicast(
        cls, domain: DeviceDomain, source: Any, *, component: Optional[str] = None
    ) -> "TransferGraph":
        return cls(
            domain,
            structured=MulticastTransfer(
                component_name=cls._default_component(domain, component),
                source=domain.device_ref(source),
            ),
        )

    @property
    def is_explicit(self) -> bool:
        return self.structured is None

    @property
    def is_structured(self) -> bool:
        return self.structured is not None

    @property
    def explicit_edge_count(self) -> Optional[int]:
        return len(self.transfer_edges) if self.is_explicit else None

    def metadata_cost(self) -> GraphMetadataCost:
        compile_time = (
            "O(E * (C + R)) explicit user edges"
            if self.is_explicit
            else "O(1 + C + R) structured descriptor"
        )
        return GraphMetadataCost(
            storage_class="compile-time metadata",
            compile_time=compile_time,
            runtime_host="none allocated by TransferGraph",
            device_visible="none allocated by TransferGraph",
        )

    def iter_edges(self) -> Iterator[TransferEdge]:
        """Iterate the transfer relation without changing its stored form."""
        if self.is_explicit:
            yield from self.transfer_edges
            return

        assert self.structured is not None
        component_index = self.domain.component_index(self.structured.component_name)
        devices = tuple(self._iter_domain_devices())

        if isinstance(self.structured, AxisNeighborTransfer):
            component = self.domain.components[component_index]
            axis_extent = component.extent[self.structured.axis]
            for source in devices:
                source_coordinates = [
                    list(coordinates) for coordinates in source.coordinates
                ]
                destination_axis = (
                    source_coordinates[component_index][self.structured.axis]
                    + self.structured.offset
                )
                if destination_axis >= axis_extent:
                    if not self.structured.wrap:
                        continue
                    destination_axis %= axis_extent
                source_coordinates[component_index][
                    self.structured.axis
                ] = destination_axis
                yield TransferEdge(source, DeviceRef(*source_coordinates))
            return

        if isinstance(self.structured, GatherTransfer):
            for source in devices:
                destination_coordinates = list(source.coordinates)
                destination_coordinates[component_index] = (
                    self.structured.root.coordinates[component_index]
                )
                destination = DeviceRef(*destination_coordinates)
                if source != destination:
                    yield TransferEdge(source, destination)
            return

        if isinstance(self.structured, MulticastTransfer):
            for destination in devices:
                source_coordinates = list(destination.coordinates)
                source_coordinates[component_index] = (
                    self.structured.source.coordinates[component_index]
                )
                source = DeviceRef(*source_coordinates)
                if source != destination:
                    yield TransferEdge(source, destination)
            return

        raise TypeError(
            f"unsupported structured transfer type " f"{type(self.structured).__name__}"
        )

    def _iter_domain_devices(self) -> Iterator[DeviceRef]:
        component_coordinates = []
        for component in self.domain.components:
            component_coordinates.append(
                tuple(
                    itertools.product(*(range(extent) for extent in component.extent))
                )
            )
        for coordinates in itertools.product(*component_coordinates):
            yield DeviceRef(*coordinates)

    @staticmethod
    def _default_component(domain: DeviceDomain, component_name: Optional[str]) -> str:
        if component_name is not None:
            domain.component_index(component_name)
            return component_name
        if domain.component_count == 1:
            return domain.components[0].name
        raise ValueError(
            "structured transfers over product domains require an explicit component"
        )

    @staticmethod
    def _normalize_structured(
        domain: DeviceDomain, structured: StructuredTransfer
    ) -> StructuredTransfer:
        if not isinstance(structured, StructuredTransfer):
            raise TypeError("structured transfer must be a StructuredTransfer")
        component_index = domain.component_index(structured.component_name)
        component = domain.components[component_index]

        if isinstance(structured, AxisNeighborTransfer):
            axis = _require_int(structured.axis, "axis")
            offset = _require_int(structured.offset, "offset")
            if axis < 0 or axis >= len(component.extent):
                raise ValueError(
                    f"axis {axis} is out of bounds for component "
                    f"{structured.component_name!r} rank {len(component.extent)}"
                )
            if offset <= 0:
                raise ValueError(f"offset must be positive, got {offset}")
            if not isinstance(structured.wrap, bool):
                raise TypeError(f"wrap must be a bool, got {structured.wrap!r}")
            return structured

        if isinstance(structured, GatherTransfer):
            return GatherTransfer(
                component_name=structured.component_name,
                root=domain.resolve_device_ref(structured.root),
            )

        if isinstance(structured, MulticastTransfer):
            return MulticastTransfer(
                component_name=structured.component_name,
                source=domain.resolve_device_ref(structured.source),
            )

        raise TypeError(
            f"unsupported structured transfer type {type(structured).__name__}"
        )

    @staticmethod
    def _range_contains(device_range: DeviceRange, device_ref: DeviceRef) -> bool:
        return all(
            all(lo <= value < hi for lo, value, hi in zip(lo_coord, coord, hi_coord))
            for lo_coord, coord, hi_coord in zip(
                device_range.lo.coordinates,
                device_ref.coordinates,
                device_range.hi.coordinates,
            )
        )


__all__ = [
    "AxisNeighborTransfer",
    "DeviceDomain",
    "DeviceRange",
    "DeviceRef",
    "DomainComponent",
    "GatherTransfer",
    "GraphMetadataCost",
    "MulticastTransfer",
    "StructuredTransfer",
    "TransferEdge",
    "TransferGraph",
]
