# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Logical device domains and device-level transfer graphs.

This module contains architecture-neutral frontend metadata. Explicit graphs
store O(E) user edges. Axis-neighbor, gather, scatter, and all-to-all graphs
store only their domain and constructor parameters; stencil graphs also store
their offsets. No graph allocates runtime host state or device-visible memory.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
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


def _normalize_stencil_offsets(
    offsets: Iterable[Sequence[int]], rank: int
) -> Tuple[Coordinate, ...]:
    if isinstance(offsets, (str, bytes)):
        raise TypeError("stencil offsets must be an iterable of integer sequences")
    try:
        offset_values = tuple(offsets)
    except TypeError as error:
        raise TypeError(
            "stencil offsets must be an iterable of integer sequences"
        ) from error
    if not offset_values:
        raise ValueError("stencil requires at least one offset")

    normalized = []
    for offset_index, offset in enumerate(offset_values):
        if isinstance(offset, (str, bytes)) or not isinstance(offset, Sequence):
            raise TypeError(
                f"stencil offset {offset_index} must be an integer sequence"
            )
        coordinate = tuple(
            _require_int(value, f"stencil offset {offset_index} component")
            for value in offset
        )
        if len(coordinate) != rank:
            raise ValueError(
                f"stencil offset {offset_index} has rank {len(coordinate)}, "
                f"expected {rank}"
            )
        if all(value == 0 for value in coordinate):
            raise ValueError("stencil offsets must not contain the zero offset")
        normalized.append(coordinate)

    if len(set(normalized)) != len(normalized):
        raise ValueError("stencil offsets must be unique")
    return tuple(normalized)


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


@dataclass(frozen=True, init=False, eq=False)
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceRef):
            return NotImplemented
        return self.coordinates == other.coordinates

    def __hash__(self) -> int:
        return hash(self.coordinates)


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
    """Device-level transfer edge without node or dataflow-buffer binding."""

    source: DeviceRef
    destination: Union[DeviceRef, DeviceRange]


@dataclass(frozen=True)
class StructuredTransfer:
    """Base class for relations stored by constructor parameters, not edge lists."""

    component_name: str


@dataclass(frozen=True)
class AxisNeighborTransfer(StructuredTransfer):
    """Transfer relation between neighbors along one domain axis."""

    axis: int
    offset: int
    wrap: bool


@dataclass(frozen=True)
class StencilTransfer(StructuredTransfer):
    """Union of directed translations within one domain component."""

    offsets: Iterable[Sequence[int]]
    wrap: bool


@dataclass(frozen=True)
class GatherTransfer(StructuredTransfer):
    """Per-slice transfer relation from every device to one component root."""

    root: Any


@dataclass(frozen=True)
class ScatterTransfer(StructuredTransfer):
    """Per-slice transfer relation from one component source to every device."""

    source: Any


@dataclass(frozen=True)
class AllToAllTransfer(StructuredTransfer):
    """Transfer relation between every ordered pair of distinct devices."""


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

    def __getitem__(self, index: Any) -> DeviceView | DevicePoint:
        return DeviceView(
            self, tuple(range(extent) for extent in self.flattened_extent)
        )[index]

    def at_node(self, node_x: int, node_y: int) -> NodeSelection:
        return self[:].at_node(node_x, node_y)

    def select(self, points: Iterable[DevicePoint]) -> DeviceSet:
        """Return the selected devices once each in parent-domain row-major order."""
        references = []
        for point in points:
            if not isinstance(point, DevicePoint):
                raise TypeError("selected points must be DevicePoint values")
            if point.domain != self:
                raise ValueError("selected points must belong to the parent domain")
            references.append(point.reference)
        return DeviceSet(self, tuple(references))

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def flattened_extent(self) -> Coordinate:
        """Return component extents in runtime mesh-coordinate order."""
        return tuple(
            dimension for component in self.components for dimension in component.extent
        )

    def iter_device_refs(self) -> Iterator[DeviceRef]:
        """Iterate every device in component-major, row-major order."""
        component_coordinates = [
            tuple(itertools.product(*(range(extent) for extent in component.extent)))
            for component in self.components
        ]
        for coordinates in itertools.product(*component_coordinates):
            yield DeviceRef(*coordinates)

    def flattened_coordinates(self, device: Any) -> Coordinate:
        """Return one device coordinate in runtime mesh-coordinate order."""
        device_ref = self.device_ref(device)
        return tuple(
            coordinate
            for component_coordinates in device_ref.coordinates
            for coordinate in component_coordinates
        )

    def _operation_identity_capture(self) -> tuple:
        return (
            "device-domain",
            tuple(
                (component.name, tuple(component.extent))
                for component in self.components
            ),
        )

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

    def is_current(self, device: Any) -> bool:
        """Return whether the kernel executes on `device` in this domain.

        The compiler replaces this predicate with target-independent logical
        device-coordinate comparisons. Calling it outside a TTL kernel is an
        error.
        """
        self.device_ref(device)
        raise RuntimeError(
            "DeviceDomain.is_current() should only be called inside a TTL kernel"
        )

    def current_index(self) -> int:
        """Return the current logical device's zero-based row-major order.

        The compiler replaces this call with target-independent logical
        device-coordinate indexing. Calling it outside a TTL kernel is an
        error.
        """
        raise RuntimeError(
            "DeviceDomain.current_index() should only be called inside a TTL kernel"
        )

    def index_order(self, device: Any) -> int:
        """Return a device's zero-based row-major order in this domain."""
        device_ref = self.device_ref(device)
        order = 0
        for component, coordinates in zip(self.components, device_ref.coordinates):
            for coordinate, extent in zip(coordinates, component.extent):
                order = order * extent + coordinate
        return order

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
            self._validate_component_coordinate(
                component,
                coordinate,
                allow_upper_bound=allow_upper_bound,
                context="DeviceRef",
            )

    @staticmethod
    def _validate_component_coordinate(
        component: DomainComponent,
        coordinate: Coordinate,
        *,
        allow_upper_bound: bool,
        context: str,
    ) -> None:
        if len(coordinate) != len(component.extent):
            raise ValueError(
                f"{context} component {component.name!r} has rank "
                f"{len(coordinate)}, expected {len(component.extent)}"
            )
        for axis, (value, extent) in enumerate(zip(coordinate, component.extent)):
            upper_ok = value <= extent if allow_upper_bound else value < extent
            if not upper_ok:
                relation = "<=" if allow_upper_bound else "<"
                raise ValueError(
                    f"{context} component {component.name!r} axis {axis} "
                    f"requires 0 <= coord {relation} {extent}, got {value}"
                )


class DeviceSelection(ABC):
    """Device membership with coordinates resolved in one parent domain."""

    domain: DeviceDomain

    @abstractmethod
    def iter_device_refs(self) -> Iterator[DeviceRef]:
        """Iterate selected parent-domain references in deterministic order."""

    @abstractmethod
    def _operation_identity_capture(self) -> tuple:
        """Return the immutable selection contract used for compilation identity."""

    def at_node(self, node_x: int, node_y: int) -> NodeSelection:
        return NodeSelection(self, (node_x, node_y))

    def __or__(self, other: DeviceSelection) -> DeviceSet:
        if not isinstance(other, DeviceSelection):
            raise TypeError(
                "device selections can be combined only with another selection"
            )
        if self.domain != other.domain:
            raise ValueError("device selections must share a parent domain")
        return DeviceSet(
            self.domain,
            tuple(itertools.chain(self.iter_device_refs(), other.iter_device_refs())),
        )


@dataclass(frozen=True)
class DevicePoint(DeviceSelection):
    """One device reference together with its parent domain."""

    domain: DeviceDomain
    reference: DeviceRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference", self.domain.resolve_device_ref(self.reference)
        )

    def iter_device_refs(self) -> Iterator[DeviceRef]:
        yield self.reference

    def _operation_identity_capture(self) -> tuple:
        return (
            "device-point",
            self.domain._operation_identity_capture(),
            self.reference.coordinates,
        )


@dataclass(frozen=True)
class DeviceView(DeviceSelection):
    """An indexed device view stored as one integer or range per parent axis.

    An integer selects one parent coordinate and removes that axis from the
    view. A range preserves the axis for later indexing. Nested slicing updates
    these selections without enumerating devices.
    """

    domain: DeviceDomain
    axes: Tuple[int | range, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if len(axes) != len(self.domain.flattened_extent):
            raise ValueError("device view must specify every parent axis")
        for axis, extent in zip(axes, self.domain.flattened_extent):
            if isinstance(axis, range):
                if axis and (
                    min(axis[0], axis[-1]) < 0 or max(axis[0], axis[-1]) >= extent
                ):
                    raise ValueError("device view range is outside the parent domain")
            else:
                coordinate = _require_int(axis, "device view coordinate")
                if not 0 <= coordinate < extent:
                    raise ValueError(
                        "device view coordinate is outside the parent domain"
                    )
        object.__setattr__(self, "axes", axes)

    @property
    def shape(self) -> Coordinate:
        return tuple(len(axis) for axis in self.axes if isinstance(axis, range))

    def __getitem__(self, index: Any) -> DeviceView | DevicePoint:
        indices = index if isinstance(index, tuple) else (index,)
        rank = len(self.shape)
        if len(indices) > rank:
            raise IndexError(f"device view has rank {rank}, got {len(indices)} indices")
        indices = iter(indices + (slice(None),) * (rank - len(indices)))
        axes = []
        for axis in self.axes:
            if isinstance(axis, range):
                selection = next(indices)
                if not isinstance(selection, slice):
                    _require_int(selection, "device index")
                axis = axis[selection]
            axes.append(axis)
        if any(isinstance(axis, range) for axis in axes):
            return DeviceView(self.domain, tuple(axes))
        return DevicePoint(self.domain, self._parent_reference(tuple(axes)))

    def _parent_reference(self, coordinates: Coordinate) -> DeviceRef:
        components = []
        offset = 0
        for component in self.domain.components:
            component_rank = len(component.extent)
            components.append(coordinates[offset : offset + component_rank])
            offset += component_rank
        return DeviceRef(*components)

    def iter_device_refs(self) -> Iterator[DeviceRef]:
        for coordinates in itertools.product(
            *(axis if isinstance(axis, range) else (axis,) for axis in self.axes)
        ):
            yield self._parent_reference(coordinates)

    def _operation_identity_capture(self) -> tuple:
        return (
            "device-view",
            self.domain._operation_identity_capture(),
            tuple(
                (axis.start, axis.stop, axis.step) if isinstance(axis, range) else axis
                for axis in self.axes
            ),
        )


@dataclass(frozen=True)
class DeviceSet(DeviceSelection):
    """Selected devices stored once each in parent-domain row-major order."""

    domain: DeviceDomain
    references: Tuple[DeviceRef, ...]

    def __post_init__(self) -> None:
        references = {
            self.domain.resolve_device_ref(reference) for reference in self.references
        }
        object.__setattr__(
            self, "references", tuple(sorted(references, key=self.domain.index_order))
        )

    def iter_device_refs(self) -> Iterator[DeviceRef]:
        yield from self.references

    def _operation_identity_capture(self) -> tuple:
        return (
            "device-set",
            self.domain._operation_identity_capture(),
            tuple(reference.coordinates for reference in self.references),
        )


@dataclass(frozen=True)
class NodeSelection:
    """A logical Tensix node on each selected device.

    Coordinates are nonnegative. Operation placement must additionally verify
    that this node exists on every selected device before dispatch.
    """

    devices: DeviceSelection
    node: Coordinate

    def __post_init__(self) -> None:
        if not isinstance(self.devices, DeviceSelection):
            raise TypeError("node selection requires a device selection")
        node = _normalize_coordinate(self.node, "node coordinate")
        if len(node) != 2:
            raise ValueError("node coordinate must have rank two")
        object.__setattr__(self, "node", node)

    def _operation_identity_capture(self) -> tuple:
        return ("node-selection", self.devices._operation_identity_capture(), self.node)


@dataclass(frozen=True, init=False)
class TransferGraph:
    """Device transfers stored as explicit edges or constructor parameters."""

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
        input_edges = tuple(edges)
        if structured is not None and input_edges:
            raise ValueError("TransferGraph must be explicit or structured, not both")
        if structured is None and not input_edges:
            raise ValueError("TransferGraph.edges requires at least one edge")
        if structured is not None:
            structured = self._normalize_structured(domain, structured)
            if not self._structured_has_edges(domain, structured):
                raise ValueError("structured transfer relation contains no edges")
            transfer_edges = ()
        else:
            transfer_edges = tuple(
                self._normalize_edge(domain, edge, edge_index)
                for edge_index, edge in enumerate(input_edges)
            )
            if len(set(transfer_edges)) != len(transfer_edges):
                raise ValueError("TransferGraph edges must be unique")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "transfer_edges", transfer_edges)
        object.__setattr__(self, "structured", structured)

    @classmethod
    def edges(
        cls,
        domain: DeviceDomain,
        edges: Iterable[Tuple[Any, Union[Any, DeviceRange]]],
    ) -> "TransferGraph":
        return cls(
            domain,
            edges=(
                TransferEdge(source=source, destination=destination)
                for source, destination in edges
            ),
        )

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
        component_name = cls._default_component(domain, component)
        return cls(
            domain,
            structured=GatherTransfer(component_name=component_name, root=root),
        )

    @classmethod
    def stencil(
        cls,
        domain: DeviceDomain,
        *,
        offsets: Iterable[Sequence[int]],
        component: Optional[str] = None,
        wrap: bool = False,
    ) -> "TransferGraph":
        component_name = cls._default_component(domain, component)
        return cls(
            domain,
            structured=StencilTransfer(
                component_name=component_name,
                offsets=offsets,
                wrap=wrap,
            ),
        )

    @classmethod
    def scatter(
        cls, domain: DeviceDomain, source: Any, *, component: Optional[str] = None
    ) -> "TransferGraph":
        component_name = cls._default_component(domain, component)
        return cls(
            domain,
            structured=ScatterTransfer(component_name=component_name, source=source),
        )

    @classmethod
    def all_to_all(
        cls, domain: DeviceDomain, *, component: Optional[str] = None
    ) -> "TransferGraph":
        return cls(
            domain,
            structured=AllToAllTransfer(
                component_name=cls._default_component(domain, component),
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
        if self.is_explicit:
            compile_time = "O(E * (C + R)) explicit user edges"
        elif isinstance(self.structured, StencilTransfer):
            compile_time = "O(K + C + R) stencil offsets and domain parameters"
        else:
            compile_time = "O(1 + C + R) domain and relation parameters"
        return GraphMetadataCost(
            storage_class="compile-time metadata",
            compile_time=compile_time,
            runtime_host="none allocated by TransferGraph",
            device_visible="none allocated by TransferGraph",
        )

    def _operation_identity_capture(self) -> tuple:
        """Return the stored graph descriptor used by operation caching."""

        def device_identity(device: DeviceRef) -> tuple:
            return tuple(tuple(coordinates) for coordinates in device.coordinates)

        if self.is_explicit:
            relation_identity = (
                "explicit",
                tuple(
                    (
                        device_identity(edge.source),
                        (
                            (
                                "range",
                                device_identity(edge.destination.lo),
                                device_identity(edge.destination.hi),
                            )
                            if isinstance(edge.destination, DeviceRange)
                            else ("device", device_identity(edge.destination))
                        ),
                    )
                    for edge in self.transfer_edges
                ),
            )
        else:
            structured = self.structured
            assert structured is not None
            if isinstance(structured, AxisNeighborTransfer):
                relation_identity = (
                    "axis-neighbor",
                    structured.component_name,
                    structured.axis,
                    structured.offset,
                    structured.wrap,
                )
            elif isinstance(structured, StencilTransfer):
                relation_identity = (
                    "stencil",
                    structured.component_name,
                    tuple(structured.offsets),
                    structured.wrap,
                )
            elif isinstance(structured, GatherTransfer):
                relation_identity = (
                    "gather",
                    structured.component_name,
                    device_identity(structured.root),
                )
            elif isinstance(structured, ScatterTransfer):
                relation_identity = (
                    "scatter",
                    structured.component_name,
                    device_identity(structured.source),
                )
            elif isinstance(structured, AllToAllTransfer):
                relation_identity = ("all-to-all", structured.component_name)
            else:
                raise TypeError(
                    f"unsupported structured transfer type "
                    f"{type(structured).__name__}"
                )

        return (
            "transfer-graph",
            self.domain._operation_identity_capture(),
            relation_identity,
        )

    def device_endpoints(self) -> frozenset[DeviceRef]:
        """Return devices used as a source or destination by an edge."""
        structured = self.structured
        if isinstance(structured, (GatherTransfer, ScatterTransfer, AllToAllTransfer)):
            return frozenset(self.domain.iter_device_refs())
        if isinstance(structured, AxisNeighborTransfer) and structured.wrap:
            return frozenset(self.domain.iter_device_refs())
        if isinstance(structured, StencilTransfer) and structured.wrap:
            return frozenset(self.domain.iter_device_refs())

        endpoints = set()
        for edge in self.iter_edges():
            endpoints.add(edge.source)
            if isinstance(edge.destination, DeviceRange):
                endpoints.update(
                    device
                    for device in self.domain.iter_device_refs()
                    if self._range_contains(edge.destination, device)
                )
            else:
                endpoints.add(edge.destination)
        return frozenset(endpoints)

    def iter_edges(self) -> Iterator[TransferEdge]:
        """Iterate the transfer relation without changing its stored form."""
        if self.is_explicit:
            yield from self.transfer_edges
            return

        assert self.structured is not None
        component_index = self.domain.component_index(self.structured.component_name)
        devices = tuple(self.domain.iter_device_refs())

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

        if isinstance(self.structured, StencilTransfer):
            component = self.domain.components[component_index]
            emitted_edges = set()
            for source in devices:
                for offset in self.structured.offsets:
                    destination_coordinates = [
                        list(coordinates) for coordinates in source.coordinates
                    ]
                    destination_component = destination_coordinates[component_index]
                    in_bounds = True
                    for axis, (coordinate, delta, extent) in enumerate(
                        zip(destination_component, offset, component.extent)
                    ):
                        translated = coordinate + delta
                        if self.structured.wrap:
                            translated %= extent
                        elif translated < 0 or translated >= extent:
                            in_bounds = False
                            break
                        destination_component[axis] = translated
                    if not in_bounds:
                        continue
                    destination = DeviceRef(*destination_coordinates)
                    edge = TransferEdge(source, destination)
                    if source == destination or edge in emitted_edges:
                        continue
                    emitted_edges.add(edge)
                    yield edge
            return

        if isinstance(self.structured, GatherTransfer):
            for source in devices:
                destination_coordinates = list(source.coordinates)
                destination_coordinates[component_index] = (
                    self.structured.root.coordinates[0]
                )
                destination = DeviceRef(*destination_coordinates)
                if source != destination:
                    yield TransferEdge(source, destination)
            return

        if isinstance(self.structured, ScatterTransfer):
            for destination in devices:
                source_coordinates = list(destination.coordinates)
                source_coordinates[component_index] = (
                    self.structured.source.coordinates[0]
                )
                source = DeviceRef(*source_coordinates)
                if source != destination:
                    yield TransferEdge(source, destination)
            return

        if isinstance(self.structured, AllToAllTransfer):
            component = self.domain.components[component_index]
            component_coordinates = itertools.product(
                *(range(extent) for extent in component.extent)
            )
            destination_components = tuple(component_coordinates)
            for source in devices:
                for destination_component in destination_components:
                    if destination_component == source.coordinates[component_index]:
                        continue
                    destination_coordinates = list(source.coordinates)
                    destination_coordinates[component_index] = destination_component
                    yield TransferEdge(source, DeviceRef(*destination_coordinates))
            return

        raise TypeError(
            f"unsupported structured transfer type " f"{type(self.structured).__name__}"
        )

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
    def _resolve_component_endpoint(
        domain: DeviceDomain,
        component_name: str,
        endpoint: Any,
        context: str,
    ) -> DeviceRef:
        component = domain.components[domain.component_index(component_name)]
        if isinstance(endpoint, DeviceRef):
            if endpoint.is_named:
                if endpoint.component_names != (component_name,):
                    raise ValueError(
                        f"{context} must name only component {component_name!r}"
                    )
                coordinates = endpoint.coordinates[0]
            else:
                if len(endpoint.coordinates) != 1:
                    raise ValueError(
                        f"{context} must provide one coordinate for component "
                        f"{component_name!r}"
                    )
                coordinates = endpoint.coordinates[0]
        else:
            coordinates = _normalize_coordinate(endpoint, context)

        domain._validate_component_coordinate(
            component,
            coordinates,
            allow_upper_bound=False,
            context=context,
        )
        return DeviceRef(coordinates)

    @staticmethod
    def _normalize_edge(
        domain: DeviceDomain, edge: TransferEdge, edge_index: int
    ) -> TransferEdge:
        if not isinstance(edge, TransferEdge):
            raise TypeError(
                f"TransferGraph edge {edge_index} must be a TransferEdge, "
                f"got {type(edge).__name__}"
            )
        source = domain.device_ref(edge.source)
        if isinstance(edge.destination, DeviceRange):
            destination = domain.resolve_device_range(edge.destination)
            if TransferGraph._range_contains(destination, source):
                raise ValueError(
                    "source must not be contained in its destination range"
                )
        else:
            destination = domain.device_ref(edge.destination)
        return TransferEdge(source, destination)

    @staticmethod
    def _structured_has_edges(
        domain: DeviceDomain, structured: StructuredTransfer
    ) -> bool:
        component = domain.components[domain.component_index(structured.component_name)]
        if isinstance(structured, AxisNeighborTransfer):
            extent = component.extent[structured.axis]
            if structured.wrap:
                return structured.offset % extent != 0
            return structured.offset < extent
        if isinstance(structured, StencilTransfer):
            for offset in structured.offsets:
                if structured.wrap:
                    if any(
                        delta % extent != 0
                        for delta, extent in zip(offset, component.extent)
                    ):
                        return True
                elif all(
                    abs(delta) < extent
                    for delta, extent in zip(offset, component.extent)
                ):
                    return True
            return False
        return prod(component.extent) > 1

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

        if isinstance(structured, StencilTransfer):
            if not isinstance(structured.wrap, bool):
                raise TypeError(f"wrap must be a bool, got {structured.wrap!r}")
            offsets = _normalize_stencil_offsets(
                structured.offsets, len(component.extent)
            )
            if structured.wrap:
                effective_offsets = []
                seen_offsets = set()
                for offset in offsets:
                    effective_offset = tuple(
                        delta % extent
                        for delta, extent in zip(offset, component.extent)
                    )
                    if not any(effective_offset) or effective_offset in seen_offsets:
                        continue
                    seen_offsets.add(effective_offset)
                    effective_offsets.append(effective_offset)
                offsets = tuple(effective_offsets)
            return StencilTransfer(
                component_name=structured.component_name,
                offsets=offsets,
                wrap=structured.wrap,
            )

        if isinstance(structured, GatherTransfer):
            return GatherTransfer(
                component_name=structured.component_name,
                root=TransferGraph._resolve_component_endpoint(
                    domain,
                    structured.component_name,
                    structured.root,
                    "gather root",
                ),
            )

        if isinstance(structured, ScatterTransfer):
            return ScatterTransfer(
                component_name=structured.component_name,
                source=TransferGraph._resolve_component_endpoint(
                    domain,
                    structured.component_name,
                    structured.source,
                    "scatter source",
                ),
            )

        if isinstance(structured, AllToAllTransfer):
            return structured

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
    "AllToAllTransfer",
    "AxisNeighborTransfer",
    "DeviceDomain",
    "DevicePoint",
    "DeviceRange",
    "DeviceRef",
    "DeviceSelection",
    "DeviceSet",
    "DeviceView",
    "DomainComponent",
    "GatherTransfer",
    "GraphMetadataCost",
    "NodeSelection",
    "ScatterTransfer",
    "StencilTransfer",
    "StructuredTransfer",
    "TransferEdge",
    "TransferGraph",
]
