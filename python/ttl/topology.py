# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Logical device domains and device-level transfer graphs.

This module is frontend metadata. It does not allocate runtime host state or
device-visible communication memory. Explicit graphs store O(E) user edges.
Structured graphs store O(1) descriptors and are not expanded unless a caller
explicitly asks for bounded diagnostics in a later pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union


Coordinate = Tuple[int, ...]
LevelCoordinates = Tuple[Coordinate, ...]


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
    for axis, dim in enumerate(normalized):
        if dim <= 0:
            raise ValueError(f"{context} axis {axis} must be positive, got {dim}")
    return normalized


def _normalize_coordinate(value: Any, context: str) -> Coordinate:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{context} must be an integer or integer sequence, got {value!r}"
        )
    coord = tuple(
        _require_int(component, f"{context} component") for component in value
    )
    if not coord:
        raise ValueError(f"{context} must have at least one component")
    for axis, component in enumerate(coord):
        if component < 0:
            raise ValueError(
                f"{context} axis {axis} must be non-negative, got {component}"
            )
    return coord


@dataclass(frozen=True)
class FabricTopology:
    """Topology metadata for one `TopologyLevelInfo`.

    Storage class: compile-time metadata only.
    Cost: O(1) per topology level; no domain-size, edge-count, PipeNet-count,
    or device-visible memory allocation.
    """

    kind: str
    cluster_axis: int = 0
    periodic: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("fabric_1d", "fabric_ring"):
            raise ValueError(f"unsupported fabric topology kind {self.kind!r}")
        _require_int(self.cluster_axis, "cluster_axis")
        if self.cluster_axis < 0:
            raise ValueError(
                f"cluster_axis must be non-negative, got {self.cluster_axis}"
            )
        if not isinstance(self.periodic, bool):
            raise TypeError(f"periodic must be a bool, got {self.periodic!r}")
        if self.kind == "fabric_ring" and not self.periodic:
            raise ValueError("fabric_ring topology requires periodic=True")
        if self.kind == "fabric_1d" and self.periodic:
            raise ValueError("fabric_1d topology requires periodic=False")

    def validate_rank(self, rank: int, context: str) -> None:
        if self.cluster_axis >= rank:
            raise ValueError(
                f"{context} cluster_axis {self.cluster_axis} exceeds rank {rank}"
            )


def Fabric1D(axis: int = 0) -> FabricTopology:
    """Return non-periodic 1D fabric topology metadata."""

    return FabricTopology(kind="fabric_1d", cluster_axis=axis, periodic=False)


def FabricRing(axis: int = 0) -> FabricTopology:
    """Return periodic 1D fabric topology metadata."""

    return FabricTopology(kind="fabric_ring", cluster_axis=axis, periodic=True)


@dataclass(frozen=True)
class TopologyLevelInfo:
    """Named level in a logical device hierarchy.

    Storage class: compile-time metadata only.
    Cost: O(1) per hierarchy level plus O(R) extent storage for rank R; no
    domain-size, edge-count, PipeNet-count, or device-visible memory allocation.
    """

    name: str
    extent: Sequence[int]
    topology: Optional[FabricTopology] = None
    mesh_id: int = 0
    routing_metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("topology level name must not be empty")
        extent = _normalize_extent(self.extent, f"topology level {self.name!r} extent")
        object.__setattr__(self, "extent", extent)
        if self.topology is not None:
            self.topology.validate_rank(len(extent), f"topology level {self.name!r}")
        _require_int(self.mesh_id, "mesh_id")
        if self.mesh_id < 0:
            raise ValueError(f"mesh_id must be non-negative, got {self.mesh_id}")
        if self.routing_metadata is None:
            metadata_items: Tuple[Tuple[str, Any], ...] = ()
        else:
            metadata_items = tuple(sorted(self.routing_metadata.items()))
        object.__setattr__(self, "routing_metadata", metadata_items)


@dataclass(frozen=True, init=False)
class DeviceRef:
    """Leaf-device coordinate in a `DeviceDomain`.

    Storage class: compile-time metadata only.
    Cost: O(L + R) coordinate storage for L hierarchy levels and total rank R;
    no domain-size, edge-count, PipeNet-count, or device-visible memory
    allocation.
    """

    coordinates: LevelCoordinates
    level_names: Tuple[str, ...]

    def __init__(self, *coordinates: Any, **named_coordinates: Any) -> None:
        if coordinates and named_coordinates:
            raise ValueError(
                "DeviceRef accepts positional coordinates or named coordinates, not both"
            )
        if named_coordinates:
            level_names = tuple(named_coordinates.keys())
            normalized = tuple(
                _normalize_coordinate(coord, f"DeviceRef {name!r}")
                for name, coord in named_coordinates.items()
            )
        else:
            if not coordinates:
                raise ValueError("DeviceRef requires at least one coordinate")
            level_names = ()
            normalized = tuple(
                _normalize_coordinate(coord, "DeviceRef coordinate")
                for coord in coordinates
            )
        object.__setattr__(self, "coordinates", normalized)
        object.__setattr__(self, "level_names", level_names)

    @property
    def is_named(self) -> bool:
        return bool(self.level_names)


@dataclass(frozen=True)
class DeviceRange:
    """Half-open device range over a `DeviceDomain`.

    Storage class: compile-time metadata only.
    Cost: O(L + R) coordinate storage for L hierarchy levels and total rank R;
    no dense domain-size or device-visible memory allocation.
    """

    lo: DeviceRef
    hi: DeviceRef


@dataclass(frozen=True)
class TransferEdge:
    """Device-level transfer edge.

    Core coordinates and DFB operands are intentionally absent. They bind at
    the PipeNet or transfer site, then lowering consumes resolved
    (device, core, DFB) endpoints.

    Storage class: compile-time metadata only.
    Cost: O(L + R) endpoint storage; explicit graphs store O(E) such edges.
    Runtime/device communication state must be allocated from local live degree
    and queue depth by later lowering, not from total domain size or E.
    """

    source: DeviceRef
    destination: Union[DeviceRef, DeviceRange]


@dataclass(frozen=True)
class StructuredTransfer:
    """Compact descriptor for a regular transfer family.

    Storage class: compile-time metadata only.
    Cost: O(1) plus O(L + R) for optional source/root coordinates; no edge list
    materialization and no device-visible memory allocation.
    """

    kind: str
    level_name: Optional[str] = None
    axis: Optional[int] = None
    offset: Optional[int] = None
    wrap: Optional[bool] = None
    source: Optional[DeviceRef] = None
    root: Optional[DeviceRef] = None


@dataclass(frozen=True)
class ProjectedTransfer:
    """Single-level projection for an explicit device-level edge."""

    edge: TransferEdge
    level_index: Optional[int]
    level_name: Optional[str]
    topology: Optional[FabricTopology]
    source_level_coordinate: Optional[Coordinate]
    destination_level_coordinate: Optional[Union[Coordinate, DeviceRange]]

    @property
    def is_local(self) -> bool:
        return self.level_index is None


@dataclass(frozen=True)
class ProjectedTransferFamily:
    """Single-level projection for a structured transfer family."""

    graph: "TransferGraph"
    level_index: int
    level_name: str
    topology: FabricTopology


@dataclass(frozen=True)
class GraphMetadataCost:
    """Asymptotic storage summary for a `TransferGraph`."""

    storage_class: str
    compile_time: str
    runtime_host: str
    device_visible: str


@dataclass(frozen=True, init=False)
class DeviceDomain:
    """Flat or hierarchical logical device domain.

    Storage class: compile-time metadata only.
    Cost: O(L + R) for L hierarchy levels and total rank R; no dense domain,
    adjacency, role, or PipeNet table is allocated here.
    """

    levels: Tuple[TopologyLevelInfo, ...]

    def __init__(
        self,
        extent: Sequence[int],
        topology: Optional[FabricTopology] = None,
        *,
        name: str = "device",
        mesh_id: int = 0,
        routing_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        level = TopologyLevelInfo(
            name=name,
            extent=extent,
            topology=topology,
            mesh_id=mesh_id,
            routing_metadata=routing_metadata,
        )
        object.__setattr__(self, "levels", (level,))
        self._validate_levels()

    @classmethod
    def hierarchy(cls, *levels: TopologyLevelInfo) -> "DeviceDomain":
        if not levels:
            raise ValueError("DeviceDomain.hierarchy requires at least one level")
        domain = cls.__new__(cls)
        object.__setattr__(domain, "levels", tuple(levels))
        domain._validate_levels()
        return domain

    def __post_init__(self) -> None:
        self._validate_levels()

    def _validate_levels(self) -> None:
        names = [level.name for level in self.levels]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"duplicate topology level names: {duplicate_names}")

    @property
    def level_names(self) -> Tuple[str, ...]:
        return tuple(level.name for level in self.levels)

    @property
    def level_count(self) -> int:
        return len(self.levels)

    def level_index(self, level_name: str) -> int:
        for index, level in enumerate(self.levels):
            if level.name == level_name:
                return index
        raise ValueError(f"unknown topology level {level_name!r}")

    def device_ref(self, value: Any) -> DeviceRef:
        if isinstance(value, DeviceRef):
            return self.resolve_device_ref(value)
        if self.level_count != 1:
            raise ValueError(
                "hierarchical domains require DeviceRef values or named coordinates"
            )
        return self.resolve_device_ref(DeviceRef(value))

    def resolve_device_ref(self, device_ref: DeviceRef) -> DeviceRef:
        coordinates = self._coordinates_in_level_order(device_ref)
        self._validate_level_coordinates(coordinates, allow_upper_bound=False)
        return DeviceRef(*coordinates)

    def resolve_device_range(self, device_range: DeviceRange) -> DeviceRange:
        lo_coordinates = self._coordinates_in_level_order(device_range.lo)
        hi_coordinates = self._coordinates_in_level_order(device_range.hi)
        self._validate_range_coordinates(lo_coordinates, hi_coordinates)
        return DeviceRange(DeviceRef(*lo_coordinates), DeviceRef(*hi_coordinates))

    def differing_levels(
        self, source: DeviceRef, destination: Union[DeviceRef, DeviceRange]
    ) -> Tuple[int, ...]:
        source = self.resolve_device_ref(source)
        if isinstance(destination, DeviceRange):
            destination = self.resolve_device_range(destination)
            return self._range_differing_levels(source, destination)
        destination = self.resolve_device_ref(destination)
        return tuple(
            index
            for index, (source_coord, destination_coord) in enumerate(
                zip(source.coordinates, destination.coordinates)
            )
            if source_coord != destination_coord
        )

    def _coordinates_in_level_order(self, device_ref: DeviceRef) -> LevelCoordinates:
        if device_ref.is_named:
            expected_names = self.level_names
            if set(device_ref.level_names) != set(expected_names):
                raise ValueError(
                    f"DeviceRef names {device_ref.level_names} do not match "
                    f"domain levels {expected_names}"
                )
            by_name = dict(zip(device_ref.level_names, device_ref.coordinates))
            return tuple(by_name[name] for name in expected_names)

        if len(device_ref.coordinates) != self.level_count:
            raise ValueError(
                f"DeviceRef has {len(device_ref.coordinates)} level coordinates, "
                f"domain has {self.level_count}"
            )
        return device_ref.coordinates

    def _validate_level_coordinates(
        self, coordinates: LevelCoordinates, *, allow_upper_bound: bool
    ) -> None:
        for level, coord in zip(self.levels, coordinates):
            if len(coord) != len(level.extent):
                raise ValueError(
                    f"DeviceRef level {level.name!r} has rank {len(coord)}, "
                    f"expected {len(level.extent)}"
                )
            for axis, (component, extent) in enumerate(zip(coord, level.extent)):
                upper_ok = (
                    component <= extent if allow_upper_bound else component < extent
                )
                if not upper_ok:
                    relation = "<=" if allow_upper_bound else "<"
                    raise ValueError(
                        f"DeviceRef level {level.name!r} axis {axis} requires "
                        f"0 <= coord {relation} {extent}, got {component}"
                    )

    def _validate_range_coordinates(
        self, lo_coordinates: LevelCoordinates, hi_coordinates: LevelCoordinates
    ) -> None:
        self._validate_level_coordinates(lo_coordinates, allow_upper_bound=False)
        self._validate_level_coordinates(hi_coordinates, allow_upper_bound=True)
        for level, lo_coord, hi_coord in zip(
            self.levels, lo_coordinates, hi_coordinates
        ):
            if len(lo_coord) != len(hi_coord):
                raise ValueError(
                    f"DeviceRange level {level.name!r} rank mismatch: "
                    f"{lo_coord} vs {hi_coord}"
                )
            for axis, (lo_component, hi_component) in enumerate(
                zip(lo_coord, hi_coord)
            ):
                if lo_component >= hi_component:
                    raise ValueError(
                        f"DeviceRange level {level.name!r} axis {axis} requires "
                        f"lo < hi, got lo={lo_component}, hi={hi_component}"
                    )

    def _range_differing_levels(
        self, source: DeviceRef, destination_range: DeviceRange
    ) -> Tuple[int, ...]:
        differing = []
        for index, (source_coord, lo_coord, hi_coord) in enumerate(
            zip(
                source.coordinates,
                destination_range.lo.coordinates,
                destination_range.hi.coordinates,
            )
        ):
            point_range = tuple(component + 1 for component in lo_coord)
            if lo_coord != source_coord or hi_coord != point_range:
                differing.append(index)
        return tuple(differing)


@dataclass(frozen=True, init=False)
class TransferGraph:
    """Device-level transfer graph over a `DeviceDomain`.

    Storage class: compile-time metadata only.
    Cost: explicit graphs store O(E) user edges; structured graphs store O(1)
    descriptors. No runtime host or device-visible communication state is
    allocated here. Later lowering must allocate only local live sender,
    receiver, and queue-depth state.
    """

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
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "transfer_edges", transfer_edges)
        object.__setattr__(self, "structured", structured)
        self._validate()

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
            else:
                destination_ref = domain.device_ref(destination)
            transfer_edges.append(
                TransferEdge(source=source_ref, destination=destination_ref)
            )
        return cls(domain, edges=transfer_edges)

    @classmethod
    def axis_neighbor(
        cls,
        domain: DeviceDomain,
        *,
        level: Optional[str] = None,
        axis: Optional[int] = None,
        offset: int = 1,
        wrap: bool = False,
    ) -> "TransferGraph":
        _require_int(offset, "offset")
        if offset <= 0:
            raise ValueError(f"offset must be positive, got {offset}")
        level_name = cls._default_structured_level(domain, level)
        level_info = domain.levels[domain.level_index(level_name)]
        topology = cls._require_level_topology(level_info)
        if axis is None:
            axis = topology.cluster_axis
        _require_int(axis, "axis")
        if axis < 0 or axis >= len(level_info.extent):
            raise ValueError(
                f"axis {axis} is out of bounds for level {level_name!r} "
                f"rank {len(level_info.extent)}"
            )
        if axis != topology.cluster_axis:
            raise ValueError(
                f"initial 1D routing on level {level_name!r} uses "
                f"cluster_axis {topology.cluster_axis}, got axis {axis}"
            )
        structured = StructuredTransfer(
            kind="axis_neighbor",
            level_name=level_name,
            axis=axis,
            offset=offset,
            wrap=wrap,
        )
        return cls(domain, structured=structured)

    @classmethod
    def gather(
        cls, domain: DeviceDomain, root: Any, *, level: Optional[str] = None
    ) -> "TransferGraph":
        level_name = cls._default_structured_level(domain, level)
        root_ref = domain.device_ref(root)
        structured = StructuredTransfer(
            kind="gather", level_name=level_name, root=root_ref
        )
        return cls(domain, structured=structured)

    @classmethod
    def multicast(
        cls, domain: DeviceDomain, source: Any, *, level: Optional[str] = None
    ) -> "TransferGraph":
        level_name = cls._default_structured_level(domain, level)
        source_ref = domain.device_ref(source)
        structured = StructuredTransfer(
            kind="multicast", level_name=level_name, source=source_ref
        )
        return cls(domain, structured=structured)

    @property
    def is_explicit(self) -> bool:
        return self.structured is None

    @property
    def is_structured(self) -> bool:
        return self.structured is not None

    @property
    def explicit_edge_count(self) -> Optional[int]:
        if not self.is_explicit:
            return None
        return len(self.transfer_edges)

    def metadata_cost(self) -> GraphMetadataCost:
        if self.is_explicit:
            compile_time = "O(E * (L + R)) explicit user edges"
        else:
            compile_time = "O(1 + L + R) structured descriptor"
        return GraphMetadataCost(
            storage_class="compile-time metadata",
            compile_time=compile_time,
            runtime_host="none allocated by TransferGraph",
            device_visible="none allocated by TransferGraph",
        )

    def project_initial(
        self,
    ) -> Union[Tuple[ProjectedTransfer, ...], ProjectedTransferFamily]:
        """Project transfers onto one topology level.

        Multi-level route materialization is represented by the domain model but
        rejected here until MD-14 adds ordered multi-level lowering.
        """

        if self.is_structured:
            assert self.structured is not None
            return self._project_structured_initial(self.structured)

        projections = []
        for edge in self.transfer_edges:
            projections.append(self._project_edge_initial(edge))
        return tuple(projections)

    def _validate(self) -> None:
        if self.structured is None:
            if not self.transfer_edges:
                raise ValueError("TransferGraph.edges requires at least one edge")
            return
        if self.structured.kind not in ("axis_neighbor", "gather", "multicast"):
            raise ValueError(
                f"unsupported structured transfer kind {self.structured.kind!r}"
            )

    def _project_edge_initial(self, edge: TransferEdge) -> ProjectedTransfer:
        differing_levels = self.domain.differing_levels(edge.source, edge.destination)
        if not differing_levels:
            return ProjectedTransfer(
                edge=edge,
                level_index=None,
                level_name=None,
                topology=None,
                source_level_coordinate=None,
                destination_level_coordinate=None,
            )
        if len(differing_levels) > 1:
            raise ValueError(
                "transfer edge requires multiple topology levels; multi-level route "
                "lowering is deferred to MD-14"
            )

        level_index = differing_levels[0]
        level = self.domain.levels[level_index]
        topology = self._require_level_topology(level)
        source_coord = edge.source.coordinates[level_index]
        if isinstance(edge.destination, DeviceRange):
            self._validate_range_routability(
                source_coord, edge.destination, level_index, level, topology
            )
            destination_coord: Union[Coordinate, DeviceRange] = edge.destination
        else:
            destination_level_coord = edge.destination.coordinates[level_index]
            self._validate_point_routability(
                source_coord, destination_level_coord, level, topology
            )
            destination_coord = destination_level_coord

        return ProjectedTransfer(
            edge=edge,
            level_index=level_index,
            level_name=level.name,
            topology=topology,
            source_level_coordinate=source_coord,
            destination_level_coordinate=destination_coord,
        )

    def _project_structured_initial(
        self, structured: StructuredTransfer
    ) -> ProjectedTransferFamily:
        assert structured.level_name is not None
        level_index = self.domain.level_index(structured.level_name)
        level = self.domain.levels[level_index]
        topology = self._require_level_topology(level)
        if structured.kind in ("gather", "multicast"):
            active_levels = [
                index
                for index, candidate in enumerate(self.domain.levels)
                if any(extent != 1 for extent in candidate.extent)
            ]
            if active_levels != [level_index]:
                raise ValueError(
                    f"{structured.kind} spans multiple topology levels; "
                    "multi-level route lowering is deferred to MD-14"
                )
        return ProjectedTransferFamily(
            graph=self,
            level_index=level_index,
            level_name=level.name,
            topology=topology,
        )

    @staticmethod
    def _default_structured_level(
        domain: DeviceDomain, level_name: Optional[str]
    ) -> str:
        if level_name is not None:
            domain.level_index(level_name)
            return level_name
        if domain.level_count == 1:
            return domain.levels[0].name
        active_levels = [
            level.name
            for level in domain.levels
            if any(extent != 1 for extent in level.extent)
        ]
        if len(active_levels) == 1:
            return active_levels[0]
        raise ValueError(
            "structured transfers over hierarchical domains require an explicit "
            "topology level"
        )

    @staticmethod
    def _require_level_topology(level: TopologyLevelInfo) -> FabricTopology:
        if level.topology is None:
            raise ValueError(f"topology level {level.name!r} has no fabric topology")
        return level.topology

    @staticmethod
    def _validate_point_routability(
        source_coord: Coordinate,
        destination_coord: Coordinate,
        level: TopologyLevelInfo,
        topology: FabricTopology,
    ) -> None:
        for axis, (source_component, destination_component) in enumerate(
            zip(source_coord, destination_coord)
        ):
            if axis == topology.cluster_axis:
                continue
            if source_component != destination_component:
                raise ValueError(
                    f"initial 1D routing on level {level.name!r} requires "
                    f"non-cluster axis {axis} to stay fixed, got "
                    f"{source_coord} -> {destination_coord}"
                )

    @staticmethod
    def _validate_range_routability(
        source_coord: Coordinate,
        destination_range: DeviceRange,
        level_index: int,
        level: TopologyLevelInfo,
        topology: FabricTopology,
    ) -> None:
        lo_coord = destination_range.lo.coordinates[level_index]
        hi_coord = destination_range.hi.coordinates[level_index]
        source_inside = True
        for axis, (source_component, lo_component, hi_component) in enumerate(
            zip(source_coord, lo_coord, hi_coord)
        ):
            if not (lo_component <= source_component < hi_component):
                source_inside = False
            if axis == topology.cluster_axis:
                continue
            if lo_component != source_component or hi_component != source_component + 1:
                raise ValueError(
                    f"initial 1D range routing on level {level.name!r} requires "
                    f"non-cluster axis {axis} to stay fixed at the source coordinate"
                )
        if source_inside:
            raise ValueError("source-in-destination multicast is deferred to MD-6")


__all__ = [
    "DeviceDomain",
    "DeviceRange",
    "DeviceRef",
    "Fabric1D",
    "FabricRing",
    "FabricTopology",
    "GraphMetadataCost",
    "ProjectedTransfer",
    "ProjectedTransferFamily",
    "StructuredTransfer",
    "TopologyLevelInfo",
    "TransferEdge",
    "TransferGraph",
]
