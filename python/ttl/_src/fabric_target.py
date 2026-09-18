# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN target binding for compiler-generated fabric routes."""

from dataclasses import dataclass
from enum import Enum, auto
import os
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FabricRouteSpec:
    """One logical local-to-remote route used by a generated kernel."""

    local_device: Tuple[int, ...]
    remote_device: Tuple[int, ...]
    source_nodes: Tuple[Tuple[int, ...], ...]
    route_index: int


class FabricManagerIntervalKind(Enum):
    GENERATED_RECEIVER = "generated_receiver"
    GENERATED_SENDER = "generated_sender"
    GENERATED_MIXED = "generated_mixed"
    EXTERNAL = "external"


@dataclass(frozen=True)
class FabricManagerIntervalSpec:
    """One compiler-proven fabric manager ownership interval."""

    identity: str
    kind: FabricManagerIntervalKind
    claim: Optional[str]
    route_indices: Tuple[int, ...]
    interfering_intervals: Tuple[str, ...]
    launch_nodes: Optional[Tuple[Tuple[int, int], ...]] = None


@dataclass(frozen=True)
class _ResolvedFabricRoute:
    device_chain: Tuple[Tuple[int, ...], ...]
    hop_count: int


@dataclass(frozen=True)
class _ResolvedFabricConnection:
    node_id: Any
    destination_node_id: Any
    direction: int


class _FabricRoutingMode(Enum):
    ONE_DIMENSIONAL = auto()
    NEIGHBOR_EXCHANGE = auto()
    TWO_DIMENSIONAL = auto()


class FabricRouteCache:
    """Cache control-plane facts for one mesh and fabric configuration."""

    def __init__(self) -> None:
        self._mesh_device = None
        self._fabric_config = None
        self._directions: Dict[Tuple[int, int, int, int], int] = {}
        self._forwarding_links: Dict[Tuple[int, int, int, int], Tuple[int, ...]] = {}

    @staticmethod
    def _node_key(node_id: Any) -> Tuple[int, int]:
        return (int(node_id.mesh_id), int(node_id.chip_id))

    def _prepare_query(self, mesh_device: Any, fabric_config: Any) -> None:
        if self._mesh_device is mesh_device and self._fabric_config == fabric_config:
            return
        self._mesh_device = mesh_device
        self._fabric_config = fabric_config
        self._directions.clear()
        self._forwarding_links.clear()

    def resolve_direction(
        self,
        ttnn_api: Any,
        mesh_device: Any,
        fabric_config: Any,
        source_node_id: Any,
        destination_node_id: Any,
    ) -> int:
        self._prepare_query(mesh_device, fabric_config)
        route_key = (
            *self._node_key(source_node_id),
            *self._node_key(destination_node_id),
        )
        if route_key not in self._directions:
            direction = ttnn_api.get_eth_forwarding_direction(
                source_node_id, destination_node_id
            )
            if direction is None:
                raise ValueError(
                    f"no fabric route from {source_node_id} to "
                    f"{destination_node_id}"
                )
            self._directions[route_key] = int(direction)
        return self._directions[route_key]

    def get_forwarding_links(
        self,
        ttnn_api: Any,
        mesh_device: Any,
        fabric_config: Any,
        source_node_id: Any,
        destination_node_id: Any,
    ) -> Tuple[int, ...]:
        self._prepare_query(mesh_device, fabric_config)
        route_key = (
            *self._node_key(source_node_id),
            *self._node_key(destination_node_id),
        )
        if route_key not in self._forwarding_links:
            get_forwarding_link_indices = getattr(
                ttnn_api, "get_forwarding_link_indices", None
            )
            if get_forwarding_link_indices is None:
                raise RuntimeError(
                    "TTNN must expose get_forwarding_link_indices() to assign "
                    "concurrent fabric connections"
                )
            forwarding_links = tuple(
                int(link_index)
                for link_index in get_forwarding_link_indices(
                    source_node_id, destination_node_id
                )
            )
            if not forwarding_links:
                raise ValueError(
                    f"no fabric forwarding link from {source_node_id} to "
                    f"{destination_node_id}"
                )
            self._forwarding_links[route_key] = forwarding_links
        return self._forwarding_links[route_key]


@dataclass(frozen=True)
class _FabricConnectionRequest:
    connection_node_id: Any
    direction: int
    eligible_links: Optional[Tuple[int, ...]]
    interval_ids: Tuple[str, ...]
    fixed_link_index: Optional[int] = None


@dataclass(frozen=True)
class _FabricManagerRequest:
    kernel_index: int
    node_coordinates: Tuple[int, int]
    fabric_runtime_metadata: Tuple[int, ...]
    connections: Tuple[_FabricConnectionRequest, ...]
    apply_binding: bool = True


@dataclass(frozen=True)
class _FabricConnectionBinding:
    connection_node_id: Any
    link_index: Optional[int]


@dataclass(frozen=True)
class _FabricManagerBinding:
    kernel_index: int
    node_coordinates: Tuple[int, int]
    caller_runtime_args: Tuple[int, ...]
    fabric_runtime_metadata: Tuple[int, ...]
    connections: Tuple[_FabricConnectionBinding, ...]


@dataclass
class _ExternalConnectionGroup:
    connection_node_id: Any
    eligible_links: Tuple[int, ...]
    fixed_link_index: int


@dataclass(frozen=True)
class FabricTargetBindingPlan:
    """Validated host-side routing-plane bindings for one logical device."""

    source_node_id: Any
    managed_kernel_indices: Tuple[int, ...]
    runtime_arg_base_common_indices: Tuple[Optional[int], ...]
    runtime_arg_bases: Tuple[Optional[int], ...]
    managers: Tuple[_FabricManagerBinding, ...]


_WORKER_SEMAPHORE_CAPACITY = 16


def _read_runtime_arg_row(
    kernel_descriptor: Any, node_coordinates: Tuple[int, int]
) -> Tuple[int, ...]:
    runtime_args = kernel_descriptor.runtime_args
    node_x, node_y = node_coordinates
    if isinstance(runtime_args, dict):
        column = runtime_args.get(node_x)
        if column is None:
            return ()
        return tuple(column.get(node_y, ()))
    try:
        return tuple(runtime_args[node_x][node_y])
    except LookupError:
        return ()


def _iter_descriptor_nodes(
    ttnn_api: Any, kernel_descriptor: Any, grid_cols: int, grid_rows: int
):
    for node_y in range(grid_rows):
        for node_x in range(grid_cols):
            if kernel_descriptor.core_ranges.contains(
                ttnn_api.CoreCoord(node_x, node_y)
            ):
                yield (node_x, node_y)


def _validate_kernel_aligned_fabric_metadata(
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    kernel_fabric_runtime_arg_base_common_indices: List[Optional[int]],
) -> None:
    kernel_count = len(program_descriptor.kernels)
    if len(kernel_fabric_routes) != kernel_count:
        raise ValueError(
            "kernel_fabric_routes must have one entry per kernel descriptor"
        )
    if len(kernel_fabric_runtime_arg_base_common_indices) != kernel_count:
        raise ValueError(
            "kernel_fabric_runtime_arg_base_common_indices must have one entry "
            "per kernel descriptor"
        )


def _plan_runtime_arg_bases(
    ttnn_api: Any,
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    kernel_fabric_runtime_arg_base_common_indices: List[Optional[int]],
    grid_cols: int,
    grid_rows: int,
) -> Tuple[Optional[int], ...]:
    runtime_arg_bases = []
    for kernel_index, routes in enumerate(kernel_fabric_routes):
        common_index = kernel_fabric_runtime_arg_base_common_indices[kernel_index]
        if not routes:
            if common_index is not None:
                raise ValueError(
                    f"kernel {kernel_index} has a fabric runtime argument base "
                    "but no fabric routes"
                )
            runtime_arg_bases.append(None)
            continue
        if common_index is None:
            raise ValueError(
                f"kernel {kernel_index} has fabric routes but no fabric runtime "
                "argument base common index"
            )

        kernel_descriptor = program_descriptor.kernels[kernel_index]
        if common_index < 0 or common_index >= len(
            kernel_descriptor.common_runtime_args
        ):
            raise ValueError(
                f"kernel {kernel_index} fabric runtime argument base common "
                f"index {common_index} is outside its common argument table"
            )
        if kernel_descriptor.common_runtime_args[common_index] != 0:
            raise ValueError(
                f"kernel {kernel_index} fabric runtime argument base common "
                "argument must be initialized to zero"
            )

        runtime_arg_base = 0
        for node_coordinates in _iter_descriptor_nodes(
            ttnn_api, kernel_descriptor, grid_cols, grid_rows
        ):
            runtime_arg_base = max(
                runtime_arg_base,
                len(_read_runtime_arg_row(kernel_descriptor, node_coordinates)),
            )
        runtime_arg_bases.append(runtime_arg_base)
    return tuple(runtime_arg_bases)


def _build_mesh_coordinate(ttnn_api: Any, coordinates: Tuple[int, ...]) -> Any:
    try:
        return ttnn_api.MeshCoordinate(*coordinates)
    except TypeError:
        return ttnn_api.MeshCoordinate(coordinates)


def _get_fabric_node_id(
    ttnn_api: Any, mesh_device: Any, coordinates: Tuple[int, ...]
) -> Any:
    return mesh_device.get_fabric_node_id(_build_mesh_coordinate(ttnn_api, coordinates))


def _flatten_device_ref(device_ref: Any) -> Tuple[int, ...]:
    return tuple(value for coordinate in device_ref.coordinates for value in coordinate)


def _get_fabric_routing_mode(ttnn_api: Any, fabric_config: Any) -> _FabricRoutingMode:
    fabric_config_type = getattr(ttnn_api, "FabricConfig", None)
    mode_config_names = (
        (
            _FabricRoutingMode.NEIGHBOR_EXCHANGE,
            ("FABRIC_1D_NEIGHBOR_EXCHANGE",),
        ),
        (
            _FabricRoutingMode.ONE_DIMENSIONAL,
            ("FABRIC_1D", "FABRIC_1D_RING"),
        ),
        (
            _FabricRoutingMode.TWO_DIMENSIONAL,
            (
                "FABRIC_2D",
                "FABRIC_2D_TORUS_X",
                "FABRIC_2D_TORUS_Y",
                "FABRIC_2D_TORUS_XY",
            ),
        ),
    )
    for routing_mode, config_names in mode_config_names:
        if any(
            (config_value := getattr(fabric_config_type, config_name, None)) is not None
            and fabric_config == config_value
            for config_name in config_names
        ):
            return routing_mode
    raise ValueError(f"unsupported fabric configuration {fabric_config}")


def _resolve_fabric_route(
    routing_mode: _FabricRoutingMode,
    local_device: Tuple[int, ...],
    remote_device: Tuple[int, ...],
) -> _ResolvedFabricRoute:
    """Resolve the manager endpoint and target-specific route metadata."""
    if routing_mode == _FabricRoutingMode.TWO_DIMENSIONAL:
        return _ResolvedFabricRoute((local_device, remote_device), 0)

    differing_axes = [
        axis
        for axis, (local_coordinate, remote_coordinate) in enumerate(
            zip(local_device, remote_device)
        )
        if local_coordinate != remote_coordinate
    ]
    if len(local_device) != len(remote_device) or len(differing_axes) != 1:
        raise ValueError(
            "FABRIC_1D routes must connect devices on one logical mesh axis"
        )

    route_axis = differing_axes[0]
    route_step = 1 if remote_device[route_axis] > local_device[route_axis] else -1
    route_coordinates = range(
        local_device[route_axis],
        remote_device[route_axis] + route_step,
        route_step,
    )
    device_chain = []
    for route_coordinate in route_coordinates:
        device = list(local_device)
        device[route_axis] = route_coordinate
        device_chain.append(tuple(device))
    if routing_mode == _FabricRoutingMode.NEIGHBOR_EXCHANGE and len(device_chain) != 2:
        raise ValueError(
            "FABRIC_1D_NEIGHBOR_EXCHANGE only supports adjacent device routes"
        )
    return _ResolvedFabricRoute(tuple(device_chain), len(device_chain) - 1)


def _resolve_fabric_connection(
    ttnn_api: Any,
    route_cache: FabricRouteCache,
    mesh_device: Any,
    fabric_config: Any,
    resolved_route: _ResolvedFabricRoute,
) -> _ResolvedFabricConnection:
    route_node_ids = tuple(
        _get_fabric_node_id(ttnn_api, mesh_device, coordinates)
        for coordinates in resolved_route.device_chain
    )
    hop_directions = tuple(
        route_cache.resolve_direction(
            ttnn_api,
            mesh_device,
            fabric_config,
            route_node_ids[hop_index],
            route_node_ids[hop_index + 1],
        )
        for hop_index in range(len(route_node_ids) - 1)
    )
    if any(direction != hop_directions[0] for direction in hop_directions[1:]):
        raise ValueError("FABRIC_1D routes require one forwarding direction")
    return _ResolvedFabricConnection(
        node_id=route_node_ids[1],
        destination_node_id=route_node_ids[-1],
        direction=hop_directions[0],
    )


def _intersect_forwarding_links(
    ttnn_api: Any,
    route_cache: FabricRouteCache,
    mesh_device: Any,
    fabric_config: Any,
    source_node_id: Any,
    connection_node_ids: List[Any],
) -> Tuple[int, ...]:
    connection_links = [
        route_cache.get_forwarding_links(
            ttnn_api,
            mesh_device,
            fabric_config,
            source_node_id,
            connection_node_id,
        )
        for connection_node_id in connection_node_ids
    ]
    return tuple(
        link_index
        for link_index in connection_links[0]
        if all(link_index in links for links in connection_links[1:])
    )


def _assign_fabric_links(
    manager_requests: List[_FabricManagerRequest],
    interference_by_interval: Dict[str, frozenset[str]],
) -> Dict[Tuple[int, int], Optional[int]]:
    connections_by_direction = {}
    for manager_index, manager_request in enumerate(manager_requests):
        for connection_index, connection in enumerate(manager_request.connections):
            connections_by_direction.setdefault(connection.direction, []).append(
                (manager_index, connection_index)
            )

    def connections_interfere(lhs_key, rhs_key):
        lhs_manager_index, lhs_connection_index = lhs_key
        rhs_manager_index, rhs_connection_index = rhs_key
        lhs = manager_requests[lhs_manager_index].connections[lhs_connection_index]
        rhs = manager_requests[rhs_manager_index].connections[rhs_connection_index]
        if not lhs.interval_ids or not rhs.interval_ids:
            return True
        for lhs_interval in lhs.interval_ids:
            for rhs_interval in rhs.interval_ids:
                if lhs_interval == rhs_interval:
                    return True
                if rhs_interval in interference_by_interval.get(
                    lhs_interval, frozenset()
                ):
                    return True
        return False

    selected_links = {}
    for direction, connection_keys in connections_by_direction.items():
        adjacency = {connection_key: set() for connection_key in connection_keys}
        for connection_position, lhs_key in enumerate(connection_keys):
            for rhs_key in connection_keys[connection_position + 1 :]:
                if connections_interfere(lhs_key, rhs_key):
                    adjacency[lhs_key].add(rhs_key)
                    adjacency[rhs_key].add(lhs_key)

        has_fixed_link = any(
            manager_requests[manager_index]
            .connections[connection_index]
            .fixed_link_index
            is not None
            for manager_index, connection_index in connection_keys
        )
        if len(connection_keys) == 1 and not has_fixed_link:
            connection_key = connection_keys[0]
            manager_index, connection_index = connection_key
            connection = manager_requests[manager_index].connections[connection_index]
            if connection.eligible_links is None:
                # The control plane's default is sufficient when no other
                # manager can contend for the same directional link.
                selected_links[connection_key] = None
                continue

        needs_explicit_links = (
            has_fixed_link
            or any(adjacency.values())
            or any(
                manager_requests[manager_index]
                .connections[connection_index]
                .eligible_links
                is not None
                for manager_index, connection_index in connection_keys
            )
        )
        if not needs_explicit_links:
            for connection_key in connection_keys:
                selected_links[connection_key] = None
            continue

        if any(
            manager_requests[manager_index].connections[connection_index].eligible_links
            is None
            for manager_index, connection_index in connection_keys
        ):
            raise RuntimeError(
                "TTNN must expose get_forwarding_link_indices() to assign "
                "interfering fabric connections"
            )

        ordered_keys = sorted(
            connection_keys,
            key=lambda connection_key: (
                manager_requests[connection_key[0]]
                .connections[connection_key[1]]
                .fixed_link_index
                is None,
                len(
                    manager_requests[connection_key[0]]
                    .connections[connection_key[1]]
                    .eligible_links
                ),
                -len(adjacency[connection_key]),
                manager_requests[connection_key[0]]
                .connections[connection_key[1]]
                .interval_ids,
                connection_key,
            ),
        )

        def assign_connection(position):
            if position == len(ordered_keys):
                return True
            connection_key = ordered_keys[position]
            manager_index, connection_index = connection_key
            connection = manager_requests[manager_index].connections[connection_index]
            assert connection.eligible_links is not None
            candidate_links = (
                (connection.fixed_link_index,)
                if connection.fixed_link_index is not None
                else connection.eligible_links
            )
            for link_index in candidate_links:
                if link_index not in connection.eligible_links:
                    continue
                if any(
                    selected_links.get(neighbor) == link_index
                    for neighbor in adjacency[connection_key]
                ):
                    continue
                selected_links[connection_key] = link_index
                if assign_connection(position + 1):
                    return True
                del selected_links[connection_key]
            return False

        if not assign_connection(0):
            participants = []
            for connection_key in ordered_keys:
                manager_index, connection_index = connection_key
                manager = manager_requests[manager_index]
                connection = manager.connections[connection_index]
                identities = connection.interval_ids or ("<unknown>",)
                participants.append(
                    f"{identities} at kernel {manager.kernel_index}, node "
                    f"{manager.node_coordinates}, fixed link "
                    f"{connection.fixed_link_index}, eligible links "
                    f"{connection.eligible_links}"
                )
            interference = tuple(
                (
                    manager_requests[lhs_key[0]].connections[lhs_key[1]].interval_ids,
                    manager_requests[rhs_key[0]].connections[rhs_key[1]].interval_ids,
                )
                for lhs_key in ordered_keys
                for rhs_key in sorted(adjacency[lhs_key])
                if lhs_key < rhs_key
            )
            raise ValueError(
                "fabric connection plan cannot assign distinct forwarding "
                f"links to interfering managers in direction {direction}; "
                f"participants: {'; '.join(participants)}; "
                f"interference: {interference}"
            )
    return selected_links


def _build_interval_interference(
    kernel_intervals: List[Tuple[FabricManagerIntervalSpec, ...]],
) -> Dict[str, frozenset[str]]:
    def specialization_invariant(interval: FabricManagerIntervalSpec):
        return (
            interval.identity,
            interval.kind,
            interval.claim,
            interval.route_indices,
            interval.interfering_intervals,
        )

    intervals_by_identity = {}
    for intervals in kernel_intervals:
        for interval in intervals:
            existing = intervals_by_identity.setdefault(interval.identity, interval)
            if specialization_invariant(existing) != specialization_invariant(interval):
                raise ValueError(
                    f"fabric manager interval {interval.identity!r} has "
                    "inconsistent specialized records"
                )
    known_identities = frozenset(intervals_by_identity)
    interference = {}
    for identity, interval in intervals_by_identity.items():
        unknown = frozenset(interval.interfering_intervals) - known_identities
        if unknown:
            raise ValueError(
                f"fabric manager interval {identity!r} references unknown "
                f"interfering intervals {tuple(sorted(unknown))}"
            )
        interference[identity] = frozenset(interval.interfering_intervals)
    for identity, interfering_identities in interference.items():
        for interfering_identity in interfering_identities:
            if identity not in interference[interfering_identity]:
                raise ValueError(
                    f"fabric manager interference between {identity!r} and "
                    f"{interfering_identity!r} is not symmetric"
                )
    return interference


def _validate_manager_semaphore_capacity(
    ttnn_api: Any,
    program_descriptor: Any,
    manager_requests: List[_FabricManagerRequest],
) -> None:
    required_by_node: Dict[Tuple[int, int], int] = {}
    for manager in manager_requests:
        if manager.apply_binding:
            required_by_node[manager.node_coordinates] = required_by_node.get(
                manager.node_coordinates, 0
            ) + 2 * len(manager.connections)

    for node_coordinates, required_count in required_by_node.items():
        worker_node = ttnn_api.CoreCoord(*node_coordinates)
        used_ids = {
            int(semaphore.id)
            for semaphore in program_descriptor.semaphores
            if getattr(semaphore.core_type, "name", semaphore.core_type) == "WORKER"
            and semaphore.core_ranges.contains(worker_node)
        }
        available_count = _WORKER_SEMAPHORE_CAPACITY - len(used_ids)
        if required_count > available_count:
            raise ValueError(
                f"fabric managers at node {node_coordinates} require "
                f"{required_count} worker semaphore IDs, but only "
                f"{available_count} of {_WORKER_SEMAPHORE_CAPACITY} remain"
            )


def build_fabric_target_binding_plan(
    ttnn_api: Any,
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    kernel_fabric_runtime_arg_base_common_indices: List[Optional[int]],
    mesh_device: Any,
    device_coordinates: Tuple[int, ...],
    grid_cols: int,
    grid_rows: int,
    kernel_fabric_manager_intervals: Optional[
        List[Tuple[FabricManagerIntervalSpec, ...]]
    ] = None,
    external_fabric_connections: Tuple[Any, ...] = (),
    route_cache: Optional[FabricRouteCache] = None,
) -> FabricTargetBindingPlan:
    """Resolve and validate all fabric managers before descriptor mutation."""
    _validate_kernel_aligned_fabric_metadata(
        program_descriptor,
        kernel_fabric_routes,
        kernel_fabric_runtime_arg_base_common_indices,
    )
    manager_intervals = kernel_fabric_manager_intervals or [
        () for _ in program_descriptor.kernels
    ]
    if len(manager_intervals) != len(program_descriptor.kernels):
        raise ValueError(
            "kernel_fabric_manager_intervals must have one entry per kernel "
            "descriptor"
        )
    interference_by_interval = _build_interval_interference(manager_intervals)
    runtime_arg_bases = _plan_runtime_arg_bases(
        ttnn_api,
        program_descriptor,
        kernel_fabric_routes,
        kernel_fabric_runtime_arg_base_common_indices,
        grid_cols,
        grid_rows,
    )

    source_node_id = _get_fabric_node_id(ttnn_api, mesh_device, device_coordinates)
    active_route_cache = route_cache or FabricRouteCache()
    fabric_config = ttnn_api.get_fabric_config()
    routing_mode = _get_fabric_routing_mode(ttnn_api, fabric_config)
    can_enumerate_forwarding_links = (
        getattr(ttnn_api, "get_forwarding_link_indices", None) is not None
    )
    manager_requests = []
    for kernel_index, routes in enumerate(kernel_fabric_routes):
        if not routes:
            continue

        kernel_descriptor = program_descriptor.kernels[kernel_index]
        route_count = max(route.route_index for route in routes) + 1
        for node_coordinates in _iter_descriptor_nodes(
            ttnn_api, kernel_descriptor, grid_cols, grid_rows
        ):
            active_routes = [
                route
                for route in routes
                if route.local_device == device_coordinates
                and node_coordinates in route.source_nodes
            ]
            active_remote_devices = tuple(
                dict.fromkeys(route.remote_device for route in active_routes)
            )
            remote_index = {
                remote_device: remote_slot
                for remote_slot, remote_device in enumerate(active_remote_devices)
            }

            resolved_routes = [
                _resolve_fabric_route(routing_mode, device_coordinates, remote_device)
                for remote_device in active_remote_devices
            ]
            resolved_connections = [
                _resolve_fabric_connection(
                    ttnn_api,
                    active_route_cache,
                    mesh_device,
                    fabric_config,
                    resolved_route,
                )
                for resolved_route in resolved_routes
            ]
            route_connection_node_ids = [
                connection.node_id for connection in resolved_connections
            ]
            destination_node_ids = [
                connection.destination_node_id for connection in resolved_connections
            ]
            route_directions = [
                connection.direction for connection in resolved_connections
            ]
            connection_index_by_direction = {}
            connection_node_ids = []
            connection_directions = []
            connection_nodes_by_connection = []
            remote_connection_slots = []
            for connection_node_id, direction in zip(
                route_connection_node_ids, route_directions
            ):
                connection_index = connection_index_by_direction.get(direction)
                if connection_index is None:
                    connection_index = len(connection_node_ids)
                    connection_index_by_direction[direction] = connection_index
                    connection_node_ids.append(connection_node_id)
                    connection_directions.append(direction)
                    connection_nodes_by_connection.append([])
                connection_nodes_by_connection[connection_index].append(
                    connection_node_id
                )
                remote_connection_slots.append(connection_index)

            route_slots = [0] * route_count
            destination_device_ids = [0] * route_count
            destination_mesh_ids = [0] * route_count
            destination_hop_counts = [0] * route_count
            active_route_indices = set()
            for route in active_routes:
                if route.route_index in active_route_indices:
                    raise ValueError(
                        "active fabric routes must have distinct route indices"
                    )
                active_route_indices.add(route.route_index)
                remote_slot = remote_index[route.remote_device]
                route_slots[route.route_index] = remote_connection_slots[remote_slot]
                destination_node_id = destination_node_ids[remote_slot]
                destination_device_ids[route.route_index] = int(
                    destination_node_id.chip_id
                )
                destination_mesh_ids[route.route_index] = int(
                    destination_node_id.mesh_id
                )
                destination_hop_counts[route.route_index] = resolved_routes[
                    remote_slot
                ].hop_count

            fabric_runtime_metadata = (
                len(connection_node_ids),
                *route_slots,
                *destination_device_ids,
                *destination_mesh_ids,
                *destination_hop_counts,
            )
            generated_interval_ids = tuple(
                interval.identity
                for interval in manager_intervals[kernel_index]
                if interval.kind != FabricManagerIntervalKind.EXTERNAL
                and any(
                    route_index in active_route_indices
                    for route_index in interval.route_indices
                )
            )
            connection_requests = []
            for connection_index, connection_node_id in enumerate(connection_node_ids):
                eligible_links = None
                if can_enumerate_forwarding_links:
                    eligible_links = _intersect_forwarding_links(
                        ttnn_api,
                        active_route_cache,
                        mesh_device,
                        fabric_config,
                        source_node_id,
                        connection_nodes_by_connection[connection_index],
                    )
                    if not eligible_links:
                        raise ValueError(
                            "fabric destinations sharing one direction have "
                            "no common forwarding link"
                        )
                connection_requests.append(
                    _FabricConnectionRequest(
                        connection_node_id=connection_node_id,
                        direction=connection_directions[connection_index],
                        eligible_links=eligible_links,
                        interval_ids=generated_interval_ids,
                    )
                )
            manager_requests.append(
                _FabricManagerRequest(
                    kernel_index=kernel_index,
                    node_coordinates=node_coordinates,
                    fabric_runtime_metadata=fabric_runtime_metadata,
                    connections=tuple(connection_requests),
                )
            )

    external_request_groups: Dict[
        Tuple[int, Tuple[int, int], int, str], _ExternalConnectionGroup
    ] = {}
    expected_external_nodes = {}
    provided_external_nodes = {}
    for binding in external_fabric_connections:
        interval_matches = [
            (kernel_index, interval)
            for kernel_index, intervals in enumerate(manager_intervals)
            for interval in intervals
            if interval.kind == FabricManagerIntervalKind.EXTERNAL
            and interval.claim == binding.claim.identity
        ]
        interval_identities = {interval.identity for _, interval in interval_matches}
        if len(interval_identities) != 1:
            raise ValueError(
                f"external fabric claim {binding.claim.identity!r} must map "
                "to one manager interval"
            )
        interval_identity = next(iter(interval_identities))
        interval_nodes = set()
        for kernel_index, interval in interval_matches:
            descriptor_nodes = set(
                _iter_descriptor_nodes(
                    ttnn_api,
                    program_descriptor.kernels[kernel_index],
                    grid_cols,
                    grid_rows,
                )
            )
            descriptor_interval_nodes = (
                descriptor_nodes
                if interval.launch_nodes is None
                else set(interval.launch_nodes)
            )
            outside_nodes = descriptor_interval_nodes - descriptor_nodes
            if outside_nodes:
                raise ValueError(
                    f"external fabric interval {interval_identity!r} has "
                    f"launch nodes outside kernel descriptor {kernel_index}: "
                    f"{tuple(sorted(outside_nodes))}"
                )
            interval_nodes.update(descriptor_interval_nodes)
        expected_external_nodes.setdefault(interval_identity, set()).update(
            interval_nodes
        )
        for requirement in binding.connections:
            local_device = _flatten_device_ref(requirement.local_device)
            if local_device != device_coordinates:
                continue
            remote_device = _flatten_device_ref(requirement.remote_device)
            resolved_route = _resolve_fabric_route(
                routing_mode, device_coordinates, remote_device
            )
            resolved_connection = _resolve_fabric_connection(
                ttnn_api,
                active_route_cache,
                mesh_device,
                fabric_config,
                resolved_route,
            )
            eligible_links = active_route_cache.get_forwarding_links(
                ttnn_api,
                mesh_device,
                fabric_config,
                source_node_id,
                resolved_connection.node_id,
            )
            if requirement.fixed_link_index not in eligible_links:
                raise ValueError(
                    f"external fabric claim {binding.claim.identity!r} fixed "
                    f"link {requirement.fixed_link_index} is not eligible for "
                    f"direction {resolved_connection.direction}; eligible links "
                    f"are {eligible_links}"
                )
            for node_coordinates in requirement.worker_nodes:
                worker_node = ttnn_api.CoreCoord(*node_coordinates)
                matching_kernel_indices = [
                    kernel_index
                    for kernel_index, _ in interval_matches
                    if program_descriptor.kernels[kernel_index].core_ranges.contains(
                        worker_node
                    )
                ]
                if len(matching_kernel_indices) != 1:
                    raise ValueError(
                        f"external fabric claim {binding.claim.identity!r} "
                        f"node {node_coordinates} must map to one kernel "
                        "descriptor"
                    )
                group_key = (
                    matching_kernel_indices[0],
                    node_coordinates,
                    resolved_connection.direction,
                    interval_identity,
                )
                provided_external_nodes.setdefault(interval_identity, set()).add(
                    node_coordinates
                )
                group = external_request_groups.setdefault(
                    group_key,
                    _ExternalConnectionGroup(
                        connection_node_id=resolved_connection.node_id,
                        eligible_links=eligible_links,
                        fixed_link_index=requirement.fixed_link_index,
                    ),
                )
                if group.fixed_link_index != requirement.fixed_link_index:
                    raise ValueError(
                        f"external fabric claim {binding.claim.identity!r} "
                        f"node {node_coordinates} assigns multiple fixed links "
                        f"in direction {resolved_connection.direction}"
                    )
                group.eligible_links = tuple(
                    link_index
                    for link_index in group.eligible_links
                    if link_index in eligible_links
                )

    for interval_identity, expected_nodes in expected_external_nodes.items():
        provided_nodes = provided_external_nodes.get(interval_identity, set())
        if provided_nodes != expected_nodes:
            missing_nodes = tuple(sorted(expected_nodes - provided_nodes))
            extra_nodes = tuple(sorted(provided_nodes - expected_nodes))
            raise ValueError(
                f"external fabric interval {interval_identity!r} does not "
                f"cover its launch-node domain; missing nodes {missing_nodes}, "
                f"extra nodes {extra_nodes}"
            )

    for (
        kernel_index,
        node_coordinates,
        direction,
        interval_identity,
    ), group in external_request_groups.items():
        if not group.eligible_links:
            raise ValueError(
                f"external fabric interval {interval_identity!r} destinations "
                "have no common forwarding link"
            )
        manager_requests.append(
            _FabricManagerRequest(
                kernel_index=kernel_index,
                node_coordinates=node_coordinates,
                fabric_runtime_metadata=(),
                connections=(
                    _FabricConnectionRequest(
                        connection_node_id=group.connection_node_id,
                        direction=direction,
                        eligible_links=group.eligible_links,
                        interval_ids=(interval_identity,),
                        fixed_link_index=group.fixed_link_index,
                    ),
                ),
                apply_binding=False,
            )
        )

    _validate_manager_semaphore_capacity(ttnn_api, program_descriptor, manager_requests)
    selected_links = _assign_fabric_links(manager_requests, interference_by_interval)
    manager_bindings = []
    for manager_index, manager_request in enumerate(manager_requests):
        connection_bindings = tuple(
            _FabricConnectionBinding(
                connection_node_id=connection.connection_node_id,
                link_index=selected_links[(manager_index, connection_index)],
            )
            for connection_index, connection in enumerate(manager_request.connections)
        )
        if not manager_request.apply_binding:
            continue
        manager_bindings.append(
            _FabricManagerBinding(
                kernel_index=manager_request.kernel_index,
                node_coordinates=manager_request.node_coordinates,
                caller_runtime_args=_read_runtime_arg_row(
                    program_descriptor.kernels[manager_request.kernel_index],
                    manager_request.node_coordinates,
                ),
                fabric_runtime_metadata=manager_request.fabric_runtime_metadata,
                connections=connection_bindings,
            )
        )
    return FabricTargetBindingPlan(
        source_node_id=source_node_id,
        managed_kernel_indices=tuple(
            kernel_index
            for kernel_index, routes in enumerate(kernel_fabric_routes)
            if routes
        ),
        runtime_arg_base_common_indices=tuple(
            kernel_fabric_runtime_arg_base_common_indices
        ),
        runtime_arg_bases=runtime_arg_bases,
        managers=tuple(manager_bindings),
    )


def apply_fabric_target_binding_plan(
    ttnn_api: Any,
    program_descriptor: Any,
    plan: FabricTargetBindingPlan,
    device_coordinates: Tuple[int, ...],
) -> None:
    """Apply a validated target-binding plan to one program descriptor."""
    applied_managers = []
    for manager in plan.managers:
        kernel_descriptor = program_descriptor.kernels[manager.kernel_index]
        node_x, node_y = manager.node_coordinates
        connection_node_ids = []
        connection_link_indices = []
        fabric_args = []
        if manager.connections:
            connection_node_ids = [
                connection.connection_node_id for connection in manager.connections
            ]
            explicit_link_indices = [
                connection.link_index
                for connection in manager.connections
                if connection.link_index is not None
            ]
            assert not explicit_link_indices or len(explicit_link_indices) == len(
                manager.connections
            )
            connection_link_indices = explicit_link_indices
            fabric_args = ttnn_api.setup_routing_plane_connection(
                plan.source_node_id,
                connection_node_ids,
                connection_link_indices,
                program_descriptor,
                manager.kernel_index,
                ttnn_api.CoreCoord(node_x, node_y),
            )
        runtime_arg_base = plan.runtime_arg_bases[manager.kernel_index]
        assert runtime_arg_base is not None
        runtime_args = [*manager.caller_runtime_args]
        runtime_args.extend([0] * (runtime_arg_base - len(runtime_args)))
        runtime_args.extend(manager.fabric_runtime_metadata)
        runtime_args.extend(fabric_args)
        applied_managers.append((manager, runtime_args))
        if os.environ.get("TTLANG_DEBUG_FABRIC_ARGS"):
            print(
                "fabric runtime args:",
                device_coordinates,
                manager.kernel_index,
                manager.node_coordinates,
                connection_node_ids,
                connection_link_indices,
                runtime_args,
                flush=True,
            )

    for kernel_index in plan.managed_kernel_indices:
        runtime_arg_base = plan.runtime_arg_bases[kernel_index]
        common_index = plan.runtime_arg_base_common_indices[kernel_index]
        assert runtime_arg_base is not None
        assert common_index is not None
        program_descriptor.kernels[kernel_index].common_runtime_args[
            common_index
        ] = runtime_arg_base
    for manager, runtime_args in applied_managers:
        node_x, node_y = manager.node_coordinates
        program_descriptor.kernels[manager.kernel_index].runtime_args[node_x][
            node_y
        ] = runtime_args


def configure_routing_plane_runtime_args(
    ttnn_api: Any,
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    kernel_fabric_runtime_arg_base_common_indices: List[Optional[int]],
    mesh_device: Any,
    device_coordinates: Tuple[int, ...],
    grid_cols: int,
    grid_rows: int,
    route_cache: Optional[FabricRouteCache] = None,
    kernel_fabric_manager_intervals: Optional[
        List[Tuple[FabricManagerIntervalSpec, ...]]
    ] = None,
    external_fabric_connections: Tuple[Any, ...] = (),
) -> None:
    """Plan and apply routing-plane target bindings for one logical device."""
    _validate_kernel_aligned_fabric_metadata(
        program_descriptor,
        kernel_fabric_routes,
        kernel_fabric_runtime_arg_base_common_indices,
    )
    if not any(kernel_fabric_routes) and not external_fabric_connections:
        if any(
            common_index is not None
            for common_index in kernel_fabric_runtime_arg_base_common_indices
        ):
            raise ValueError(
                "fabric runtime argument base common indices require fabric routes"
            )
        return
    plan = build_fabric_target_binding_plan(
        ttnn_api=ttnn_api,
        program_descriptor=program_descriptor,
        kernel_fabric_routes=kernel_fabric_routes,
        kernel_fabric_runtime_arg_base_common_indices=(
            kernel_fabric_runtime_arg_base_common_indices
        ),
        kernel_fabric_manager_intervals=kernel_fabric_manager_intervals,
        external_fabric_connections=external_fabric_connections,
        mesh_device=mesh_device,
        device_coordinates=device_coordinates,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        route_cache=route_cache,
    )
    apply_fabric_target_binding_plan(
        ttnn_api=ttnn_api,
        program_descriptor=program_descriptor,
        plan=plan,
        device_coordinates=device_coordinates,
    )


__all__ = [
    "FabricManagerIntervalKind",
    "FabricManagerIntervalSpec",
    "FabricRouteCache",
    "FabricRouteSpec",
    "FabricTargetBindingPlan",
    "apply_fabric_target_binding_plan",
    "build_fabric_target_binding_plan",
    "configure_routing_plane_runtime_args",
]
