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


@dataclass(frozen=True)
class _ResolvedFabricRoute:
    device_chain: Tuple[Tuple[int, ...], ...]
    hop_count: int

    @property
    def connection_device(self) -> Tuple[int, ...]:
        return self.device_chain[1]


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


@dataclass(frozen=True)
class _FabricManagerRequest:
    kernel_index: int
    node_coordinates: Tuple[int, int]
    runtime_prefix: Tuple[int, ...]
    connections: Tuple[_FabricConnectionRequest, ...]


@dataclass(frozen=True)
class _FabricConnectionBinding:
    connection_node_id: Any
    link_index: Optional[int]


@dataclass(frozen=True)
class _FabricManagerBinding:
    kernel_index: int
    node_coordinates: Tuple[int, int]
    runtime_prefix: Tuple[int, ...]
    connections: Tuple[_FabricConnectionBinding, ...]


@dataclass(frozen=True)
class FabricTargetBindingPlan:
    """Validated host-side routing-plane bindings for one logical device."""

    source_node_id: Any
    managed_kernel_indices: Tuple[int, ...]
    managers: Tuple[_FabricManagerBinding, ...]


def _build_mesh_coordinate(ttnn_api: Any, coordinates: Tuple[int, ...]) -> Any:
    try:
        return ttnn_api.MeshCoordinate(*coordinates)
    except TypeError:
        return ttnn_api.MeshCoordinate(coordinates)


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


def _assign_distinct_links(
    manager_requests: List[_FabricManagerRequest],
) -> Dict[Tuple[int, int], Optional[int]]:
    connections_by_direction = {}
    for manager_index, manager_request in enumerate(manager_requests):
        for connection_index, connection in enumerate(manager_request.connections):
            connections_by_direction.setdefault(connection.direction, []).append(
                (manager_index, connection_index)
            )

    selected_links = {}
    for direction, connection_keys in connections_by_direction.items():
        if len(connection_keys) == 1:
            connection_key = connection_keys[0]
            manager_index, connection_index = connection_key
            connection = manager_requests[manager_index].connections[connection_index]
            if connection.eligible_links is None:
                # The control plane's default is sufficient when no other
                # manager can contend for the same directional link.
                selected_links[connection_key] = None
                continue

        if any(
            manager_requests[manager_index].connections[connection_index].eligible_links
            is None
            for manager_index, connection_index in connection_keys
        ):
            raise RuntimeError(
                "TTNN must expose get_forwarding_link_indices() to assign "
                "concurrent fabric connections"
            )

        link_owner = {}

        def assign_connection(connection_key, visited_links):
            manager_index, connection_index = connection_key
            connection = manager_requests[manager_index].connections[connection_index]
            assert connection.eligible_links is not None

            # Preserve the control plane's preference unless a later request
            # cannot be satisfied without reassignment.
            for link_index in connection.eligible_links:
                if link_index in visited_links or link_index in link_owner:
                    continue
                visited_links.add(link_index)
                link_owner[link_index] = connection_key
                selected_links[connection_key] = link_index
                return True

            for link_index in connection.eligible_links:
                if link_index in visited_links:
                    continue
                visited_links.add(link_index)
                current_owner = link_owner.get(link_index)
                if current_owner is None:
                    continue
                if not assign_connection(current_owner, visited_links):
                    continue
                link_owner[link_index] = connection_key
                selected_links[connection_key] = link_index
                return True
            return False

        for connection_key in connection_keys:
            if assign_connection(connection_key, set()):
                continue
            raise ValueError(
                "fabric connection plan cannot assign distinct forwarding "
                f"links to {len(connection_keys)} concurrent connections in "
                f"direction {direction}"
            )
    return selected_links


def build_fabric_target_binding_plan(
    ttnn_api: Any,
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    mesh_device: Any,
    device_coordinates: Tuple[int, ...],
    grid_cols: int,
    grid_rows: int,
    route_cache: Optional[FabricRouteCache] = None,
) -> FabricTargetBindingPlan:
    """Resolve and validate all fabric managers before descriptor mutation."""
    if len(kernel_fabric_routes) != len(program_descriptor.kernels):
        raise ValueError(
            "kernel_fabric_routes must have one entry per kernel descriptor"
        )

    source_node_id = mesh_device.get_fabric_node_id(
        _build_mesh_coordinate(ttnn_api, device_coordinates)
    )
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
        for node_y in range(grid_rows):
            for node_x in range(grid_cols):
                worker_node = ttnn_api.CoreCoord(node_x, node_y)
                if not kernel_descriptor.core_ranges.contains(worker_node):
                    continue
                node_coordinates = (node_x, node_y)
                active_remote_devices = []
                remote_index = {}
                route_remote_slots = [0] * len(routes)
                for route_index, route in enumerate(routes):
                    if route.local_device != device_coordinates:
                        continue
                    if node_coordinates not in route.source_nodes:
                        continue
                    if route.remote_device not in remote_index:
                        remote_index[route.remote_device] = len(active_remote_devices)
                        active_remote_devices.append(route.remote_device)
                    route_remote_slots[route_index] = remote_index[route.remote_device]

                destination_node_ids = [
                    mesh_device.get_fabric_node_id(
                        _build_mesh_coordinate(ttnn_api, coordinates)
                    )
                    for coordinates in active_remote_devices
                ]
                resolved_routes = [
                    _resolve_fabric_route(
                        routing_mode, device_coordinates, remote_device
                    )
                    for remote_device in active_remote_devices
                ]
                route_connection_node_ids = [
                    mesh_device.get_fabric_node_id(
                        _build_mesh_coordinate(
                            ttnn_api, resolved_route.connection_device
                        )
                    )
                    for resolved_route in resolved_routes
                ]
                route_directions = []
                for resolved_route in resolved_routes:
                    route_node_ids = [
                        mesh_device.get_fabric_node_id(
                            _build_mesh_coordinate(ttnn_api, coordinates)
                        )
                        for coordinates in resolved_route.device_chain
                    ]
                    hop_directions = [
                        active_route_cache.resolve_direction(
                            ttnn_api,
                            mesh_device,
                            fabric_config,
                            route_node_ids[hop_index],
                            route_node_ids[hop_index + 1],
                        )
                        for hop_index in range(len(route_node_ids) - 1)
                    ]
                    if any(
                        direction != hop_directions[0]
                        for direction in hop_directions[1:]
                    ):
                        raise ValueError(
                            "FABRIC_1D routes require one forwarding direction"
                        )
                    route_directions.append(hop_directions[0])
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
                for route_index, remote_slot in enumerate(route_remote_slots):
                    if remote_slot >= len(route_directions):
                        continue
                    route = routes[route_index]
                    if route.local_device != device_coordinates:
                        continue
                    if node_coordinates not in route.source_nodes:
                        continue
                    if route.route_index in active_route_indices:
                        raise ValueError(
                            "active fabric routes must have distinct route indices"
                        )
                    active_route_indices.add(route.route_index)
                    route_slots[route.route_index] = remote_connection_slots[
                        remote_slot
                    ]
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

                runtime_prefix = (
                    len(connection_node_ids),
                    *route_slots,
                    *destination_device_ids,
                    *destination_mesh_ids,
                    *destination_hop_counts,
                )
                connection_requests = []
                for connection_index, connection_node_id in enumerate(
                    connection_node_ids
                ):
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
                        )
                    )
                manager_requests.append(
                    _FabricManagerRequest(
                        kernel_index=kernel_index,
                        node_coordinates=node_coordinates,
                        runtime_prefix=runtime_prefix,
                        connections=tuple(connection_requests),
                    )
                )

    selected_links = _assign_distinct_links(manager_requests)
    manager_bindings = []
    for manager_index, manager_request in enumerate(manager_requests):
        connection_bindings = tuple(
            _FabricConnectionBinding(
                connection_node_id=connection.connection_node_id,
                link_index=selected_links[(manager_index, connection_index)],
            )
            for connection_index, connection in enumerate(manager_request.connections)
        )
        manager_bindings.append(
            _FabricManagerBinding(
                kernel_index=manager_request.kernel_index,
                node_coordinates=manager_request.node_coordinates,
                runtime_prefix=manager_request.runtime_prefix,
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
        managers=tuple(manager_bindings),
    )


def apply_fabric_target_binding_plan(
    ttnn_api: Any,
    program_descriptor: Any,
    plan: FabricTargetBindingPlan,
    device_coordinates: Tuple[int, ...],
) -> None:
    """Apply a validated target-binding plan to one program descriptor."""
    for manager in plan.managers:
        kernel_descriptor = program_descriptor.kernels[manager.kernel_index]
        node_x, node_y = manager.node_coordinates
        # TTNN stores per-core runtime arguments sparsely, so a manager core
        # without operation-owned arguments has no descriptor entry.
        try:
            caller_runtime_args = list(kernel_descriptor.runtime_args[node_x][node_y])
        except LookupError:
            caller_runtime_args = []
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
        # Generated code indexes its route metadata and connection records from
        # zero. Keep that ABI stable and append operation-owned arguments.
        runtime_args = [
            *manager.runtime_prefix,
            *fabric_args,
            *caller_runtime_args,
        ]
        kernel_descriptor.runtime_args[node_x][node_y] = runtime_args
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


def configure_routing_plane_runtime_args(
    ttnn_api: Any,
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    mesh_device: Any,
    device_coordinates: Tuple[int, ...],
    grid_cols: int,
    grid_rows: int,
    route_cache: Optional[FabricRouteCache] = None,
) -> None:
    """Plan and apply routing-plane target bindings for one logical device."""
    if not any(kernel_fabric_routes):
        if len(kernel_fabric_routes) != len(program_descriptor.kernels):
            raise ValueError(
                "kernel_fabric_routes must have one entry per kernel descriptor"
            )
        return
    plan = build_fabric_target_binding_plan(
        ttnn_api=ttnn_api,
        program_descriptor=program_descriptor,
        kernel_fabric_routes=kernel_fabric_routes,
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
    "FabricRouteCache",
    "FabricRouteSpec",
    "FabricTargetBindingPlan",
    "apply_fabric_target_binding_plan",
    "build_fabric_target_binding_plan",
    "configure_routing_plane_runtime_args",
]
