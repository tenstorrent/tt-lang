# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validation and binding of compiler-finalized SRAM node layouts."""


def validate_node_layouts(configs, coordinates):
    if not any(config.sram_node_layouts for config in configs):
        return {}
    coordinates = set(coordinates)
    sizes = {}
    domains = {}
    owners = {}
    if any(
        config.l1_offset is None or config.storage_index is None for config in configs
    ):
        raise ValueError(
            "SRAM node layouts require finalized control and storage identities"
        )
    control_end = max(config.l1_offset + 8 for config in configs)
    for config in configs:
        layouts = {layout.node: layout for layout in config.sram_node_layouts}
        if len(layouts) != len(config.sram_node_layouts) or set(layouts) != coordinates:
            raise ValueError(
                "SRAM node layouts must cover each participating node exactly once"
            )
        domain_layouts = {}
        for node, layout in layouts.items():
            if len(node) != 2 or any(
                type(value) is not int or value < 0 for value in node
            ):
                raise ValueError(
                    "SRAM node coordinates must contain two nonnegative integers"
                )
            if type(layout.domain) is not int or layout.domain < 0:
                raise ValueError("SRAM allocation domain must be a nonnegative integer")
            if node in domains and domains[node] != layout.domain:
                raise ValueError("inconsistent SRAM allocation domains for one node")
            domains[node] = layout.domain
            placement = (
                layout.payload_offset,
                layout.payload_present,
                layout.arena_bytes,
            )
            if (
                layout.domain in domain_layouts
                and domain_layouts[layout.domain] != placement
            ):
                raise ValueError(
                    "SRAM nodes in one allocation domain must share one layout"
                )
            domain_layouts[layout.domain] = placement
            if layout.arena_bytes < control_end:
                raise ValueError("SRAM node arena does not cover its control records")
            if node in sizes and sizes[node] != layout.arena_bytes:
                raise ValueError("inconsistent SRAM arena sizes for one node")
            sizes[node] = layout.arena_bytes
            if layout.payload_present:
                if config.l1_allocation_bytes is None:
                    raise ValueError(
                        "tensor-backed SRAM storage cannot have an arena payload"
                    )
                if (
                    layout.payload_offset < control_end
                    or layout.payload_offset + config.l1_allocation_bytes
                    > layout.arena_bytes
                ):
                    raise ValueError("SRAM payload is outside its node arena")
            elif layout.payload_offset != config.l1_offset:
                raise ValueError(
                    "inactive SRAM payload must retain its control-relative zero offset"
                )
        owner_layout = (config.l1_offset, config.l1_allocation_bytes, layouts)
        if (
            config.storage_index in owners
            and owners[config.storage_index] != owner_layout
        ):
            raise ValueError(
                "DFBs sharing one SRAM storage owner must share its node layouts"
            )
        owners[config.storage_index] = owner_layout
    return sizes


def payload_defines(configs, coordinate):
    result = []
    for config in configs:
        if config.l1_payload_offset is None:
            continue
        layout = next(
            layout for layout in config.sram_node_layouts if layout.node == coordinate
        )
        result.append(
            (
                f"TTLANG_SRAM_DFB_{config.dfb_index}_PAYLOAD_OFFSET",
                str(layout.payload_offset - config.l1_offset),
            )
        )
    return result


def node_domains(configs):
    groups = {}
    for layout in configs[0].sram_node_layouts:
        groups.setdefault(layout.domain, []).append(layout.node)
    return [tuple(sorted(nodes)) for _, nodes in sorted(groups.items())]


def tensor_devices(tensor):
    try:
        return {tuple(coordinate) for coordinate in tensor.device_coords()}
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("SRAM tensor has invalid device coordinates") from error


def validate_receiver_targets(kernel_specs, configs, tensors=None):
    devices_by_tensor_index = {}
    for spec in kernel_specs:
        if len(spec.sram_receiver_targets) != len(
            spec.pipe_computed_address_dfb_indices
        ):
            raise ValueError(
                "per-node SRAM receiver arguments require complete destination identities"
            )
        for target, index in zip(
            spec.sram_receiver_targets, spec.pipe_computed_address_dfb_indices
        ):
            if target.dfb_index != index or not 0 <= index < len(configs):
                raise ValueError("SRAM receiver target references an invalid DFB")
            config = configs[index]
            receivers = target.receivers or (target,)
            if target.receivers and not any(
                receiver.node == target.node and receiver.device == target.device
                for receiver in receivers
            ):
                raise ValueError("SRAM receiver target is absent from its receiver set")
            if len({(receiver.node, receiver.device) for receiver in receivers}) != len(
                receivers
            ):
                raise ValueError("SRAM receiver target contains duplicate destinations")
            for receiver in receivers:
                layout = next(
                    (
                        layout
                        for layout in config.sram_node_layouts
                        if layout.node == receiver.node
                    ),
                    None,
                )
                if layout is None or (
                    config.l1_payload_offset is not None and not layout.payload_present
                ):
                    raise ValueError(
                        "SRAM receiver target has no payload on its destination node"
                    )
                if config.l1_payload_offset is not None:
                    continue
                segment = next(
                    (
                        segment
                        for segment in config.storage_segments
                        if segment.is_tensor_backed and receiver.node in segment.nodes
                    ),
                    None,
                )
                if segment is None:
                    raise ValueError(
                        "SRAM receiver target has no tensor-backed storage segment "
                        "on its destination node"
                    )
                if tensors is None or not receiver.device:
                    continue
                tensor_index = segment.tensor_index
                if tensor_index is None or not 0 <= tensor_index < len(tensors):
                    raise ValueError(
                        "SRAM receiver target references an invalid tensor"
                    )
                tensor = tensors[tensor_index]
                if tensor is None:
                    raise ValueError("SRAM receiver target references an absent tensor")
                if tensor_index not in devices_by_tensor_index:
                    devices_by_tensor_index[tensor_index] = tensor_devices(tensor)
                if receiver.device not in devices_by_tensor_index[tensor_index]:
                    raise ValueError(
                        "SRAM receiver tensor backing is absent from destination device"
                    )


def tensor_base(ttnn_api, tensor, node, device_coordinate):
    if not tensor.is_per_core_allocated():
        if device_coordinate is not None and device_coordinate not in tensor_devices(
            tensor
        ):
            raise ValueError("SRAM tensor has no storage on destination device")
        return int(tensor.buffer_address())
    if device_coordinate is None:
        raise ValueError(
            "independent SRAM address binding requires a logical device coordinate"
        )
    return int(
        tensor.experimental_per_core_buffer_address(
            ttnn_api.MeshCoordinate(device_coordinate), ttnn_api.CoreCoord(*node)
        )
    )


def receiver_base(ttnn_api, target, configs, arenas, tensors, mesh_coordinate):
    config = configs[target.dfb_index]
    addresses = []
    for receiver in target.receivers or (target,):
        device_coordinate = receiver.device or mesh_coordinate
        if config.l1_payload_offset is None:
            segment = next(
                segment
                for segment in config.storage_segments
                if receiver.node in segment.nodes
            )
            tensor = tensors[segment.tensor_index]
            offset = segment.byte_offset
        else:
            tensor = arenas[receiver.node]
            offset = next(
                layout.payload_offset
                for layout in config.sram_node_layouts
                if layout.node == receiver.node
            )
        addresses.append(
            tensor_base(ttnn_api, tensor, receiver.node, device_coordinate) + offset
        )
    if any(address != addresses[0] for address in addresses[1:]):
        raise ValueError(
            "SRAM multicast receivers have different physical SRAM addresses"
        )
    return addresses[0]
