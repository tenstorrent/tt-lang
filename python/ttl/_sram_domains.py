# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validation and binding of compiler-finalized SRAM core layouts."""


def validate_core_layouts(configs, coordinates):
    if not any(config.sram_core_layouts for config in configs):
        return {}
    coordinates = set(coordinates)
    sizes = {}
    domains = {}
    owners = {}
    if any(
        config.l1_offset is None or config.storage_index is None for config in configs
    ):
        raise ValueError(
            "SRAM core layouts require finalized control and storage identities"
        )
    control_end = max(config.l1_offset + 8 for config in configs)
    for config in configs:
        layouts = {layout.node: layout for layout in config.sram_core_layouts}
        if len(layouts) != len(config.sram_core_layouts) or set(layouts) != coordinates:
            raise ValueError(
                "SRAM core layouts must cover each participating core exactly once"
            )
        domain_layouts = {}
        for node, layout in layouts.items():
            if len(node) != 2 or any(
                type(value) is not int or value < 0 for value in node
            ):
                raise ValueError(
                    "SRAM core coordinates must contain two nonnegative integers"
                )
            if type(layout.domain) is not int or layout.domain < 0:
                raise ValueError("SRAM allocation domain must be a nonnegative integer")
            if node in domains and domains[node] != layout.domain:
                raise ValueError("inconsistent SRAM allocation domains for one core")
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
                    "SRAM cores in one allocation domain must share one layout"
                )
            domain_layouts[layout.domain] = placement
            if layout.arena_bytes < control_end:
                raise ValueError("SRAM core arena does not cover its control records")
            if node in sizes and sizes[node] != layout.arena_bytes:
                raise ValueError("inconsistent SRAM arena sizes for one core")
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
                    raise ValueError("SRAM payload is outside its core arena")
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
                "DFBs sharing one SRAM storage owner must share its core layouts"
            )
        owners[config.storage_index] = owner_layout
    return sizes


def payload_defines(configs, coordinate):
    result = []
    for config in configs:
        if config.l1_payload_offset is None:
            continue
        layout = next(
            layout for layout in config.sram_core_layouts if layout.node == coordinate
        )
        result.append(
            (
                f"TTLANG_SRAM_DFB_{config.dfb_index}_PAYLOAD_OFFSET",
                str(layout.payload_offset - config.l1_offset),
            )
        )
    return result


def core_domains(configs):
    groups = {}
    for layout in configs[0].sram_core_layouts:
        groups.setdefault(layout.domain, []).append(layout.node)
    return [tuple(sorted(nodes)) for _, nodes in sorted(groups.items())]


def validate_receiver_targets(kernel_specs, configs):
    for spec in kernel_specs:
        if len(spec.sram_receiver_targets) != len(
            spec.pipe_computed_address_dfb_indices
        ):
            raise ValueError(
                "per-core SRAM receiver arguments require complete destination identities"
            )
        for target, index in zip(
            spec.sram_receiver_targets, spec.pipe_computed_address_dfb_indices
        ):
            if target.dfb_index != index or not 0 <= index < len(configs):
                raise ValueError("SRAM receiver target references an invalid DFB")
            config = configs[index]
            layout = next(
                (
                    layout
                    for layout in config.sram_core_layouts
                    if layout.node == target.node
                ),
                None,
            )
            if layout is None or (
                config.l1_payload_offset is not None and not layout.payload_present
            ):
                raise ValueError(
                    "SRAM receiver target has no payload on its destination core"
                )


def tensor_base(ttnn_api, tensor, node, device_coordinate):
    if not tensor.is_per_core_allocated():
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
    device_coordinate = target.device or mesh_coordinate
    if config.l1_payload_offset is None:
        segment = next(
            segment
            for segment in config.storage_segments
            if target.node in segment.nodes
        )
        tensor = tensors[segment.tensor_index]
        offset = segment.byte_offset
    else:
        tensor = arenas[target.node]
        offset = next(
            layout.payload_offset
            for layout in config.sram_core_layouts
            if layout.node == target.node
        )
    return tensor_base(ttnn_api, tensor, target.node, device_coordinate) + offset
