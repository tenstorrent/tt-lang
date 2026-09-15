# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent SRAM tensor declarations and TTNN submission ownership."""

from dataclasses import dataclass
import threading

from ._persistent_storage import (
    StorageOwner,
    StorageReference,
    with_persistent_storage,
)


class _TTNNStorageBackend:
    def __init__(self, device, api, queue_ids, sub_device_ids):
        self.device = device
        self.api = api
        self._is_initialized = getattr(device, "is_initialized", None)
        self._get_manager_id = getattr(device, "get_active_sub_device_manager_id", None)
        if not callable(getattr(device, "id", None)):
            raise TypeError("persistent storage requires a TTNN mesh device")
        if callable(self._is_initialized) and not self._is_initialized():
            raise RuntimeError("persistent storage requires an open mesh device")
        self.device_id = device.id()
        self.manager_id = (
            self._get_manager_id() if callable(self._get_manager_id) else None
        )
        get_queue_count = getattr(device, "num_hw_cqs", None)
        queue_count = get_queue_count() if callable(get_queue_count) else 1
        self.queue_ids = tuple(queue_ids)
        if (
            not self.queue_ids
            or len(set(self.queue_ids)) != len(self.queue_ids)
            or any(
                type(index) is not int or not 0 <= index < queue_count
                for index in self.queue_ids
            )
            or 0 not in self.queue_ids
        ):
            raise ValueError(
                "queue_ids must be distinct valid queues including queue 0"
            )
        get_sub_device_ids = getattr(device, "get_sub_device_ids", None)
        active = tuple(get_sub_device_ids()) if callable(get_sub_device_ids) else None
        if sub_device_ids is None:
            selected = active if active is not None else (api.SubDeviceId(0),)
        else:
            selected = tuple(sub_device_ids)
        if not selected or (
            active is not None and any(value not in active for value in selected)
        ):
            raise ValueError("completion requires nonempty active sub-device IDs")
        if any(value in selected[:index] for index, value in enumerate(selected)):
            raise ValueError("completion sub-device IDs must be distinct")
        self.sub_device_ids = selected

    def validate(self, resources):
        device_closed = callable(self._is_initialized) and not self._is_initialized()
        if device_closed or self.device.id() != self.device_id:
            raise RuntimeError("persistent storage device is closed or replaced")
        if self.manager_id is not None and self._get_manager_id() != self.manager_id:
            raise RuntimeError("persistent storage sub-device configuration changed")
        if any(not resource.is_allocated() for resource in resources):
            raise RuntimeError("persistent tensor backing has been released externally")

    def order_after(self, completion):
        for queue_id in self.queue_ids:
            for event in completion:
                self.api.wait_for_event(cq_id=queue_id, mesh_event=event)

    def record_completion(self):
        return tuple(
            self.api.record_event(
                self.device,
                cq_id=queue_id,
                sub_device_ids=list(self.sub_device_ids),
            )
            for queue_id in self.queue_ids
        )

    def wait(self, completion):
        self.validate(())
        for event in completion:
            self.api.event_synchronize(event)

    def recover(self):
        self.validate(())
        for queue_id in self.queue_ids:
            self.api.synchronize_device(
                self.device,
                cq_id=queue_id,
                sub_device_ids=list(self.sub_device_ids),
            )

    def release(self, resource):
        self.validate(())
        if not resource.is_allocated():
            raise RuntimeError("persistent tensor backing was released externally")
        self.api.deallocate(resource)

    def aliases(self, resource, candidate):
        if resource is candidate:
            return True
        if type(resource) is not type(candidate):
            return False
        required_methods = (
            "buffer_address",
            "device",
            "is_allocated",
            "is_per_core_allocated",
            "memory_config",
        )
        if any(
            not callable(getattr(candidate, method, None))
            for method in required_methods
        ):
            return False
        try:
            if (
                not candidate.is_allocated()
                or candidate.device().id() != self.device_id
            ):
                return False
            if (
                tuple(candidate.shape) != tuple(resource.shape)
                or tuple(candidate.padded_shape) != tuple(resource.padded_shape)
                or candidate.dtype != resource.dtype
                or candidate.layout != resource.layout
                or candidate.tile != resource.tile
                or candidate.memory_config() != resource.memory_config()
                or candidate.is_per_core_allocated() != resource.is_per_core_allocated()
            ):
                return False
            if not resource.is_per_core_allocated():
                return candidate.buffer_address() == resource.buffer_address()
            shard_grid = resource.memory_config().shard_spec.grid
            cores = sorted(
                self.api.corerange_to_cores(shard_grid),
                key=lambda core: (core.y, core.x),
            )
            resource_devices = tuple(resource.device_coords())
            candidate_devices = tuple(candidate.device_coords())
            if tuple(map(tuple, resource_devices)) != tuple(
                map(tuple, candidate_devices)
            ):
                return False
            for device_coordinate in resource_devices:
                for core in cores:
                    if resource.experimental_per_core_buffer_address(
                        device_coordinate, core
                    ) != candidate.experimental_per_core_buffer_address(
                        device_coordinate, core
                    ):
                        return False
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class _TensorDeclaration:
    spec: object
    addressing: str
    initialize: str


class SRAMStorage:
    """Own SRAM tensors across launches until completion-aware close.

    queue_ids and sub_device_ids cover every device access through this owner.
    Device close, reset, mesh reshape, and sub-device manager changes must not
    race storage operations. External launchers use submit() and must not retain
    raw tensor arguments or deallocate the owner's backing storage.
    """

    def __init__(self, *, device, queue_ids=(0,), sub_device_ids=None):
        import ttnn

        self._api = ttnn
        self._backend = _TTNNStorageBackend(device, ttnn, queue_ids, sub_device_ids)
        self._owner = StorageOwner((), self._backend, declared=True)
        self._declarations = []
        self._lock = threading.RLock()

    def tensor(
        self,
        *,
        shape,
        shard_shape,
        cores,
        dtype,
        layout,
        addressing="uniform",
        initialize="zeros",
        sharding=None,
    ):
        """Declare a BF16/FP32 tiled tensor using TTNN shard-layout validation."""
        api = self._api
        with self._lock:
            self._owner.require_declared()
            self._backend.validate(())
            shape = tuple(shape)
            shard_shape = tuple(shard_shape)
            cores = tuple(tuple(core) for core in cores)
            if (
                not 2 <= len(shape) <= 4
                or len(shard_shape) != 2
                or any(
                    type(value) is not int or value <= 0
                    for value in (*shape, *shard_shape)
                )
                or any(value % 32 for value in (*shape[-2:], *shard_shape))
            ):
                raise ValueError(
                    "tensor and shard extents require positive full-tile dimensions"
                )
            if (
                not cores
                or len(set(cores)) != len(cores)
                or any(
                    len(core) != 2
                    or any(type(value) is not int or value < 0 for value in core)
                    for core in cores
                )
            ):
                raise ValueError("cores must be distinct nonnegative coordinate pairs")
            if dtype not in (api.bfloat16, api.float32) or layout != api.TILE_LAYOUT:
                raise ValueError("persistent tensors require BF16/FP32 TILE layout")
            if addressing not in ("uniform", "per-core"):
                raise ValueError("addressing must be uniform or per-core")
            if initialize not in ("zeros", "uninitialized"):
                raise ValueError("initialize must be zeros or uninitialized")
            grid = self._backend.device.compute_with_storage_grid_size()
            if any(core[0] >= grid.x or core[1] >= grid.y for core in cores):
                raise ValueError("persistent tensor core is outside the device grid")
            core_ranges = api.CoreRangeSet(
                [
                    api.CoreRange(api.CoreCoord(*core), api.CoreCoord(*core))
                    for core in cores
                ]
            )
            rectangle = core_ranges.bounding_box()
            extent = rectangle.grid_size()
            complete_rectangle = extent.x * extent.y == len(cores)
            if complete_rectangle:
                core_ranges = api.CoreRangeSet([rectangle])
            shard_spec = api.ShardSpec(
                core_ranges, shard_shape, api.ShardOrientation.ROW_MAJOR
            )
            sharding = (
                api.TensorMemoryLayout.HEIGHT_SHARDED if sharding is None else sharding
            )
            if sharding not in (
                api.TensorMemoryLayout.HEIGHT_SHARDED,
                api.TensorMemoryLayout.WIDTH_SHARDED,
                api.TensorMemoryLayout.BLOCK_SHARDED,
            ):
                raise ValueError("persistent tensors require a sharded memory layout")
            if (
                sharding == api.TensorMemoryLayout.BLOCK_SHARDED
                and not complete_rectangle
            ):
                raise ValueError("block-sharded cores must form a complete rectangle")
            spec = api.TensorSpec(
                api.Shape(shape), dtype, layout, sharding, shard_spec, api.BufferType.L1
            )
            self._declarations.append(_TensorDeclaration(spec, addressing, initialize))
            return self._owner.declare_reference()

    def allocate(self):
        """Allocate all declarations and initialize once before publishing bindings."""
        with self._lock:
            if not self._declarations:
                raise ValueError("storage has no tensor declarations")

            def build_resources(retain):
                resources = []
                for declaration in self._declarations:
                    spec = declaration.spec
                    config = spec.memory_config
                    config.experimental_set_per_core_allocation(
                        declaration.addressing == "per-core"
                    )
                    resource = self._api.empty(
                        spec.shape,
                        dtype=spec.dtype,
                        layout=spec.layout,
                        device=self._backend.device,
                        memory_config=config,
                    )
                    retain(resource)
                    resources.append(resource)
                for resource, declaration in zip(resources, self._declarations):
                    if declaration.initialize == "zeros":
                        spec = declaration.spec
                        self._api.full(
                            spec.shape,
                            0.0,
                            dtype=spec.dtype,
                            layout=spec.layout,
                            device=self._backend.device,
                            optional_tensor=resource,
                        )

            self._owner.allocate(build_resources)

    def submit(self, function, *args, **kwargs):
        """Submit an external launcher under the declared completion contract."""
        if not any(self._owner.owns(value) for value in (*args, *kwargs.values())):
            raise ValueError(
                "external submission must borrow a tensor from this storage"
            )
        return with_persistent_storage(function)(*args, **kwargs)

    def close(self):
        with self._lock:
            self._owner.close()

    def __enter__(self):
        self._owner.require_open()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
