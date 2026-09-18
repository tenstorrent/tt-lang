# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent SRAM tensor declarations and TTNN submission ownership."""

from dataclasses import dataclass
from itertools import product
import threading
import time

from ._persistent_storage import (
    StorageOwner,
    StorageReference,
    with_persistent_storage,
)
from ._sram_requirements import (
    PersistentSRAMDeclaration,
    PreparedSRAMStorage,
    SRAMAddressing,
    SRAMUseKind,
    prepare_persistent_storage,
)


def _align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _core_ranges(api, cores):
    ranges = api.CoreRangeSet(
        [api.CoreRange(api.CoreCoord(*core), api.CoreCoord(*core)) for core in cores]
    )
    rectangle = ranges.bounding_box()
    extent = rectangle.grid_size()
    if extent.x * extent.y == len(cores):
        return api.CoreRangeSet([rectangle])
    return ranges


def _storage_spec(api, cores, num_bytes, addressing):
    aligned_bytes = _align_up(num_bytes, 32)
    elements_per_core = aligned_bytes // 4
    shard_spec = api.ShardSpec(
        _core_ranges(api, cores),
        (1, elements_per_core),
        api.ShardOrientation.ROW_MAJOR,
    )
    spec = api.TensorSpec(
        api.Shape((len(cores), elements_per_core)),
        api.float32,
        api.ROW_MAJOR_LAYOUT,
        api.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_spec,
        api.BufferType.L1,
    )
    if addressing is SRAMAddressing.PER_CORE:
        spec.memory_config.experimental_set_per_core_allocation(True)
    return spec


def _validate_joint_operation(compiled_kernel):
    unsupported = []
    for attribute in (
        "dfb_reconfiguration_plan",
        "runtime_resource_factory",
        "mesh_program_placements",
        "device_domain",
    ):
        if getattr(compiled_kernel, attribute, None) is not None:
            unsupported.append(attribute)
    if any(getattr(compiled_kernel, "kernel_fabric_routes", ())):
        unsupported.append("kernel_fabric_routes")
    for attribute in (
        "num_pipe_sync_semaphores",
        "pipe_sram_scratch_bytes",
        "num_pipe_global_semaphores",
        "num_dfb_resets",
    ):
        if int(getattr(compiled_kernel, attribute, 0)) != 0:
            unsupported.append(attribute)
    if unsupported:
        raise ValueError(
            "joint SRAM placement does not model operation resources: "
            + ", ".join(unsupported)
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

    def is_same_allocation(self, resource, candidate):
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
    storage: PersistentSRAMDeclaration


class _PreparedStorageTensor:
    """Provide declaration metadata to compilation before device reservation."""

    __ttlang_prepared_tensor__ = True

    def __init__(self, declaration_index, declaration, device, api):
        self.declaration_index = declaration_index
        self._declaration = declaration
        self._device = device
        self._api = api
        self.shape = declaration.spec.shape
        self.padded_shape = declaration.spec.shape
        self.dtype = declaration.spec.dtype
        self.layout = declaration.spec.layout

    def storage_type(self):
        return self._api.StorageType.DEVICE

    def memory_config(self):
        return self._declaration.spec.memory_config

    def get_tile(self):
        return self._declaration.spec.tile

    def device(self):
        return self._device

    def is_allocated(self):
        return False

    def is_per_core_allocated(self):
        return self._declaration.addressing == "per-core"

    def device_coords(self):
        device_shape = getattr(self._device, "shape", None)
        if device_shape is None:
            return (self._api.MeshCoordinate(0, 0),)
        dimensions = tuple(int(dimension) for dimension in device_shape)
        return tuple(
            self._api.MeshCoordinate(*coordinate)
            for coordinate in product(*(range(dimension) for dimension in dimensions))
        )

    def buffer_address(self):
        return 0

    def experimental_per_core_buffer_address(self, device_coordinate, core):
        del device_coordinate, core
        return 0


@dataclass(frozen=True)
class PreparedSRAMOperationCall:
    """Record one compiled operation specialization prepared for joint placement."""

    operation: object
    compiled_kernel: object
    requirements: object
    runtime_args: tuple


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
        self._preparation_tensors = []
        self._prepared_operations = []
        self._preparation_seconds = 0.0
        self._allocation_metrics = None
        self._lock = threading.RLock()

    def tensor(
        self,
        *,
        shape,
        shard_shape,
        cores,
        dtype,
        layout,
        sharding,
        addressing="uniform",
        initialize="zeros",
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
            if addressing == "per-core":
                spec.memory_config.experimental_set_per_core_allocation(True)
            element_bytes = 2 if dtype == api.bfloat16 else 4
            storage = PersistentSRAMDeclaration(
                extent_bytes=shard_shape[0] * shard_shape[1] * element_bytes,
                alignment_bytes=api.get_l1_alignment(),
                cores=cores,
                addressing=(
                    SRAMAddressing.UNIFORM
                    if addressing == "uniform"
                    else SRAMAddressing.PER_CORE
                ),
            )
            self._declarations.append(
                _TensorDeclaration(spec, addressing, initialize, storage)
            )
            self._preparation_tensors.append(
                _PreparedStorageTensor(
                    len(self._declarations) - 1,
                    self._declarations[-1],
                    self._backend.device,
                    api,
                )
            )
            return self._owner.declare_reference()

    def prepare_operation(self, operation, *args, **kwargs):
        """Compile one operation specialization and collect its SRAM requirements."""
        with self._lock:
            started = time.perf_counter()
            self._owner.require_declared()
            prepare = getattr(operation, "_ttlang_prepare_operation", None)
            if not callable(prepare):
                raise TypeError("operation must be defined with @ttl.operation")
            argument_values = (*args, *kwargs.values())
            if not any(self._owner.owns(value) for value in argument_values):
                raise ValueError(
                    "prepared operation must borrow a tensor from this storage"
                )

            def resolve(value):
                if not isinstance(value, StorageReference):
                    return value
                if not self._owner.owns(value):
                    raise ValueError(
                        "operation preparation cannot borrow another storage owner"
                    )
                return self._preparation_tensors[value._index]

            prepared_args = tuple(resolve(value) for value in args)
            prepared_kwargs = {name: resolve(value) for name, value in kwargs.items()}
            compiled_kernel, runtime_args = prepare(*prepared_args, **prepared_kwargs)
            _validate_joint_operation(compiled_kernel)
            requirements = compiled_kernel.prepare_sram_requirements(*runtime_args)
            prepared = PreparedSRAMOperationCall(
                operation, compiled_kernel, requirements, tuple(runtime_args)
            )
            if any(
                existing.compiled_kernel is compiled_kernel
                for existing in self._prepared_operations
            ):
                raise ValueError(
                    "operation specialization is already prepared for this storage"
                )
            self._prepared_operations.append(prepared)
            self._preparation_seconds += time.perf_counter() - started
            return prepared

    def requirements(self) -> PreparedSRAMStorage:
        """Return immutable movable requirements before physical reservation."""
        with self._lock:
            self._owner.require_declared()
            return prepare_persistent_storage(
                tuple(declaration.storage for declaration in self._declarations)
            )

    def allocate(self):
        """Allocate all declarations and initialize once before publishing bindings."""
        with self._lock:
            if not self._declarations:
                raise ValueError("storage has no tensor declarations")
            if self._prepared_operations:
                self._allocate_joint()
                return

            def build_resources(retain):
                self.requirements()
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

    def _allocate_joint(self):
        from ._sram_placement import plan_joint_sram
        from .kernel_runner import (
            PreparedSRAMResources,
            get_min_remaining_l1_for_device,
        )

        experimental_api = getattr(self._api, "experimental", None)
        create_view = getattr(experimental_api, "create_sharded_tensor_view", None)
        if not callable(create_view):
            raise RuntimeError(
                "joint SRAM allocation requires owner-retaining TTNN tensor views"
            )
        started = time.perf_counter()
        budget_bytes = get_min_remaining_l1_for_device(
            self._backend.device, ttnn_api=self._api
        )
        plan = plan_joint_sram(
            self.requirements(),
            tuple(prepared.requirements for prepared in self._prepared_operations),
            budget_bytes=budget_bytes,
        )
        operation_resources = [
            {"uniform": None, "cores": {}, "controls": []}
            for _ in self._prepared_operations
        ]
        pool_resources = []
        program_metrics = []
        program_preparation_seconds = 0.0
        prepared_metrics = None

        def build_resources(retain):
            persistent_resources = [None] * len(self._declarations)
            for pool in plan.pools:
                pool_cores = tuple(
                    sorted({location.core for location in pool.locations})
                )
                pool_spec = _storage_spec(
                    self._api,
                    pool_cores,
                    pool.reservation_bytes_per_location,
                    pool.addressing,
                )
                pool_resource = self._api.empty(
                    pool_spec.shape,
                    dtype=pool_spec.dtype,
                    layout=pool_spec.layout,
                    device=self._backend.device,
                    memory_config=pool_spec.memory_config,
                )
                retain(pool_resource)
                pool_resources.append(
                    (
                        pool_resource,
                        len(pool.locations),
                        pool.reservation_bytes_per_location,
                    )
                )
                for placement in pool.placements:
                    region = placement.region
                    if region.is_persistent:
                        declaration = self._declarations[region.requirement_index]
                        resource = create_view(
                            pool_resource, declaration.spec, placement.offset
                        )
                        retain(resource, alias=True)
                        persistent_resources[region.requirement_index] = resource
                        continue

                    prepared = self._prepared_operations[region.operation_index]
                    requirement = region.requirement
                    arena_cores = tuple(
                        sorted(
                            {
                                location.core
                                for domain in requirement.address_domains
                                for location in domain.locations
                            }
                        )
                    )
                    arena_spec = _storage_spec(
                        self._api,
                        arena_cores,
                        requirement.extent_bytes,
                        requirement.addressing,
                    )
                    arena = create_view(pool_resource, arena_spec, placement.offset)
                    retain(arena, alias=True)
                    resources = operation_resources[region.operation_index]
                    has_core_layouts = any(
                        config.sram_core_layouts
                        for config in prepared.compiled_kernel.cb_configs
                    )
                    if has_core_layouts:
                        for core in arena_cores:
                            resources["cores"][core] = arena
                    else:
                        if resources["uniform"] is not None:
                            raise RuntimeError(
                                "uniform operation has multiple prepared SRAM arenas"
                            )
                        resources["uniform"] = arena

                    control_end = max(
                        (
                            use.byte_offset + use.byte_size
                            for use in prepared.requirements.uses
                            if use.requirement_index == region.requirement_index
                            and use.kind is SRAMUseKind.DFB_CONTROL
                        ),
                        default=0,
                    )
                    if control_end:
                        control_spec = _storage_spec(
                            self._api,
                            arena_cores,
                            control_end,
                            requirement.addressing,
                        )
                        control_tensor = create_view(
                            pool_resource, control_spec, placement.offset
                        )
                        retain(control_tensor, alias=True)
                        resources["controls"].append(control_tensor)

            if any(resource is None for resource in persistent_resources):
                raise RuntimeError(
                    "joint SRAM placement did not bind every persistent declaration"
                )
            prepared_bindings = []
            for prepared, resources in zip(
                self._prepared_operations, operation_resources
            ):
                prepared_bindings.append(
                    PreparedSRAMResources(
                        requirements=prepared.requirements,
                        uniform_arena=resources["uniform"],
                        core_arenas=tuple(sorted(resources["cores"].items())),
                        control_tensors=tuple(resources["controls"]),
                    )
                )

            nonlocal program_preparation_seconds
            program_started = time.perf_counter()
            for prepared, binding in zip(self._prepared_operations, prepared_bindings):
                runtime_args = tuple(
                    (
                        persistent_resources[value.declaration_index]
                        if isinstance(value, _PreparedStorageTensor)
                        else value
                    )
                    for value in prepared.runtime_args
                )
                program_metrics.append(
                    prepared.compiled_kernel.prepare_device_program(
                        *runtime_args, prepared_sram_resources=binding
                    )
                )
            program_preparation_seconds += time.perf_counter() - program_started

            for resource, declaration in zip(persistent_resources, self._declarations):
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
            for prepared, binding in zip(self._prepared_operations, prepared_bindings):
                self._owner.bind_operation(prepared.compiled_kernel, binding)

            actual_reservation_bytes = 0
            for resource, location_count, planned_bytes in pool_resources:
                get_page_size = getattr(resource, "buffer_aligned_page_size", None)
                bytes_per_location = (
                    int(get_page_size()) if callable(get_page_size) else planned_bytes
                )
                actual_reservation_bytes += bytes_per_location * location_count
            if actual_reservation_bytes < plan.metrics.required_peak_bytes:
                raise RuntimeError(
                    "allocated SRAM pools are smaller than the required live data"
                )
            nonlocal prepared_metrics
            prepared_metrics = {
                "required_peak_bytes": plan.metrics.required_peak_bytes,
                "planned_reservation_bytes": plan.metrics.reservation_bytes,
                "actual_reservation_bytes": actual_reservation_bytes,
                "fragmentation_bytes": (
                    actual_reservation_bytes - plan.metrics.required_peak_bytes
                ),
                "efficiency": (
                    plan.metrics.required_peak_bytes / actual_reservation_bytes
                    if actual_reservation_bytes
                    else 1.0
                ),
                "separate_planned_peak_bytes": (
                    plan.metrics.separate_planned_peak_bytes
                ),
                "programs": tuple(
                    {
                        "max_program_config_size_bytes": int(
                            result.max_program_config_size_bytes
                        ),
                        "max_kernel_binary_size_bytes": int(
                            result.max_kernel_binary_size_bytes
                        ),
                    }
                    for result in program_metrics
                ),
            }
            return tuple(persistent_resources)

        self._owner.allocate(build_resources)
        self._joint_plan = plan
        allocation_seconds = time.perf_counter() - started
        self._preparation_seconds += program_preparation_seconds
        self._allocation_metrics = dict(prepared_metrics)
        self._allocation_metrics["preparation_seconds"] = self._preparation_seconds
        self._allocation_metrics["allocation_seconds"] = allocation_seconds

    def allocation_metrics(self):
        """Return measured joint-placement metrics after successful allocation."""
        with self._lock:
            if self._allocation_metrics is None:
                raise RuntimeError("joint SRAM storage has not been allocated")
            return dict(self._allocation_metrics)

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
