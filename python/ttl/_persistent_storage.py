# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ownership and completion transactions for persistent tensor arguments."""

from contextlib import ExitStack
from enum import Enum, auto
from functools import wraps
import threading
from typing import Callable, Protocol


class StorageBackend(Protocol):
    """Own one submission domain, including its queues and remote users.

    Validation rejects invalid device generations and incompatible resources.
    Completion records cover every submitted access to borrowed resources.
    Recovery establishes completion even if recording failed. Release either
    succeeds or raises without releasing the resource, permitting close retry.
    """

    def validate(self, resources: tuple[object, ...]) -> None: ...

    def order_after(self, completion: object) -> None: ...

    def record_completion(self) -> object: ...

    def wait(self, completion: object) -> None: ...

    def recover(self) -> None: ...

    def release(self, resource: object) -> None: ...

    def is_same_allocation(self, resource: object, candidate: object) -> bool: ...


class _State(Enum):
    DECLARED = auto()
    READY = auto()
    CLOSING = auto()
    CLOSED = auto()


_pending_owners: set["StorageOwner"] = set()
_pending_lock = threading.Lock()
_submission_thread = threading.local()


class StorageOwner:
    """Retain initialized tensor resources until explicit, completion-aware close.

    The caller transfers exclusive release authority for distinct allocations.
    Aliased arguments use references to the same owner and resource index.
    """

    def __init__(
        self, resources: tuple[object, ...], backend: StorageBackend, *, declared=False
    ):
        if not resources and not declared:
            raise ValueError("persistent storage requires at least one resource")
        if len({id(resource) for resource in resources}) != len(resources):
            raise ValueError("persistent resources must have distinct owners")
        self._resources = list(resources)
        self._declared_resource_count = len(resources)
        self._backend = backend
        self._lock = threading.RLock()
        self._state = _State.DECLARED if declared else _State.READY
        self._completion = None
        self._unknown_completion = False
        self._submitting = False

    def reference(self, index: int) -> "StorageReference":
        with self._lock:
            self._require_ready()
            self._validate_reference_index(index, len(self._resources))
            return StorageReference(self, index)

    def declare_reference(self) -> "StorageReference":
        with self._lock:
            if self._state is not _State.DECLARED:
                raise RuntimeError("persistent declarations are complete")
            index = self._declared_resource_count
            self._declared_resource_count += 1
            return StorageReference(self, index)

    def allocate(self, build_resources: Callable[[Callable[[object], None]], None]):
        """Publish all retained resources together after initialization completes."""
        with self._lock:
            if self._state is not _State.DECLARED:
                raise RuntimeError(
                    "storage allocation requires unallocated declarations"
                )
            self._backend.validate(())
            with _pending_lock:
                _pending_owners.add(self)

            def retain(resource: object):
                if any(resource is existing for existing in self._resources):
                    raise ValueError("persistent resources must have distinct owners")
                self._resources.append(resource)

            try:
                build_resources(retain)
                if len(self._resources) != self._declared_resource_count:
                    raise ValueError(
                        "persistent storage requires one resource per declaration"
                    )
                self._unknown_completion = True
                completion = self._backend.record_completion()
                if completion is None:
                    raise RuntimeError("backend returned no completion record")
                self._backend.wait(completion)
                self._backend.validate(tuple(self._resources))
                self._unknown_completion = False
                self._state = _State.READY
            except BaseException:
                self._unknown_completion = True
                self._state = _State.CLOSING
                self._close_locked()
                self._state = _State.DECLARED
                raise

    def owns(self, reference: object) -> bool:
        return isinstance(reference, StorageReference) and reference._owner is self

    def require_open(self):
        with self._lock:
            if self._state in (_State.CLOSING, _State.CLOSED):
                raise RuntimeError("persistent storage is closing or closed")

    def require_declared(self):
        with self._lock:
            if self._state is not _State.DECLARED:
                raise RuntimeError("persistent declarations are complete")

    @staticmethod
    def _validate_reference_index(index: int, extent: int):
        if type(index) is not int or not 0 <= index < extent:
            raise ValueError("persistent resource index is out of range")

    def _require_ready(self):
        if self._state is _State.DECLARED:
            raise RuntimeError("persistent storage has not been allocated")
        if self._state is not _State.READY:
            raise RuntimeError("persistent storage is closing or closed")
        if self._unknown_completion:
            raise RuntimeError("persistent storage requires completion recovery")

    def close(self):
        """Reject new submissions and release resources after all uses complete.

        Failure retains unreleased resources and permits retrying close.
        """
        with self._lock:
            if self._submitting:
                raise RuntimeError("cannot close storage inside its submission")
            if self._state is _State.CLOSED:
                return
            self._state = _State.CLOSING
            self._close_locked()

    def _close_locked(self):
        if self._unknown_completion:
            self._backend.recover()
            self._unknown_completion = False
            self._completion = None
        elif self._completion is not None:
            self._backend.wait(self._completion)
            self._completion = None
        while self._resources:
            self._backend.release(self._resources[-1])
            self._resources.pop()
        self._state = _State.CLOSED
        with _pending_lock:
            _pending_owners.discard(self)


class StorageReference:
    """A tensor argument retaining its allocation owner without exposing storage."""

    __slots__ = ("_owner", "_index")

    def __init__(self, owner: StorageOwner, index: int):
        self._owner = owner
        self._index = index


def with_persistent_storage(function):
    """Normalize owned arguments while serializing their submission transactions.

    Unknown operation effects conservatively serialize uses of each owner.
    Backend dependencies order device accesses without assuming host return
    means completion. The result preserves directly returned argument aliases.
    """

    @wraps(function)
    def invoke(*args, **kwargs):
        references = [
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, StorageReference)
        ]
        if not references:
            return function(*args, **kwargs)
        if getattr(_submission_thread, "active", False):
            raise RuntimeError("nested persistent submissions are not supported")
        owners = sorted({reference._owner for reference in references}, key=id)
        with ExitStack() as locks:
            for owner in owners:
                locks.enter_context(owner._lock)
            for owner in owners:
                owner._require_ready()
                owner._backend.validate(tuple(owner._resources))
            for owner in owners:
                if owner._completion is not None:
                    owner._backend.order_after(owner._completion)

            def resolve(value):
                if isinstance(value, StorageReference):
                    return value._owner._resources[value._index]
                return value

            resolved_args = tuple(resolve(value) for value in args)
            resolved_kwargs = {name: resolve(value) for name, value in kwargs.items()}
            with _pending_lock:
                _pending_owners.update(owners)
            for owner in owners:
                owner._submitting = True
            _submission_thread.active = True
            try:
                try:
                    result = function(*resolved_args, **resolved_kwargs)
                finally:
                    # A throwing launcher may already have submitted device work.
                    for owner in owners:
                        owner._unknown_completion = True
                    for owner in owners:
                        completion = owner._backend.record_completion()
                        if completion is None:
                            raise RuntimeError("backend returned no completion record")
                        owner._completion = completion
                        owner._unknown_completion = False
            finally:
                for owner in owners:
                    owner._submitting = False
                _submission_thread.active = False
            owned_allocations = [
                (reference._owner._backend, resolve(reference), reference)
                for reference in references
            ]

            def restore(value):
                for backend, resource, reference in owned_allocations:
                    if backend.is_same_allocation(resource, value):
                        return reference
                if type(value) is tuple:
                    return tuple(restore(element) for element in value)
                if type(value) is list:
                    return [restore(element) for element in value]
                if type(value) is dict:
                    return {name: restore(element) for name, element in value.items()}
                return value

            return restore(result)

    return invoke
