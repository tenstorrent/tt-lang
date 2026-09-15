# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only ownership tests with explicitly controlled device completion."""

from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref

import pytest

from ttl._persistent_storage import StorageOwner, with_persistent_storage


class Backend:
    def __init__(self):
        self.events = []
        self.released = []
        self.ordered = []
        self.waited = []
        self.fail = None
        self.recoveries = 0
        self.release_attempts = 0

    def validate(self, resources):
        if self.fail == "validate":
            raise RuntimeError("invalid device generation")
        assert all(resource not in self.released for resource in resources)

    def order_after(self, completion):
        if self.fail == "order":
            raise RuntimeError("dependency failed")
        self.ordered.append(completion)

    def record_completion(self):
        if self.fail == "record":
            raise RuntimeError("record failed")
        event = threading.Event()
        self.events.append(event)
        return event

    def wait(self, completion):
        self.waited.append(completion)
        if self.fail == "wait":
            raise RuntimeError("wait failed")
        if not completion.wait(timeout=3):
            raise RuntimeError("test completion timed out")

    def recover(self):
        if self.fail == "recover":
            raise RuntimeError("recovery failed")
        self.recoveries += 1

    def release(self, resource):
        self.release_attempts += 1
        if self.fail == "release":
            raise RuntimeError("release failed")
        assert resource not in self.released
        self.released.append(resource)

    def is_same_allocation(self, resource, candidate):
        return resource is candidate

    def complete(self):
        for event in self.events:
            event.set()


def make_owner(count=1):
    backend = Backend()
    resources = tuple(object() for _ in range(count))
    owner = StorageOwner(resources, backend)
    return owner, backend, resources


def test_repeated_updates_share_payload_and_order_completion():
    backend = Backend()
    payload = {"value": 0}
    owner = StorageOwner((payload,), backend)
    reference = owner.reference(0)

    @with_persistent_storage
    def update(state, amount):
        state["value"] += amount
        return state

    assert update(reference, 2) is reference
    assert update(state=reference, amount=3) is reference
    assert payload["value"] == 5
    assert backend.ordered == [backend.events[0]]
    backend.complete()
    owner.close()
    assert backend.waited == [backend.events[-1]]
    assert backend.released == [payload]


def test_alias_arguments_record_and_release_once():
    owner, backend, resources = make_owner()

    @with_persistent_storage
    def consume(first, second):
        assert first is second is resources[0]

    consume(owner.reference(0), owner.reference(0))
    assert len(backend.events) == 1
    backend.complete()
    owner.close()
    owner.close()
    assert backend.released == list(resources)


def test_distinct_wrappers_for_same_initial_allocation_are_rejected():
    backend = Backend()
    backend.is_same_allocation = lambda resource, candidate: True
    with pytest.raises(ValueError, match="distinct allocations"):
        StorageOwner((object(), object()), backend)


def test_distinct_wrappers_for_same_retained_allocation_roll_back():
    backend = Backend()
    backend.is_same_allocation = lambda resource, candidate: True
    owner = StorageOwner((), backend, declared=True)
    owner.declare_reference()
    owner.declare_reference()
    first_resource = object()

    def retain_aliases(retain):
        retain(first_resource)
        retain(object())

    with pytest.raises(ValueError, match="distinct allocations"):
        owner.allocate(retain_aliases)
    assert backend.released == [first_resource]
    owner.close()


@pytest.mark.parametrize("failure", ["validate", "order"])
def test_rejected_submission_does_not_launch(failure):
    owner, backend, resources = make_owner()
    launch = with_persistent_storage(lambda value: value)
    reference = owner.reference(0)
    launch(reference)
    backend.fail = failure
    with pytest.raises(RuntimeError):
        launch(reference)
    assert len(backend.events) == 1
    backend.fail = None
    backend.complete()
    owner.close()


def test_all_owners_validate_before_any_dependency_or_launch():
    first, first_backend, _ = make_owner()
    second, second_backend, _ = make_owner()
    launch = with_persistent_storage(lambda *values: None)
    launch(first.reference(0))
    second_backend.fail = "validate"
    with pytest.raises(RuntimeError, match="generation"):
        launch(first.reference(0), second.reference(0))
    assert first_backend.ordered == []
    first_backend.complete()
    first.close()
    second.close()


def test_throwing_launcher_still_records_completion():
    owner, backend, resources = make_owner()

    @with_persistent_storage
    def launch(value):
        raise ValueError("failure after enqueue")

    with pytest.raises(ValueError, match="after enqueue"):
        launch(owner.reference(0))
    assert len(backend.events) == 1
    assert backend.released == []
    backend.complete()
    owner.close()


def test_failed_record_retains_owner_and_requires_recovery():
    owner, backend, resources = make_owner()
    backend.fail = "record"
    launch = with_persistent_storage(lambda value: None)
    reference = owner.reference(0)
    with pytest.raises(RuntimeError, match="record failed"):
        launch(reference)
    with pytest.raises(RuntimeError, match="requires completion recovery"):
        launch(reference)
    backend.fail = "recover"
    with pytest.raises(RuntimeError, match="recovery failed"):
        owner.close()
    assert backend.released == []
    backend.fail = None
    owner.close()
    assert backend.recoveries == 1
    assert backend.released == list(resources)


@pytest.mark.parametrize("failure", ["wait", "release"])
def test_close_failure_retains_storage_and_rejects_new_uses(failure):
    owner, backend, resources = make_owner()
    launch = with_persistent_storage(lambda value: None)
    reference = owner.reference(0)
    launch(reference)
    backend.complete()
    backend.fail = failure
    with pytest.raises(RuntimeError, match="failed"):
        owner.close()
    assert backend.released == []
    with pytest.raises(RuntimeError, match="closing or closed"):
        launch(reference)
    backend.fail = None
    owner.close()
    assert backend.released == list(resources)


def test_partial_release_retry_does_not_double_release():
    owner, backend, resources = make_owner(3)
    release = backend.release

    def fail_second(resource):
        if len(backend.released) == 1:
            raise RuntimeError("release failed")
        release(resource)

    backend.release = fail_second
    with pytest.raises(RuntimeError, match="release failed"):
        owner.close()
    assert backend.released == [resources[-1]]
    backend.release = release
    owner.close()
    assert backend.released == list(reversed(resources))


def test_close_waits_for_submission_and_device_completion():
    owner, backend, resources = make_owner()
    reference = owner.reference(0)
    entered = threading.Event()
    finish_submission = threading.Event()
    closing = threading.Event()

    @with_persistent_storage
    def launch(value):
        entered.set()
        assert finish_submission.wait(timeout=3)

    def close():
        closing.set()
        owner.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_launch = executor.submit(launch, reference)
        assert entered.wait(timeout=3)
        pending_close = executor.submit(close)
        assert closing.wait(timeout=3)
        assert backend.released == []
        finish_submission.set()
        pending_launch.result(timeout=3)
        assert not pending_close.done()
        assert backend.released == []
        backend.complete()
        pending_close.result(timeout=3)
    assert backend.released == list(resources)


def test_reverse_argument_order_cannot_deadlock():
    first, first_backend, _ = make_owner()
    second, second_backend, _ = make_owner()
    first_ref, second_ref = first.reference(0), second.reference(0)
    start = threading.Barrier(2)
    launch = with_persistent_storage(lambda *values: None)

    def submit(references):
        start.wait(timeout=3)
        for _ in range(20):
            launch(*references)

    with ThreadPoolExecutor(max_workers=2) as executor:
        forward = executor.submit(submit, (first_ref, second_ref))
        reverse = executor.submit(submit, (second_ref, first_ref))
        forward.result(timeout=3)
        reverse.result(timeout=3)
    for owner, backend in ((first, first_backend), (second, second_backend)):
        assert len(backend.events) == 40
        assert backend.ordered == backend.events[:-1]
        backend.complete()
        owner.close()


def test_reentrant_close_is_rejected_before_release():
    owner, backend, _ = make_owner()
    launch = with_persistent_storage(lambda value: owner.close())
    with pytest.raises(RuntimeError, match="inside its submission"):
        launch(owner.reference(0))
    assert backend.released == []
    backend.complete()
    owner.close()


def test_nested_submission_is_rejected_before_locking_another_owner():
    first, first_backend, _ = make_owner()
    second, second_backend, _ = make_owner()
    inner = with_persistent_storage(lambda value: None)
    outer = with_persistent_storage(lambda value: inner(second.reference(0)))
    with pytest.raises(RuntimeError, match="nested persistent"):
        outer(first.reference(0))
    assert second_backend.events == []
    first_backend.complete()
    first.close()
    second.close()


def test_pending_work_survives_reference_garbage_collection():
    owner, backend, _ = make_owner()
    launch = with_persistent_storage(lambda value: None)
    launch(owner.reference(0))
    retained = weakref.ref(owner)
    del owner
    gc.collect()
    assert retained() is not None
    backend.complete()
    retained().close()
    gc.collect()
    assert retained() is None


def test_plain_arguments_preserve_existing_call_behavior():
    payload = object()
    invoke = with_persistent_storage(lambda value, *, option: (value, option))
    assert invoke(payload, option=3) == (payload, 3)


@pytest.mark.parametrize("index", [-1, 1, True, "0"])
def test_invalid_reference_index(index):
    owner, _, _ = make_owner()
    with pytest.raises(ValueError, match="index"):
        owner.reference(index)
    owner.close()


@pytest.mark.parametrize("container", [tuple, list, dict])
def test_returned_argument_aliases_keep_their_owner(container):
    owner, backend, _ = make_owner()
    reference = owner.reference(0)

    @with_persistent_storage
    def launch(value):
        return {"state": value} if container is dict else container([value])

    result = launch(reference)
    assert (result["state"] if container is dict else result[0]) is reference
    backend.complete()
    owner.close()


def test_failed_first_record_retains_all_submitted_owners():
    first, first_backend, _ = make_owner()
    second, second_backend, _ = make_owner()
    earlier = min((first, second), key=id)
    earlier._backend.fail = "record"
    launch = with_persistent_storage(lambda *values: None)
    with pytest.raises(RuntimeError, match="record failed"):
        launch(first.reference(0), second.reference(0))
    for owner, backend in ((first, first_backend), (second, second_backend)):
        assert backend.released == []
        backend.fail = None
        owner.close()
        assert backend.recoveries == 1


def test_missing_completion_is_diagnosed_and_requires_recovery():
    owner, backend, _ = make_owner()
    backend.record_completion = lambda: None
    launch = with_persistent_storage(lambda value: None)
    with pytest.raises(RuntimeError, match="no completion record"):
        launch(owner.reference(0))
    owner.close()
    assert backend.recoveries == 1


@pytest.mark.parametrize("retained_count", [0, 2])
def test_allocation_requires_one_resource_for_each_declaration(retained_count):
    backend = Backend()
    owner = StorageOwner((), backend, declared=True)
    owner.declare_reference()

    def retain_resources(retain_resource):
        for resource_index in range(retained_count):
            retain_resource(object())

    with pytest.raises(ValueError, match="one resource per declaration"):
        owner.allocate(retain_resources)
    assert backend.recoveries == 1
    assert len(backend.released) == retained_count
    owner.close()
