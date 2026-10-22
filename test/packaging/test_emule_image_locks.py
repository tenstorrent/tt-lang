# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pinned downloads and network isolation of the emule runtime image build."""

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from conftest import REPO_ROOT

CONTAINERS = REPO_ROOT / ".github/containers"
LOCK_TOOL = CONTAINERS / "emule-lock.py"
PIP_LOCK = CONTAINERS / "emule-requirements.lock"
METAL_LOCK = CONTAINERS / "emule-metal-deps.lock"
DOCKERFILE = CONTAINERS / "Dockerfile.emule"
PRIVATE_CONTEXT = "tt-emule-source"
# The image's Python environment: CPython 3.12 on Linux/x86-64.
IMAGE_MARKERS = {
    **default_environment(),
    "implementation_name": "cpython",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_system": "Linux",
    "python_full_version": "3.12.3",
    "python_version": "3.12",
    "sys_platform": "linux",
}
# Ubuntu 24.04 provides glibc 2.39.
IMAGE_PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(39, 4, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
]
IMAGE_TAGS = set(cpython_tags((3, 12), ["cp312"], IMAGE_PLATFORMS)) | set(
    compatible_tags((3, 12), "cp312", IMAGE_PLATFORMS)
)

lock_tool = runpy.run_path(str(LOCK_TOOL))


def report(*items):
    return {"version": "1", "install": list(items)}


def download(name, url, archive_info=None):
    if archive_info is None:
        archive_info = {"hashes": {"sha256": "a" * 64}}
    return {
        "metadata": {"name": name, "version": "1.0"},
        "download_info": {"url": url, "archive_info": archive_info},
    }


def write_reports(tmp_path, *reports):
    paths = []
    for index, content in enumerate(reports):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(content))
        paths.append(path)
    return paths


def test_pip_lock_pins_every_wheel_by_url_and_hash(tmp_path):
    torch = "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
    numpy = "https://files.pythonhosted.org/packages/ab/numpy-2.3.4-cp312-cp312-manylinux_2_28_x86_64.whl"
    upgrade = numpy.replace("numpy-2.3.4", "typing_extensions-4.16.0")
    reports = write_reports(
        tmp_path,
        report(download("torch", torch), download("Typing_Extensions", torch)),
        report(
            download("numpy", numpy, {"hash": "sha256=" + "b" * 64}),
            download("typing-extensions", upgrade),
        ),
    )
    result = subprocess.run(
        [sys.executable, str(LOCK_TOOL), "pip", *map(str, reports)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == lock_tool["PIP_HEADER"] + (
        f"numpy @ {numpy} \\\n    --hash=sha256:{'b' * 64}\n"
        f"torch @ {torch} \\\n    --hash=sha256:{'a' * 64}\n"
        f"typing-extensions @ {upgrade} \\\n    --hash=sha256:{'a' * 64}\n"
    )


WHEEL = "https://files.pythonhosted.org/packages/ab/pkg-1.0-py3-none-any.whl"


@pytest.mark.parametrize(
    "items,message",
    [
        ([download("pkg", WHEEL.replace(".whl", ".tar.gz"))], "HTTPS wheel"),
        ([download("pkg", WHEEL.replace("https:", "http:"))], "HTTPS wheel"),
        ([download("pkg", WHEEL + "#sha256=" + "a" * 64)], "HTTPS wheel"),
        (
            [{"metadata": {"name": "pkg"}, "download_info": {"url": WHEEL}}],
            "HTTPS wheel",
        ),
        ([download("pkg", WHEEL, {})], "no SHA-256 hash"),
        ([download("pkg", WHEEL, {"hash": "md5=" + "a" * 32})], "no SHA-256 hash"),
        (
            [download("pkg", WHEEL), download("PKG", WHEEL.replace("1.0", "2.0"))],
            "more than one file for pkg",
        ),
        ([], "list no packages"),
    ],
)
def test_pip_lock_rejects_downloads_it_cannot_pin(tmp_path, items, message):
    with pytest.raises(SystemExit, match=message):
        lock_tool["pip_lock"](write_reports(tmp_path, report(*items)))


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def make_metal_source(tmp_path):
    source = tmp_path / "tt-metal"
    cache = source / ".cpmcache"
    checkout = cache / "fmt" / "1a2b3"
    checkout.mkdir(parents=True)
    (checkout / "format.h").write_text("formatted\n")
    git("-C", str(checkout), "init", "-q")
    git("-C", str(checkout), "add", ".")
    git(
        "-C",
        str(checkout),
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "Pinned dependency",
    )
    (checkout / "format.h").write_text("patched\n")
    (cache / "fmt" / "cmake.lock").write_text("")
    archive = cache / "boost" / "9f8e7"
    (archive / "tools").mkdir(parents=True)
    (archive / "tools" / "build.sh").write_text("echo build\n")
    (archive / "include.hpp").write_text("boost\n")
    (archive / "alias.hpp").symlink_to("include.hpp")
    (cache / "cpm").mkdir()
    (cache / "cpm" / "CPM_0.40.2.cmake").write_text("cpm\n")
    (source / "runtime" / "sfpi" / "bin").mkdir(parents=True)
    (source / "runtime" / "sfpi" / "bin" / "riscv-gcc").write_text("gcc\n")
    return source, git("-C", str(checkout), "rev-parse", "HEAD")


def lock_entries(text, header):
    assert text.startswith(header)
    return dict(line.split(" ", 1) for line in text[len(header) :].splitlines())


def test_metal_lock_records_cache_entries_and_sfpi(tmp_path):
    source, commit = make_metal_source(tmp_path)
    lock = lock_tool["metal_lock"](source)
    entries = lock_entries(lock, lock_tool["METAL_HEADER"])
    assert list(entries) == [
        ".cpmcache/boost/9f8e7",
        ".cpmcache/cpm/CPM_0.40.2.cmake",
        ".cpmcache/fmt/1a2b3",
        ".cpmcache/fmt/cmake.lock",
        "runtime/sfpi",
    ]
    assert entries[".cpmcache/fmt/1a2b3"] == f"git {commit}"
    assert entries[".cpmcache/fmt/cmake.lock"] == (
        "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    for path in (".cpmcache/boost/9f8e7", "runtime/sfpi"):
        assert re.fullmatch(r"tree-sha256 [0-9a-f]{64}", entries[path])
    result = subprocess.run(
        [sys.executable, str(LOCK_TOOL), "metal", str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == lock


def test_metal_lock_digests_every_change_to_extracted_content(tmp_path):
    source, _ = make_metal_source(tmp_path)
    archive = source / ".cpmcache" / "boost" / "9f8e7"
    changes = [
        lambda: (archive / "include.hpp").write_text("changed\n"),
        lambda: (archive / "tools" / "build.sh").chmod(0o755),
        lambda: (archive / "alias.hpp").unlink()
        or (archive / "alias.hpp").symlink_to("tools/build.sh"),
        lambda: (archive / "tools" / "build.sh").rename(archive / "tools" / "run.sh"),
        lambda: (archive / "extra.hpp").write_text(""),
        lambda: (archive / "empty").mkdir(),
    ]
    digests = set()
    for change in [lambda: None, *changes]:
        change()
        lock = lock_tool["metal_lock"](source)
        digests.add(
            lock_entries(lock, lock_tool["METAL_HEADER"])[".cpmcache/boost/9f8e7"]
        )
    assert len(digests) == len(changes) + 1


@pytest.mark.parametrize("missing", [".cpmcache", "runtime/sfpi"])
def test_metal_lock_requires_the_cache_and_sfpi(tmp_path, missing):
    source, _ = make_metal_source(tmp_path)
    subprocess.run(["rm", "-rf", str(source / missing)], check=True)
    with pytest.raises(SystemExit, match="no CPM source cache or SFPI toolchain"):
        lock_tool["metal_lock"](source)


@pytest.mark.parametrize("change", ["", "missing", "version", "unexpected"])
def test_check_venv_requires_exactly_the_locked_versions(tmp_path, change):
    lock = tmp_path / "emule-requirements.lock"
    lock.write_text(
        lock_tool["PIP_HEADER"]
        + "torch @ https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl \\\n"
        + f"    --hash=sha256:{'a' * 64}\n"
        + "typing-extensions @ https://files.pythonhosted.org/packages/ab/typing_extensions-4.16.0-py3-none-any.whl \\\n"
        + f"    --hash=sha256:{'b' * 64}\n"
    )
    installed = {"torch": "2.9.0+cpu", "Typing_Extensions": "4.16.0", "pip": "24.0"}
    if change == "missing":
        del installed["Typing_Extensions"]
    elif change == "version":
        installed["torch"] = "2.9.0"
    elif change == "unexpected":
        installed["nvidia-cublas"] = "13.0"
    site = tmp_path / "site-packages"
    for name, version in installed.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    result = subprocess.run(
        [sys.executable, "-S", str(LOCK_TOOL), "check-venv", str(lock)],
        env={**os.environ, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == (1 if change else 0), result.stderr
    if change:
        assert "the environment differs from emule-requirements.lock" in result.stderr


def locked_wheels():
    text = PIP_LOCK.read_text()
    body = [line for line in text.splitlines() if not line.startswith("#")]
    assert text.startswith(lock_tool["PIP_HEADER"])
    assert len(body) % 2 == 0 and body
    wheels = {}
    for requirement, digest in zip(body[::2], body[1::2]):
        match = re.fullmatch(r"(\S+) @ (https://\S+\.whl) \\", requirement)
        assert match, requirement
        assert re.fullmatch(r"    --hash=sha256:[0-9a-f]{64}", digest), digest
        name, url = match.groups()
        wheel = parse_wheel_filename(unquote(urlsplit(url).path.rsplit("/", 1)[1]))
        assert wheel[0] == name
        wheels[name] = (url, wheel)
    assert list(wheels) == sorted(wheels)
    return wheels


def requirements(path):
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            yield from requirements(path.parent / line[3:].strip())
        elif line:
            yield Requirement(line)


def test_pip_lock_satisfies_the_test_requirements_on_the_image_platform():
    wheels = locked_wheels()
    for requirement in requirements(REPO_ROOT / "requirements-test.txt"):
        if requirement.marker and not requirement.marker.evaluate(IMAGE_MARKERS):
            continue
        name = canonicalize_name(requirement.name)
        assert name in wheels, f"{name} is missing from {PIP_LOCK.name}"
        version = wheels[name][1][1]
        assert requirement.specifier.contains(
            version, prereleases=True
        ), f"{name} {version} does not satisfy {requirement}"


def test_pip_lock_holds_cpu_only_wheels_for_the_image_platform():
    wheels = locked_wheels()
    torch_url, torch = wheels["torch"]
    assert urlsplit(torch_url).hostname.endswith(".pytorch.org")
    assert urlsplit(torch_url).path.startswith("/whl/cpu/")
    assert torch[1].local == "cpu"
    for name, (_, (_, _, _, tags)) in wheels.items():
        assert not name.startswith(("nvidia-", "cuda-")) and "triton" not in name
        assert tags & IMAGE_TAGS, f"{name} is not a wheel for the image platform"


def test_metal_lock_pins_every_cache_entry_and_sfpi():
    entries = lock_entries(METAL_LOCK.read_text(), lock_tool["METAL_HEADER"])
    paths = list(entries)
    assert paths[-1] == "runtime/sfpi"
    cache = [tuple(path.split("/")) for path in paths[:-1]]
    assert cache == sorted(cache) and all(path[0] == ".cpmcache" for path in cache)
    for path, pin in entries.items():
        assert re.fullmatch(r"\.cpmcache/[^/\s]+(/[^/\s]+)?|runtime/sfpi", path), path
        assert re.fullmatch(r"git [0-9a-f]{40}|(tree-sha256|sha256) [0-9a-f]{64}", pin)
    assert any(pin.startswith("git ") for pin in entries.values())


def dockerfile_stages():
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text())
    stages = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, arguments = line.partition(" ")
        if keyword == "FROM":
            base, name = re.fullmatch(r"(\S+) AS (\S+)", arguments).groups()
            stages.append((name, base, []))
        elif stages:
            stages[-1][2].append((keyword, arguments))
    return stages


def sources(arguments):
    return set(re.findall(r"(?:^|[ ,])(?:--)?from=([^ ,]+)", arguments))


def run_flags(arguments):
    return re.match(r"(?:--\S+\s+)*", arguments)[0].split()


def private_stages(stages):
    """Stages that hold emulator source, from a copy, a mount, or their base."""
    private = set()
    while True:
        reached = {
            name
            for name, base, instructions in stages
            if base in private
            or any(
                sources(arguments) & (private | {PRIVATE_CONTEXT})
                for _, arguments in instructions
            )
        }
        if reached == private:
            return private
        private = reached


def test_only_network_isolated_commands_see_the_emulator_source():
    stages = dockerfile_stages()
    assert stages[-1][0] == "runtime"
    private = private_stages(stages)
    assert private == {"build", "runtime"}
    receivers = []
    for name, base, instructions in stages:
        exposed = base in private
        for keyword, arguments in instructions:
            assert keyword != "ADD", "ADD can download without a pin"
            reads_private = bool(sources(arguments) & (private | {PRIVATE_CONTEXT}))
            if keyword == "COPY" and reads_private:
                if PRIVATE_CONTEXT in sources(arguments):
                    receivers.append(name)
                exposed = True
            if keyword == "RUN":
                assert PRIVATE_CONTEXT not in sources(arguments)
                # A cache mount is shared between stages and builds.
                assert "type=cache" not in arguments
                if exposed or reads_private:
                    assert "--network=none" in run_flags(arguments), (
                        f"{name} runs with network access next to the "
                        f"emulator source: RUN {arguments[:80]}"
                    )
    assert receivers == ["build"]


def stage(name):
    return next(
        instructions for stage, _, instructions in dockerfile_stages() if stage == name
    )


def test_locked_stages_feed_the_image():
    build_base = next(base for name, base, _ in dockerfile_stages() if name == "build")
    assert build_base == "metal-source"
    (verify,) = [arguments for keyword, arguments in stage("metal-source")]
    assert "diff -u /tmp/emule-metal-deps.lock /locks/emule-metal-deps.lock" in verify
    (install,) = [arguments for keyword, arguments in stage("venv") if keyword == "RUN"]
    steps = [
        "pip download --no-cache-dir --no-index --no-deps --require-hashes "
        "--only-binary :all: --dest /tmp/wheels -r /tmp/emule-requirements.lock",
        "pip install --no-cache-dir --no-index --find-links /tmp/wheels torch ",
        "pip install --no-cache-dir --no-index --find-links /tmp/wheels "
        "-r /tmp/requirements/requirements-test.txt",
        "emule-lock.py check-venv /tmp/emule-requirements.lock",
    ]
    install = re.sub(r"\s+", " ", install)
    assert [install.index(step) for step in steps] == sorted(
        install.index(step) for step in steps
    )
    runtime = stage("runtime")
    assert (
        "COPY",
        "--from=venv /opt/ttlang-toolchain/venv /opt/ttlang-toolchain/venv/",
    ) in runtime
    assert {
        source
        for keyword, arguments in runtime
        if keyword == "COPY"
        for source in sources(arguments)
    } == {"build", "venv"}
    exports = {
        source
        for keyword, arguments in stage("emule-locks")
        for source in sources(arguments)
    }
    assert exports == {"metal-fetch", "venv-resolve"}


def test_lock_files_are_image_inputs_and_build_context_files():
    launcher = (REPO_ROOT / "scripts/tt-lang-emule-container.sh").read_text()
    for name in (
        "configure-emule-metal.sh",
        "emule-lock.py",
        "emule-metal-deps.lock",
        "emule-requirements.lock",
    ):
        assert f"    .github/containers/{name}\n" in launcher
        assert (CONTAINERS / name).is_file()
    workflow = (REPO_ROOT / ".github/workflows/tt-lang-emule.yml").read_text()
    assert '      - ".github/containers/*emule*"\n' in workflow
