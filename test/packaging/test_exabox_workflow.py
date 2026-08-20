# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow contracts for shared Exabox Galaxy testing."""

from __future__ import annotations

from conftest import REPO_ROOT

CALL_BUILD = REPO_ROOT / ".github" / "workflows" / "call-build.yml"
CALL_BUILD_DOCKER = REPO_ROOT / ".github" / "workflows" / "call-build-docker.yml"
CALL_TEST_EXABOX = REPO_ROOT / ".github" / "workflows" / "call-test-exabox.yml"
CALL_TEST_HARDWARE = REPO_ROOT / ".github" / "workflows" / "call-test-hardware.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MANUAL_TEST_EXABOX = REPO_ROOT / ".github" / "workflows" / "manual-test-exabox.yml"
IRD_DOCKERFILE = REPO_ROOT / ".github" / "containers" / "Dockerfile"
INSTALL_EXABOX_WORKER = (
    REPO_ROOT / ".github" / "containers" / "install-exabox-worker.sh"
)
UPLIFT_PATHS = REPO_ROOT / ".github" / "scripts" / "uplift-paths.sh"


def test_pull_request_ci_targets_main_only() -> None:
    ci_workflow = CI_WORKFLOW.read_text()

    assert (
        '  pull_request:\n    branches: ["main"]\n'
        "    types: [opened, synchronize, reopened, ready_for_review, edited]"
        in ci_workflow
    )


def test_hardware_event_policy_and_manual_controls() -> None:
    ci_workflow = CI_WORKFLOW.read_text()
    call_build = CALL_BUILD.read_text()

    assert ci_workflow.count("run_galaxy_tests:") == 2
    assert ci_workflow.count("run_loudbox_tests:") == 2
    assert (
        "run_galaxy_tests: ${{ github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && inputs.run_galaxy_tests) }}"
        in ci_workflow
    )
    assert (
        "run_loudbox_tests: ${{ github.event_name == 'pull_request' || "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main') || "
        "(github.event_name == 'workflow_dispatch' && inputs.run_loudbox_tests) }}"
        in ci_workflow
    )
    assert call_build.count("run_galaxy_tests:") == 2
    assert call_build.count("run_loudbox_tests:") == 2


def test_exabox_configuration_remains_available() -> None:
    call_build = CALL_BUILD.read_text()

    assert "uses: ./.github/workflows/call-test-exabox.yml" in call_build
    assert "if: ${{ inputs.run_galaxy_tests }}" in call_build
    assert "needs: [test-hardware, test-exabox]" in call_build
    test_exabox = call_build.split("\n    test-exabox:", 1)[1].split(
        "\n    report-skipped-galaxy:", 1
    )[0]
    assert "timeout: 90" in test_exabox


def test_hardware_matrix_adds_manual_blackhole_loudbox() -> None:
    call_build = CALL_BUILD.read_text()
    hardware_workflow = CALL_TEST_HARDWARE.read_text()

    assert "inputs.run_loudbox_tests && fromJSON" in call_build
    assert '["n150","bh-loudbox-viommu"]' in call_build
    assert "matrix.hardware == 'bh-loudbox-viommu'" in call_build
    assert "tt-ubuntu-2204-bh-loudbox-viommu-stable" in call_build
    assert "tt-ubuntu-2204-n150-stable" in call_build
    assert (
        "defer_result_check: ${{ matrix.hardware != 'bh-loudbox-viommu' }}"
        in call_build
    )
    assert "--optional-runner" not in call_build
    assert "inputs.run_galaxy_tests && 3 || 2" in call_build
    assert "inputs.run_galaxy_tests && 2 || 1" in call_build
    assert 'name: "Hardware Tests (${{ inputs.hardware }})"' in hardware_workflow
    assert "runs-on: ${{ inputs.runner_label }}" in hardware_workflow
    assert "RUNS_ON: ${{ inputs.hardware }}" in hardware_workflow


def test_exabox_workflow_dispatches_all_worker_operations_through_scripts() -> None:
    workflow = CALL_TEST_EXABOX.read_text()

    assert "runs-on: exabox-multihost-ci-sc1" in workflow
    assert "image: ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:" in workflow
    assert "run: |" not in workflow
    assert "default: 90" in workflow
    assert "CCACHE_DIR: /ci/ccache" in workflow
    assert "HW_TEST_WORKERS: 32" in workflow
    assert "HW_PYTEST_TIMEOUT" not in workflow
    assert "uses: hendrikmuhs/ccache-action@v1.2" in workflow
    assert "key: Linux-ttlang-hw-galaxy" in workflow
    assert ".github/scripts/prepare-exabox-workspace.sh stage" in workflow
    required_phases = {
        "configure",
        "build",
        "install-dependencies",
        "reset",
        "smoketest",
        "simple-add",
        "simulator",
        "python-lit",
        "python-pytests",
        "me2e",
        "examples",
        "tutorials",
        "collect-exabox-reports",
    }
    for phase in required_phases:
        assert f".github/scripts/run-exabox-hardware-phase.sh {phase}" in workflow
    assert workflow.count("steps.stage.outcome == 'success'") == 2
    assert "options: --device /dev/tenstorrent" not in workflow


def test_manual_workflow_calls_the_reusable_exabox_workflow() -> None:
    workflow = MANUAL_TEST_EXABOX.read_text()

    assert "workflow_dispatch:" in workflow
    assert "default: 90" in workflow
    assert "uses: ./.github/workflows/call-build-docker.yml" in workflow
    assert "push: true" in workflow
    assert "uses: ./.github/workflows/call-test-exabox.yml" in workflow
    assert "docker_tag: ${{ needs.build-image.outputs.docker-tag }}" in workflow


def test_ird_image_installs_versioned_exabox_worker_support() -> None:
    dockerfile = IRD_DOCKERFILE.read_text()
    installer = INSTALL_EXABOX_WORKER.read_text()
    build_workflow = CALL_BUILD_DOCKER.read_text()
    uplift_paths = UPLIFT_PATHS.read_text()

    assert "install-exabox-worker.sh" in dockerfile
    assert "OMPI_TAG=v5.0.7" in dockerfile
    assert "ENV LD_LIBRARY_PATH=$OMPI_PREFIX" not in dockerfile
    assert "ENV CPATH=$OMPI_PREFIX" not in dockerfile
    assert "ENV PKG_CONFIG_PATH=$OMPI_PREFIX" not in dockerfile
    assert 'test -x "$OMPI_PREFIX/bin/prted"' in installer
    assert 'install -d -o "$worker_uid" -g "$worker_gid" -m 0755' in installer
    assert "install -d -m 0777" not in installer
    assert "xargs -r" not in installer
    assert "build/profiler/build_wasm/traces" in installer
    assert "prted --version" in build_workflow
    assert "docker run --rm --user 1001:1001" in build_workflow
    assert "export HOME=/home/user" in build_workflow
    assert "export TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal" in build_workflow
    assert "export PYTHONPATH=$TT_METAL_HOME/python_packages/ttnn" in build_workflow
    assert 'test -w "$TT_METAL_HOME/build/profiler/build_wasm/traces"' in build_workflow
    assert 'test -w "$TT_METAL_HOME/generated/profiler"' in build_workflow
    assert 'python3 -c "import ttnn"' in build_workflow
    assert ".github/containers/install-exabox-worker.sh" in uplift_paths
