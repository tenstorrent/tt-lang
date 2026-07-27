# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for packaging helper scripts used by wheel workflows."""

from __future__ import annotations

import datetime
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import REPO_ROOT  # noqa: E402

SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
CHECK_WHEEL_TTNN_METADATA = SCRIPTS_DIR / "check-wheel-ttnn-metadata.py"
CHECK_LIGHT_METAPACKAGE = SCRIPTS_DIR / "check-light-metapackage.py"
BUILD_MANYLINUX_CORE_WHEEL = SCRIPTS_DIR / "build-manylinux-core-wheel.sh"
RESOLVE_PYPI_PUBLISH_INPUTS = SCRIPTS_DIR / "resolve-pypi-publish-inputs.sh"
RESOLVE_S3_PUBLISH_INPUTS = SCRIPTS_DIR / "resolve-s3-publish-inputs.sh"
RESOLVE_S3_PUBLISH_PREFIX = SCRIPTS_DIR / "resolve-s3-publish-prefix.sh"
INJECT_S3_PUBLISH_README = SCRIPTS_DIR / "inject-s3-publish-readme.sh"
COMPUTE_NIGHTLY_VERSION = SCRIPTS_DIR / "compute-nightly-version.py"
CHECK_INSTALLED_TTNN = SCRIPTS_DIR / "check-installed-ttnn.py"
CHECK_BUNDLED_PAYLOAD = SCRIPTS_DIR / "check-wheel-bundled-payload.py"
PUBLISH_S3_PYPI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-s3-pypi.yml"
PUBLISH_PYPI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CALL_BUILD_DOCKER_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-build-docker.yml"
)
CALL_BUILD_WHEEL_IMAGES_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-build-wheel-images.yml"
)
CALL_BUILD_MANYLINUX_WHEELS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-build-manylinux-wheels.yml"
)
CALL_TEST_MANYLINUX_WHEELS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-test-manylinux-wheels.yml"
)
SETUP_WHEEL_IMAGE_BUILD_ACTION = (
    REPO_ROOT / ".github" / "actions" / "setup-wheel-image-build" / "action.yml"
)
CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-ttmetal-light-wheel.yml"
)
DETECT_TTLANG_TTMETAL_UPLIFT = SCRIPTS_DIR / "detect-ttmetal-uplift.sh"
RECORD_TTMETAL_MISS = SCRIPTS_DIR / "record-ttmetal-miss.sh"
CALL_BUILD_WHEELS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-build-wheels.yml"
)
CALL_TEST_DIST_TUTORIALS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "call-test-dist-tutorials.yml"
)
TTMETAL_LIGHT_ON_DEMAND_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "ttmetal-light-on-demand.yml"
)
TTMETAL_LIGHT_XLA_ON_DEMAND_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "ttmetal-light-xla-on-demand.yml"
)
S3_PYPI_OPS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "s3-pypi-ops.yml"
MANYLINUX_WHEEL_DOCKERFILE = (
    REPO_ROOT / ".github" / "containers" / "Dockerfile.wheel-manylinux-2-34"
)
WHEEL_TOOLCHAIN_CMAKELISTS = (
    REPO_ROOT / ".github" / "containers" / "CMakeLists.wheel-toolchain"
)
TOOLCHAIN_COMPONENT_MODULE = (
    REPO_ROOT / "cmake" / "modules" / "TTLangToolchainComponent.cmake"
)
TOOLCHAIN_OPTIONS_MODULE = (
    REPO_ROOT / "cmake" / "modules" / "TTLangToolchainOptions.cmake"
)
SETUP_PY = REPO_ROOT / "setup.py"


def _run_script(
    script: Path,
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_wheel(dist_dir: Path, filename: str, metadata: str) -> Path:
    wheel_path = dist_dir / filename
    dist_info = filename.split("-", 1)[0] + "-0.0.0.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{dist_info}/METADATA", metadata)
    return wheel_path


def _add_config_py(wheel_path: Path, *, tt_metal_commit: str) -> None:
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttl/config.py", f'TT_METAL_COMMIT = "{tt_metal_commit}"\n')


def test_manylinux_core_helper_resolves_sibling_scripts_from_its_directory() -> None:
    helper = BUILD_MANYLINUX_CORE_WHEEL.read_text()

    assert 'script_dir=$(CDPATH=\'\' cd -- "$(dirname -- "$0")" && pwd)' in helper
    assert '"$script_dir/resolve-wheel-versions.sh"' in helper
    assert '"$script_dir/configure-ttlang-build.sh"' in helper


def test_s3_workflow_routes_light_wheels_to_manylinux_builder() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    shared_build = CALL_BUILD_MANYLINUX_WHEELS_WORKFLOW.read_text()

    assert (
        "github.event_name == 'workflow_dispatch' && inputs.wheel_variant || ''"
    ) in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow
    assert "uses: ./.github/workflows/call-build-manylinux-wheels.yml" in workflow
    assert "ttnn_dep_mode: ${{ matrix.ttnn_dep_mode }}" in workflow
    assert "tt-lang-wheel-manylinux-2-34-${{ matrix.python_tag }}" in shared_build
    assert "python_tag: [cp310, cp312]" in shared_build
    assert ".github/scripts/build-manylinux-wheel-set-member.sh" in shared_build
    assert ".github/scripts/test-s3-light-wheels.sh" in shared_build
    assert ".github/scripts/inject-s3-index-readme.sh" not in workflow
    assert ".github/scripts/inject-s3-publish-readme.sh" in workflow
    assert "PUBLISH_PREFIX: ${{ needs.preflight.outputs.publish_prefix }}" in workflow
    assert '--find-links-subdir "$PUBLISH_PREFIX"' in workflow
    assert "python3 -m pip install s3pypi" not in workflow
    assert "manylinux_wheel_matrix" in workflow


def test_regular_s3_publish_prefix_routes_dev_to_month_and_final_to_releases() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    resolver = RESOLVE_S3_PUBLISH_PREFIX.read_text()
    assert "resolve-s3-publish-prefix.sh" in workflow
    assert "prefix=tt-lang/releases" in resolver
    assert 'prefix="tt-lang/${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"' in resolver
    assert '--prefix "$PUBLISH_PREFIX"' in workflow


def test_regular_s3_index_injection_targets_parent_slash_key() -> None:
    # Regular direct publishing regenerates the top tt-lang/ slash-key listing;
    # the README injection must target that parent listing, not the month dir.
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    injector = INJECT_S3_PUBLISH_README.read_text()
    assert "inject-s3-publish-readme.sh" in workflow
    assert 'parent="$(dirname "$prefix")"' in injector
    assert '--key "$parent/"' in injector
    assert 'key="$prefix/index.html"' not in injector
    assert 'key="$prefix/"' not in injector


def test_ttmetal_light_workflow_builds_and_validates_metapackage() -> None:
    workflow = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()

    assert "name: Build tt-lang-light per-tt-metal-SHA wheel (reusable)" in workflow
    assert "preflight:" in workflow
    assert 'name: "Validate tt-lang-light per-tt-metal-SHA inputs"' in workflow
    assert "S3 publishing is restricted to refs/heads/main" in workflow
    assert "python_tags must include cp312" in workflow
    assert "python_tags: ${{ steps.python_tags.outputs.python_tags }}" in workflow
    assert "needs: preflight" in workflow
    assert "build-metapackage:" in workflow
    assert "matrix.python_tag == 'cp312'" not in workflow
    assert "name: ttmetal-light-metapackage" in workflow
    assert "path: dist/tt_lang_light-*.whl" in workflow
    assert (
        "needs: [build-ttmetal, find-compatible, build-wheels, build-metapackage]"
        in workflow
    )
    assert (
        "needs: [preflight, find-compatible, build-wheels, build-metapackage, device-validate]"
        in workflow
    )
    assert (
        "needs.find-compatible.outputs.found == 'false' && inputs.dry_run != true && github.ref == 'refs/heads/main'"
        in workflow
    )
    assert (
        "needs.find-compatible.outputs.found == 'true' && inputs.dry_run != true && github.ref == 'refs/heads/main'"
        in workflow
    )
    assert "metapackage_wheel=$(ls dist/tt_lang_light-*-py3-none-any.whl)" in workflow
    assert "--find-links dist" in workflow
    assert '"$metapackage_wheel"' in workflow
    assert "tt-lang-setup" in workflow
    # tt-metal is built once (build-ttmetal, in the oldest-glibc manylinux
    # container) and the install is shared as an artifact; find-compatible,
    # build-wheels, and device-validate download it instead of rebuilding.
    assert "build-ttmetal:" in workflow
    assert workflow.count(".github/scripts/build-ttmetal-at-sha.sh") == 1
    assert "tt-lang-wheel-manylinux-2-34-cp312" in workflow
    # One upload + three downloads of the shared install.
    assert workflow.count("name: ttmetal-install") == 4
    assert "TTLANG_EXTERNAL_TT_METAL_DIR: /tmp/ttmetal-install" in workflow
    assert "TTMETAL_INSTALL_DIR: /tmp/ttmetal-install" in workflow
    assert "ttmetal_sha: ${{ steps.ttmetal.outputs.sha }}" in workflow
    assert "ttmetal_sha: ${{ needs.build-ttmetal.outputs.ttmetal_sha }}" in workflow
    assert "TT_METAL_COMMIT: ${{ needs.build-ttmetal.outputs.ttmetal_sha }}" in workflow
    assert (
        "TT_METAL_COMMIT: ${{ needs.find-compatible.outputs.ttmetal_sha }}" in workflow
    )
    assert (
        "EXPECTED_TT_METAL_COMMIT: ${{ needs.find-compatible.outputs.ttmetal_sha }}"
        in workflow
    )
    assert 'actual = ttl.build_info()["tt_metal"]' in workflow
    # tar transfer (not a bare artifact) so the sfpi compiler keeps its +x bit;
    # unpacked by each of the three consumers.
    assert "Package tt-metal install" in workflow
    assert workflow.count("--strip-components=1") == 3
    # Wheel validation is a device smoke -- smoketest plus the tutorials, which
    # import only ttl and ttnn. The exhaustive test/python and test/me2e
    # regression runs against the source build, so it is not repeated here, and
    # the test-tree import shim it needed is gone.
    assert "test/python/smoketest.py" in workflow
    assert ".github/scripts/run-tutorials.sh ." in workflow
    assert "compile-and-run-examples.sh" not in workflow
    assert "test_import_root" not in workflow
    assert 'pytest -c /dev/null --rootdir "$PWD" test/python' not in workflow
    assert 'pytest -c /dev/null --rootdir "$PWD" test/me2e' not in workflow
    assert "simple_add" not in workflow
    assert ".github/scripts/publish-s3-direct-wheels.sh" in workflow
    assert "Install Markdown renderer" in workflow
    assert (
        '--light-python-tags "${{ needs.preflight.outputs.python_tags }}"' in workflow
    )
    assert (
        '--prefix "tt-lang/ttmetal/${{ needs.find-compatible.outputs.ttmetal_short }}"'
        in (workflow)
    )
    assert (
        '--find-links-subdir "tt-lang/ttmetal/${{ needs.find-compatible.outputs.ttmetal_short }}"'
        in workflow
    )
    assert "Inject S3 index README" not in workflow


def test_per_sha_prefix_is_consistent_across_publish_detect_record() -> None:
    detect_script = DETECT_TTLANG_TTMETAL_UPLIFT.read_text()
    record_script = RECORD_TTMETAL_MISS.read_text()
    workflow = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()

    assert "tt-lang/ttmetal/" in detect_script
    assert "tt-lang/ttmetal/" in record_script
    assert "tt-lang/ttmetal/" in workflow

    old_forms = (
        "s3://$S3_BUCKET/tt-lang/$1/",
        "s3://$S3_BUCKET/tt-lang/$short/attempt.json",
    )
    for script_text in (detect_script, record_script):
        for old_form in old_forms:
            assert old_form not in script_text


def test_ttmetal_light_workflow_names_are_specific() -> None:
    on_demand = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    reusable = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()
    xla = TTMETAL_LIGHT_XLA_ON_DEMAND_WORKFLOW.read_text()
    publish = PUBLISH_S3_PYPI_WORKFLOW.read_text()

    assert "name: Build tt-lang-light per-tt-metal-SHA wheel (on demand)" in on_demand
    assert 'name: "Detect tt-lang-light per-tt-metal-SHA build"' in on_demand
    assert 'name: "Build tt-lang-light per-tt-metal-SHA wheel"' in on_demand
    assert "name: Build tt-lang-light per-tt-metal-SHA wheel (reusable)" in reusable

    assert "name: Build tt-lang-light XLA per-tt-metal-SHA wheel (on demand)" in xla
    assert 'name: "Resolve tt-lang-light XLA per-tt-metal-SHA inputs"' in xla
    assert 'name: "Build Ubuntu tt-lang-light XLA per-tt-metal-SHA wheel"' in xla
    assert 'name: "Device-validate tt-lang-light XLA per-tt-metal-SHA wheel"' in xla

    assert 'name: "Detect tt-lang-light per-tt-metal-SHA build"' in publish
    assert 'name: "Build tt-lang-light per-tt-metal-SHA wheel"' in publish


def test_ttmetal_light_max_age_crosses_reusable_workflow_as_string() -> None:
    on_demand_workflow = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    reusable_workflow = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()
    on_demand_max_age = on_demand_workflow.split("      max_age_days:", 1)[1].split(
        "      python_tags:", 1
    )[0]
    reusable_max_age = reusable_workflow.split("      max_age_days:", 1)[1].split(
        "      python_tags:", 1
    )[0]

    assert "type: string" in on_demand_max_age
    assert 'default: "14"' in on_demand_max_age
    assert (
        "max_age_days: ${{ format('{0}', inputs.max_age_days) }}" in on_demand_workflow
    )
    assert "type: string" in reusable_max_age
    assert 'default: "14"' in reusable_max_age


def test_ttmetal_light_on_demand_detect_skips_s3_for_dry_run() -> None:
    workflow = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    # Dry-run and forced-SHA branch runs do not need S3 credentials.
    assert "if: ${{ inputs.dry_run != true && inputs.tt_metal_sha == '' }}" in workflow
    assert (
        "if: ${{ inputs.dry_run != true && github.ref != 'refs/heads/main' }}"
        in workflow
    )
    assert "S3 publishing is restricted to refs/heads/main" in workflow
    assert "detect-ttmetal-uplift.sh --assume-new" in workflow
    assert 'forced_sha="$(printf \'%s\' "$FORCED_SHA"' in workflow
    assert 'echo "tt_metal_sha=$forced_sha" >> "$GITHUB_OUTPUT"' in workflow


def test_ttmetal_light_on_demand_detect_uses_version_file() -> None:
    workflow = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    assert "submodules: true" not in workflow
    assert "git submodule update --init" not in workflow
    assert "third-party/tt-metal-version" in workflow


def test_ttlang_ref_threads_from_on_demand_to_reusable_workflow() -> None:
    on_demand = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    reusable = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()
    # The on-demand input is declared and passed through to the reusable build.
    assert "ttlang_ref:" in on_demand
    assert "ttlang_ref: ${{ inputs.ttlang_ref }}" in on_demand
    assert "ttlang_ref:" in reusable


def test_pinned_ttlang_ref_skips_search_and_tt_metal_build() -> None:
    reusable = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()
    # A pinned ref is checked out directly, falling back to the trigger commit
    # in search mode, and drives the build.
    assert "ref: ${{ inputs.ttlang_ref || github.sha }}" in reusable
    # The tt-metal build (the search's device gate) is skipped when pinning.
    assert "if: ${{ inputs.ttlang_ref == '' }}" in reusable
    # The pin path emits the winner without the compatibility search.
    assert "python3 .github/scripts/compute-nightly-version.py" in reusable
    assert 'winner_sha="$(git rev-parse HEAD)"' in reusable


def test_publish_s3_supports_pinned_ref_and_wheel_patches() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    bundled_build = CALL_BUILD_WHEELS_WORKFLOW.read_text()
    manylinux_build = CALL_BUILD_MANYLINUX_WHEELS_WORKFLOW.read_text()
    # Dispatch inputs to rebuild from a pinned ref and optionally patch it.
    assert "ttlang_ref:" in workflow
    assert "apply_patches:" in workflow
    # Direct checkouts honor the pinned ref, falling back to the trigger commit.
    assert "ref: ${{ inputs.ttlang_ref || github.sha }}" in workflow
    # The override threads to the build reusables under their own input name.
    assert "ttlang_sha_override: ${{ inputs.ttlang_ref }}" in workflow
    # Patches come from the workflow commit (a checkout of github.sha), not the
    # target ref -- an older ref that needs a patch predates the patch files.
    assert "ref: ${{ github.sha }}" in bundled_build
    assert "path: .wheel-patch-src" in bundled_build
    assert (
        ".wheel-patch-src/.github/scripts/apply-wheel-patches.sh"
        ' --target-dir "$GITHUB_WORKSPACE"'
    ) in bundled_build
    assert "ref: ${{ github.sha }}" in manylinux_build
    assert "path: .wheel-workflow-src" in manylinux_build
    assert (
        ".wheel-workflow-src/.github/scripts/apply-wheel-patches.sh"
        ' --target-dir "$GITHUB_WORKSPACE"'
    ) in manylinux_build
    # apply_patches stays a valid boolean on push/schedule (no dispatch inputs).
    assert (
        "apply_patches: ${{ github.event_name == 'workflow_dispatch'"
        " && inputs.apply_patches }}"
    ) in workflow


def test_scheduled_s3_publish_skips_unchanged_source_sha() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()

    assert (
        "# Weekly S3-only publish: 08:00 UTC Monday (00:00 PST / 01:00 PDT)."
        in workflow
    )
    assert '- cron: "0 8 * * 1"' in workflow
    assert '- cron: "0 8 * * *"' not in workflow
    assert "publish_needed: ${{ steps.nightly.outputs.publish-needed }}" in workflow
    assert "s3-nightly-state.py check" in workflow
    for job_name in (
        "build-docker",
        "build-wheel-images",
        "build-bundled-wheels",
        "build-manylinux-wheels",
        "publish",
        "ttmetal-light-detect",
        "ttmetal-light-build",
    ):
        job_prefix = workflow.split(f"\n  {job_name}:", 1)[1][:1000]
        assert "needs.preflight.outputs.publish_needed == 'true'" in job_prefix
    assert workflow.index("Record scheduled publish state") > workflow.index(
        "Restore S3 index README"
    )


def test_s3_publish_resolves_prefix_before_building() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    preflight_job = workflow.split("\n  preflight:", 1)[1].split(
        "\n  build-docker:", 1
    )[0]
    publish_job = workflow.split("\n  publish:", 1)[1].split(
        "\n  ttmetal-light-detect:", 1
    )[0]

    assert preflight_job.count("resolve-s3-publish-prefix.sh") == 1
    assert "publish_prefix: ${{ steps.publish-prefix.outputs.prefix }}" in (
        preflight_job
    )
    assert "resolve-s3-publish-prefix.sh" not in publish_job
    assert publish_job.count("needs.preflight.outputs.publish_prefix") == 3


def test_s3_publish_requires_every_selected_wheel_build_to_succeed() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    publish_job = workflow.split("\n  publish:", 1)[1].split(
        "\n  ttmetal-light-detect:", 1
    )[0]

    assert (
        "needs.preflight.outputs.bundled_selected != 'true' || "
        "needs.build-bundled-wheels.result == 'success'"
    ) in publish_job
    assert (
        "needs.preflight.outputs.manylinux_selected != 'true' || "
        "needs.build-manylinux-wheels.result == 'success'"
    ) in publish_job


def test_publish_pypi_supports_release_sha_from_main() -> None:
    workflow = PUBLISH_PYPI_WORKFLOW.read_text()
    resolver = RESOLVE_PYPI_PUBLISH_INPUTS.read_text()
    preflight_job = workflow.split("\n  preflight:", 1)[1].split(
        "\n  build-wheel-images:", 1
    )[0]
    publish_job = workflow.split("\n  publish:", 1)[1].split("\n  dry-run-summary:", 1)[
        0
    ]

    assert "ttlang_sha:" in workflow
    assert "ref: ${{ inputs.ttlang_sha || github.sha }}" in workflow
    assert "path: release-source" in preflight_job
    assert 'git -C "$RELEASE_SOURCE" rev-parse HEAD' in resolver
    assert "TTLANG_TT_METAL_VERSION_FILE:" in preflight_job
    assert "ttlang_sha must be a full 40-character commit SHA" in resolver
    assert (
        'git -C "$RELEASE_SOURCE" merge-base --is-ancestor'
        ' "$ttlang_sha" "${GITHUB_SHA:?GITHUB_SHA is required}"'
    ) in resolver
    assert "git -C \"$RELEASE_SOURCE\" tag --list 'v[0-9]*' --points-at" in resolver
    assert '"$script_dir/require-release-tag.sh" "$release_ref"' in resolver
    assert workflow.count("ttlang_sha_override: ${{ inputs.ttlang_sha }}") == 3
    assert 'echo "dry_run=${{ inputs.dry_run }}"' not in workflow
    assert 'echo "docker_tag=${{ inputs.docker_tag }}"' not in workflow
    assert 'echo "ttlang_sha=${{ inputs.ttlang_sha }}"' not in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.dry_run != true" in (
        publish_job
    )
    # The OIDC-enabled job runs verification code from the dispatch ref, not
    # from the older source commit.
    assert "inputs.ttlang_sha" not in publish_job


def test_publish_pypi_terminal_jobs_override_skipped_docker_build_status() -> None:
    workflow = PUBLISH_PYPI_WORKFLOW.read_text()
    publish_job = workflow.split("\n  publish:", 1)[1].split("\n  dry-run-summary:", 1)[
        0
    ]
    dry_run_summary_job = workflow.split("\n  dry-run-summary:", 1)[1]

    for terminal_job in (publish_job, dry_run_summary_job):
        assert "!cancelled()" in terminal_job
        assert "needs.preflight.result == 'success'" in terminal_job
        assert "needs.build-wheels.result == 'success'" in terminal_job
    assert "needs.test-wheels.result == 'success'" in publish_job


def test_dist_tutorial_workflow_supports_pinned_ref() -> None:
    workflow = CALL_TEST_DIST_TUTORIALS_WORKFLOW.read_text()

    assert workflow.count("ttlang_sha_override:") == 2
    assert "ref: ${{ inputs.ttlang_sha_override || github.sha }}" in workflow


def test_call_build_wheels_supports_pinned_ref_and_patches() -> None:
    workflow = CALL_BUILD_WHEELS_WORKFLOW.read_text()
    assert "ttlang_sha_override:" in workflow
    assert "apply_patches:" in workflow
    assert "ref: ${{ inputs.ttlang_sha_override || github.sha }}" in workflow
    # Patches are sourced from the workflow commit and applied to the target tree.
    assert "path: .wheel-patch-src" in workflow
    assert (
        ".wheel-patch-src/.github/scripts/apply-wheel-patches.sh"
        ' --target-dir "$GITHUB_WORKSPACE"'
    ) in workflow


def test_on_demand_requires_tt_metal_sha_when_ttlang_ref_pinned() -> None:
    workflow = TTMETAL_LIGHT_ON_DEMAND_WORKFLOW.read_text()
    # Auto-detect reads the dispatch ref's tt-metal pin, so a pinned tt-lang ref
    # without an explicit tt_metal_sha would target the wrong tt-metal -- reject it.
    assert 'if [ -n "$TTLANG_REF" ] && [ -z "$forced_sha" ]; then' in workflow
    assert "TTLANG_REF: ${{ inputs.ttlang_ref }}" in workflow


def test_scheduled_light_wheel_soft_fails_without_failing_publish() -> None:
    publish = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    reusable = CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW.read_text()
    # The scheduled detect job tolerates its own failure, and the reusable
    # build it feeds is invoked in soft-fail mode.
    assert "continue-on-error: true" in publish
    assert "soft_fail: true" in publish
    # Every job in the reusable build honors soft_fail (caller-level
    # continue-on-error is not valid on a reusable-workflow job).
    assert "soft_fail:" in reusable
    assert reusable.count("continue-on-error: ${{ inputs.soft_fail }}") == 7


def test_ttmetal_light_xla_workflow_uses_ubuntu_external_builder() -> None:
    workflow = TTMETAL_LIGHT_XLA_ON_DEMAND_WORKFLOW.read_text()
    docker_tag_input = workflow.split("      docker_tag:", 1)[1].split(
        "      version_override:", 1
    )[0]

    assert "ttlang_ref:" in workflow
    assert "light-wheel packaging and build_info() provenance support" in workflow
    assert "tt_metal_sha:" in workflow
    assert "Leave empty to resolve the closest existing tag from ttlang_ref" in (
        docker_tag_input
    )
    assert "required: true" in workflow
    assert "resolve-xla-build-inputs.sh" in workflow
    assert "required: false" in docker_tag_input
    assert 'default: ""' in docker_tag_input
    assert "--resolve-existing-docker-tag" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert "TTLANG_IRD_DOCKER_OWNER: tenstorrent" in workflow
    assert (
        "tt-lang-ird-ubuntu-24-04:${{ needs.resolve.outputs.docker_tag }}" in workflow
    )
    assert ".github/scripts/build-ttmetal-at-sha.sh" in workflow
    assert "--scratch-dir /tmp/ttmetal-xla-sha" in workflow
    assert "TTNN_DEP_MODE: external" in workflow
    assert "TTLANG_TTNN_DEP_MODE: external" in workflow
    assert (
        "TTLANG_EXTERNAL_TT_METAL_DIR: ${{ steps.ttmetal.outputs.install_dir }}"
        in workflow
    )
    assert "TT_METAL_COMMIT: ${{ steps.ttmetal.outputs.sha }}" in workflow
    assert "pip wheel . --wheel-dir=dist-raw --no-deps --no-build-isolation" in workflow
    # Standard tt-lang-light wheels: no name/version suffix mechanism at all.
    assert "EXTERNAL_WHEEL_SUFFIX" not in workflow
    assert "--package-suffix" not in workflow
    assert "light.xla" not in workflow
    # XLA is distinguished by the dist/xla/<ttmetal7> location, not the wheel name.
    assert "dist/xla/${{ steps.ttmetal.outputs.short }}" in workflow
    assert "tt-lang-light-xla-wheels" in workflow
    # The wheel is device-validated against the same tt-metal SHA it was built on.
    assert "Device-validate tt-lang-light XLA per-tt-metal-SHA wheel" in workflow
    assert "options: --device /dev/tenstorrent" in workflow
    assert "bash .github/scripts/run-tutorials.sh ." in workflow
    # tt-metal built once (build job), shared as a tar artifact preserving the
    # sfpi +x bit; device-validate downloads and unpacks it, not rebuilds.
    assert "Package tt-metal install" in workflow
    assert workflow.count("name: ttmetal-install") == 2
    assert "TTMETAL_INSTALL_DIR: /tmp/ttmetal-install" in workflow
    assert "--strip-components=1" in workflow
    assert "tt-lang-wheel-manylinux-2-34" not in workflow
    assert "build-s3-light-core-wheel.sh" not in workflow
    assert "build-s3-light-metapackage-wheel.sh" not in workflow
    assert '--expect-tt-metal-commit "${{ steps.ttmetal.outputs.sha }}"' in workflow
    # Device-validate must compare against the resolved full SHA embedded in the
    # wheel (exported as a build-job output), not the raw dispatch input: a short
    # SHA, tag, or branch would not equal build_info()["tt_metal"] and would fail
    # the provenance check spuriously.
    assert "ttmetal_sha: ${{ steps.ttmetal.outputs.sha }}" in workflow
    assert (
        "EXPECTED_TT_METAL_COMMIT: ${{ needs.build.outputs.ttmetal_sha }}" in workflow
    )
    assert "EXPECTED_TT_METAL_COMMIT: ${{ inputs.tt_metal_sha }}" not in workflow


def test_s3_pypi_ops_workflow_is_main_gated_and_dry_run_by_default() -> None:
    workflow = S3_PYPI_OPS_WORKFLOW.read_text()

    assert "name: S3 PyPI ops (tt-lang)" in workflow
    assert (
        'run-name: "S3 PyPI ops: ${{ inputs.operation }} '
        '(dry_run=${{ inputs.dry_run }})"'
    ) in workflow
    assert "options: [inspect, put-index, move, copy, delete, readonly-cmd]" in workflow
    assert "id-token: write" in workflow
    assert "uses: ./.github/actions/configure-tt-s3-credentials" in workflow
    # Creds are configured only on main, so a non-main run never assumes the
    # shared-bucket role, for any operation (read or write, dry-run or not).
    assert "if: ${{ github.ref == 'refs/heads/main' }}" in workflow
    assert (
        "if: ${{ inputs.dry_run != true && github.ref != 'refs/heads/main' }}"
        in workflow
    )
    assert "### S3 PyPI operation" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "printf 'operation=%s\\n' \"$OP\"" in workflow
    assert "printf 'dry_run=%s\\n' \"$DRY_RUN\"" in workflow
    assert "printf 'ref=%s\\n' \"$GITHUB_REF\"" in workflow
    assert "printf 'actor=%s\\n' \"$GITHUB_ACTOR\"" in workflow
    assert workflow.index("- name: Summarize operation") < workflow.index(
        "- name: Require main ref for writes"
    )
    assert "Install Markdown renderer" in workflow
    assert (
        "inputs.operation == 'put-index' && inputs.dry_run != true && "
        "(inputs.prefix == 'tt-lang' || inputs.prefix == 'tt-lang/')"
    ) in workflow
    assert ".github/scripts/s3-pypi-ops.sh" in workflow
    # dry_run input defaults to true
    dry_run_input = workflow.split("      dry_run:", 1)[1].split("concurrency:", 1)[0]
    assert "default: true" in dry_run_input


def test_light_core_builder_checks_tt_metal_provenance_when_exported() -> None:
    script = BUILD_MANYLINUX_CORE_WHEEL.read_text()

    assert (
        'TT_METAL_COMMIT="${TT_METAL_COMMIT:-$(git rev-parse HEAD:third-party/tt-metal)}"'
        in script
    )
    assert '--expect-tt-metal-commit "$TT_METAL_COMMIT"' in script
    assert '--expect-ttnn-version "$TTNN_PYPI"' in script


@pytest.mark.parametrize(
    "workflow",
    [
        CALL_BUILD_WHEEL_IMAGES_WORKFLOW,
        CALL_BUILD_MANYLINUX_WHEELS_WORKFLOW,
        CALL_TEST_MANYLINUX_WHEELS_WORKFLOW,
        PUBLISH_PYPI_WORKFLOW,
        PUBLISH_S3_PYPI_WORKFLOW,
        SETUP_WHEEL_IMAGE_BUILD_ACTION,
    ],
)
def test_shared_manylinux_workflows_have_no_multiline_shell(
    workflow: Path,
) -> None:
    workflow_text = workflow.read_text()
    assert "run: |" not in workflow_text
    for line in workflow_text.splitlines():
        if line.lstrip().startswith("run:"):
            assert "${{ inputs." not in line


@pytest.mark.parametrize(
    "workflow",
    [
        CALL_BUILD_MANYLINUX_WHEELS_WORKFLOW,
        CALL_TEST_MANYLINUX_WHEELS_WORKFLOW,
        CALL_TTMETAL_LIGHT_WHEEL_WORKFLOW,
    ],
)
def test_manylinux_consumers_use_the_current_repository_namespace(
    workflow: Path,
) -> None:
    workflow_text = workflow.read_text()
    assert "ghcr.io/${{ github.repository }}/tt-lang-wheel-manylinux" in workflow_text
    assert "ghcr.io/tenstorrent/tt-lang/tt-lang-wheel-manylinux" not in workflow_text


def test_manylinux_builder_images_are_opt_in_for_docker_workflows() -> None:
    build_docker_workflow = CALL_BUILD_DOCKER_WORKFLOW.read_text()
    wheel_images_workflow = CALL_BUILD_WHEEL_IMAGES_WORKFLOW.read_text()
    ci_workflow = CI_WORKFLOW.read_text()
    s3_workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    pypi_workflow = PUBLISH_PYPI_WORKFLOW.read_text()
    manylinux_dockerfile = MANYLINUX_WHEEL_DOCKERFILE.read_text()

    # The core Docker image build never builds the wheel-builder images, so its
    # consumers (hardware tests, standard wheels) do not wait on them.
    assert "build-manylinux-wheel-images:" not in build_docker_workflow
    assert "build_manylinux_wheel_images" not in build_docker_workflow
    assert "build-wheel-manylinux-images.sh" not in build_docker_workflow

    # The wheel-builder images live in their own opt-in workflow.
    assert "build-manylinux-wheel-images:" in wheel_images_workflow
    assert "python_tag: [cp310, cp312]" in wheel_images_workflow
    assert "id: resolve" in wheel_images_workflow
    assert ".github/scripts/resolve-wheel-builder-images.sh" in wheel_images_workflow
    assert "cache-components:" in wheel_images_workflow
    assert "component: ttmetal" in wheel_images_workflow
    assert wheel_images_workflow.count("component: llvm") == 2
    assert "linux-amd64-ttmetal-cp312" in wheel_images_workflow
    assert "cache_tag: linux-amd64-llvm-cp310" in wheel_images_workflow
    assert "cache_tag: linux-amd64-llvm-cp312" in wheel_images_workflow
    assert "CACHE_REF: ${{ format(" in wheel_images_workflow
    assert "path: .wheel-workflow-src" in wheel_images_workflow
    assert "--workflow-source .wheel-workflow-src" in wheel_images_workflow
    assert "PYTHON_TAG: ${{ matrix.python_tag }}" in wheel_images_workflow
    assert "DOCKER_TAG: ${{ needs.resolve.outputs.docker-tag }}" in (
        wheel_images_workflow
    )
    assert (
        'build-wheel-manylinux-images.sh --python-tags "$PYTHON_TAG"'
        ' --image-tag "$DOCKER_TAG"'
    ) in wheel_images_workflow
    assert "publish-latest:" in wheel_images_workflow
    assert ".github/scripts/publish-wheel-builder-latest.sh" in wheel_images_workflow
    assert "--publish-latest" not in wheel_images_workflow
    assert "--build-parallel-level 2" in wheel_images_workflow
    assert "ARG WORKFLOW_SOURCE=." in manylinux_dockerfile
    assert (
        "COPY ${WORKFLOW_SOURCE}/.github/containers/"
        "CMakeLists.wheel-toolchain CMakeLists.txt"
    ) in manylinux_dockerfile
    assert "COPY ${WORKFLOW_SOURCE}/CMakeLists.txt" not in manylinux_dockerfile
    llvm_stage = manylinux_dockerfile.split("FROM base AS llvm-toolchain", 1)[1].split(
        "FROM base AS ttmetal-toolchain", 1
    )[0]
    ttmetal_stage = manylinux_dockerfile.split("FROM base AS ttmetal-toolchain", 1)[
        1
    ].split("FROM base AS wheel-builder", 1)[0]
    final_stage = manylinux_dockerfile.split("FROM base AS wheel-builder", 1)[1]
    assert "COPY requirements.txt requirements-runtime.txt" not in llvm_stage
    assert "COPY requirements.txt requirements-runtime.txt" not in ttmetal_stage
    assert (
        "COPY requirements.txt requirements-runtime.txt " "/tmp/ttlang-requirements/"
    ) in final_stage
    assert "-r /tmp/ttlang-requirements/requirements.txt" in final_stage
    assert "TTLANG_INSTALL_RUNTIME_REQUIREMENTS OFF" in (
        WHEEL_TOOLCHAIN_CMAKELISTS.read_text()
    )
    assert (
        "if(TTLANG_INSTALL_RUNTIME_REQUIREMENTS)"
        in (REPO_ROOT / "cmake" / "modules" / "BuildLLVM.cmake").read_text()
    )
    assert "ARG TTLANG_BUILD_PARALLEL_LEVEL" in manylinux_dockerfile
    assert (
        'ENV CMAKE_BUILD_PARALLEL_LEVEL="${TTLANG_BUILD_PARALLEL_LEVEL}"'
        in manylinux_dockerfile
    )

    # CI, S3 manylinux variants, and public PyPI all use the same builder.
    assert "uses: ./.github/workflows/call-build-wheel-images.yml" in ci_workflow
    assert "uses: ./.github/workflows/call-build-wheel-images.yml" in s3_workflow
    assert "needs.preflight.outputs.manylinux_selected == 'true'" in s3_workflow
    assert "uses: ./.github/workflows/call-build-wheel-images.yml" in pypi_workflow
    assert "uses: ./.github/workflows/call-build-manylinux-wheels.yml" in pypi_workflow
    assert "ttnn_dep_mode: pypi" in pypi_workflow


@pytest.mark.parametrize(
    ("component", "expected_marker"),
    [("llvm", "llvm"), ("tt-metal", "tt-metal")],
)
def test_wheel_toolchain_cmake_configures_only_selected_component(
    tmp_path: Path,
    component: str,
    expected_marker: str,
) -> None:
    source_dir = tmp_path / "source"
    module_dir = source_dir / "cmake" / "modules"
    module_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text(WHEEL_TOOLCHAIN_CMAKELISTS.read_text())
    (module_dir / "TTLangToolchainComponent.cmake").write_text(
        TOOLCHAIN_COMPONENT_MODULE.read_text()
    )
    (module_dir / "TTLangToolchainOptions.cmake").write_text(
        TOOLCHAIN_OPTIONS_MODULE.read_text()
    )
    (module_dir / "TTLangUtils.cmake").write_text(
        "function(ttlang_pip_install_requirements)\nendfunction()\n"
    )
    for module_name in ("TTLangCompilerSetup", "TTLangPython"):
        (module_dir / f"{module_name}.cmake").write_text("")
    (module_dir / "BuildLLVM.cmake").write_text(
        'file(WRITE "${CMAKE_BINARY_DIR}/configured-component" "llvm")\n'
    )
    (module_dir / "BuildTTMetal.cmake").write_text(
        'file(WRITE "${CMAKE_BINARY_DIR}/configured-component" "tt-metal")\n'
    )

    build_dir = tmp_path / "build"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_C_COMPILER=cc",
            "-DCMAKE_CXX_COMPILER=c++",
            f"-DTTLANG_TOOLCHAIN_COMPONENT={component}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (build_dir / "configured-component").read_text() == expected_marker


def test_wheel_toolchain_cmake_rejects_full_project_configuration(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    module_dir = source_dir / "cmake" / "modules"
    module_dir.mkdir(parents=True)
    (source_dir / "CMakeLists.txt").write_text(WHEEL_TOOLCHAIN_CMAKELISTS.read_text())
    for module in (
        TOOLCHAIN_OPTIONS_MODULE,
        REPO_ROOT / "cmake" / "modules" / "TTLangUtils.cmake",
        REPO_ROOT / "cmake" / "modules" / "TTLangCompilerSetup.cmake",
    ):
        (module_dir / module.name).write_text(module.read_text())

    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(tmp_path / "build"),
            "-DCMAKE_C_COMPILER=cc",
            "-DCMAKE_CXX_COMPILER=c++",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires llvm or tt-metal" in result.stdout + result.stderr


def test_setup_py_removes_stale_native_payloads_before_wheel_install() -> None:
    setup_py = SETUP_PY.read_text()

    assert 'mlir_libs_dir = install_dir / "ttl" / "_mlir_libs"' in setup_py
    assert "shutil.rmtree(mlir_libs_dir)" in setup_py
    assert setup_py.index("self._remove_stale_native_payloads(install_dir)") < (
        setup_py.index('"cmake",\n                "--install"')
    )


def test_s3_workflow_publishes_only_from_main_ref() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()
    resolver = RESOLVE_S3_PUBLISH_INPUTS.read_text()
    trigger_block = workflow.split("\nconcurrency:", maxsplit=1)[0]

    assert "push:" not in trigger_block
    assert "tags:" not in trigger_block
    assert "- 'v[0-9]+.[0-9]+.[0-9]+'" not in trigger_block
    assert '"${GITHUB_REF:-}" != refs/heads/main' in resolver
    assert "Publishing is restricted to refs/heads/main" in resolver
    assert "Non-main dry runs must provide docker_tag" in resolver
    assert (
        "needs.preflight.outputs.dry_run == 'true' || github.ref == 'refs/heads/main'"
        in workflow
    )


def test_check_wheel_ttnn_metadata_matches_requirement_name(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\nRequires-Dist: ttnn-foo >= 1\n",
    )

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA, "--mode", "pypi", "--dist-dir", str(dist_dir)
    )

    assert result.returncode != 0
    assert "default wheel metadata must require ttnn" in result.stderr


def test_check_wheel_ttnn_metadata_requires_exact_pypi_version(
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\nRequires-Dist: ttnn == 1.2.3\n",
    )

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA,
        "--mode",
        "pypi",
        "--dist-dir",
        str(dist_dir),
        "--expect-ttnn-version",
        "1.2.4",
    )

    assert result.returncode != 0
    assert "ttnn dependency mismatch: expected ==1.2.4, got ==1.2.3" in result.stderr


def test_check_wheel_ttnn_metadata_accepts_exact_pypi_version(
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\nRequires-Dist: ttnn == 1.2.3\n",
    )

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA,
        "--mode",
        "pypi",
        "--dist-dir",
        str(dist_dir),
        "--expect-ttnn-version",
        "1.2.3",
    )

    assert result.returncode == 0


def test_check_wheel_ttnn_metadata_rejects_external_payload(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525+light-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttnn/__init__.py", "")

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA, "--mode", "external", "--dist-dir", str(dist_dir)
    )

    assert result.returncode != 0
    assert "external wheel must not bundle a ttnn payload" in result.stderr


def test_check_wheel_ttnn_metadata_checks_tt_metal_commit(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525+light-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    _add_config_py(wheel_path, tt_metal_commit="aaaaaaaa")

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA,
        "--mode",
        "external",
        "--dist-dir",
        str(dist_dir),
        "--expect-tt-metal-commit",
        "bbbbbbbb",
    )

    assert result.returncode != 0
    assert "tt-metal provenance mismatch" in result.stderr


def test_check_light_metapackage_parses_requires_dist(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang_light-0.71.0.dev20260525-py3-none-any.whl",
        (
            "Metadata-Version: 2.1\n"
            "Requires-Python: >=3.10\n"
            "Requires-Dist: tt-lang == 0.71.0.dev20260525+light ; "
            'python_version >= "3.12"\n'
        ),
    )

    result = _run_script(
        CHECK_LIGHT_METAPACKAGE,
        "--dist-dir",
        str(dist_dir),
        "--expect-ttlang-version",
        "0.71.0.dev20260525+light",
    )

    assert result.returncode == 0, result.stderr


def test_check_light_metapackage_requires_python_metadata(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang_light-0.71.0.dev20260525-py3-none-any.whl",
        (
            "Metadata-Version: 2.1\n"
            "Requires-Dist: tt-lang == 0.71.0.dev20260525+light\n"
        ),
    )

    result = _run_script(
        CHECK_LIGHT_METAPACKAGE,
        "--dist-dir",
        str(dist_dir),
        "--expect-ttlang-version",
        "0.71.0.dev20260525+light",
    )

    assert result.returncode != 0
    assert "Requires-Python: >=3.10" in result.stderr


def test_compute_nightly_version_uses_latest_stable_tag(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "file.txt").write_text("first\n")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.2.3"], cwd=tmp_path, check=True)

    subprocess.run(["git", "checkout", "-b", "release"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("release\n")
    subprocess.run(["git", "commit", "-am", "release"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.2.4"], cwd=tmp_path, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "second"], cwd=tmp_path, check=True)

    result = _run_script(COMPUTE_NIGHTLY_VERSION, cwd=tmp_path)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"1.2.4.dev{today}"


def _write_fake_ttnn(root: Path, *, with_native_libs: bool) -> None:
    package = root / "ttnn"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if with_native_libs:
        (package / "_ttnn.so").write_bytes(b"")
        build_lib = package / "build" / "lib"
        build_lib.mkdir(parents=True)
        (build_lib / "_ttnncpp.so").write_bytes(b"")
        (build_lib / "libtt_metal.so").write_bytes(b"")


def _env_with_pythonpath(path: Path) -> dict[str, str]:
    env = {**os.environ}
    env["PYTHONPATH"] = str(path)
    return env


def _env_with_pythonpath_and_ldd_output(
    path: Path,
    *,
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
) -> dict[str, str]:
    env = _env_with_pythonpath(path)
    script = (
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"raise SystemExit({exit_code})\n"
    )
    env["TTLANG_LDD_COMMAND"] = shlex.join([sys.executable, "-c", script])
    return env


def _env_with_pythonpath_and_ldd(path: Path) -> dict[str, str]:
    ttnncpp_path = path / "ttnn" / "build" / "lib" / "_ttnncpp.so"
    return _env_with_pythonpath_and_ldd_output(
        path,
        stdout=f"\t_ttnncpp.so => {ttnncpp_path} (0x00000000)\n",
    )


def test_check_installed_ttnn_pypi_fails_when_ttnn_absent(tmp_path: Path) -> None:
    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "pypi",
        env=_env_with_pythonpath(tmp_path),
    )
    assert result.returncode != 0
    assert "did not install its ttnn dependency" in result.stderr


def test_check_installed_ttnn_pypi_passes_when_ttnn_present(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=False)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "pypi",
        env=_env_with_pythonpath(tmp_path),
    )
    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_external_passes_when_ttnn_absent(
    tmp_path: Path,
) -> None:
    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "external",
        env=_env_with_pythonpath(tmp_path),
    )
    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_external_fails_when_ttnn_present(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=False)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "external",
        env=_env_with_pythonpath(tmp_path),
    )

    assert result.returncode != 0
    assert "external wheel unexpectedly installed ttnn" in result.stderr


def test_check_installed_ttnn_bundled_passes_with_required_files(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "bundled",
        env=_env_with_pythonpath_and_ldd(tmp_path),
    )

    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_bundled_fails_when_files_missing(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=False)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "bundled",
        env=_env_with_pythonpath(tmp_path),
    )

    assert result.returncode != 0
    assert "bundled ttnn is missing files" in result.stderr


def test_check_installed_ttnn_bundled_fails_when_ldd_exits_nonzero(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="",
        stderr="ldd: cannot read object\n",
        exit_code=1,
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "ldd failed for bundled ttnn extension" in result.stderr


def test_check_installed_ttnn_bundled_fails_on_unresolved_libraries(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="\tlibmissing.so => not found\n",
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "unresolved libraries" in result.stderr


def test_check_installed_ttnn_bundled_fails_when_ttnncpp_resolves_elsewhere(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    # ldd reports _ttnncpp.so resolved to a system path rather than the bundled one.
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="\t_ttnncpp.so => /usr/lib/_ttnncpp.so (0x00000000)\n",
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "does not resolve _ttnncpp.so from" in result.stderr


def _write_bundled_wheel(dist_dir: Path, *, complete: bool) -> Path:
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttnn/__init__.py", "")
        wheel.writestr("ttnn/_ttnn.so", b"")
        if complete:
            wheel.writestr("triage/inspector.capnp", "")
            wheel.writestr("triage/requirements.txt", "")
            wheel.writestr("triage/triage.py", "")
            wheel.writestr("ttnn/build/lib/_ttnncpp.so", b"")
            wheel.writestr("ttnn/build/lib/libtt_metal.so", b"")
    return wheel_path


def test_check_wheel_bundled_payload_accepts_complete_wheel(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_bundled_wheel(dist_dir, complete=True)

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode == 0, result.stderr


def test_check_wheel_bundled_payload_rejects_missing_files(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_bundled_wheel(dist_dir, complete=False)

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "bundled wheel is missing" in result.stderr


def test_check_wheel_bundled_payload_rejects_empty_dist(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "expected one tt-lang wheel" in result.stderr


def test_internal_wheel_metadata_fails_when_git_version_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "packaging"))
    import internal_wheel_metadata

    monkeypatch.delenv("TTLANG_VERSION_OVERRIDE", raising=False)

    def fail_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", fail_git)

    with pytest.raises(SystemExit, match="failed to derive internal wheel version"):
        internal_wheel_metadata.get_version(REPO_ROOT)
