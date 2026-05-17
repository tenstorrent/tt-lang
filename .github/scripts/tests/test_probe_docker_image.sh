#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/probe-docker-image.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SCRIPT="$SCRIPTS_DIR/probe-docker-image.sh"

# Synthetic tags exercising the three forms the script must classify. Versions
# are chosen well outside any real release range so the literals can never be
# confused with a production tag.
#   BARE_TAG    -> release form (must refuse rebuild when image missing)
#   UPLIFT_TAG  -> uplift form  (must allow rebuild when image missing)
#   RC_TAG      -> release pre-release form (also bare-release class)
#   DEV_TAG     -> release dev form         (also bare-release class)
# The ird image reference used by the script under test:
IRD_IMAGE_BASE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04"
BARE_TAG="v99.99.99"
UPLIFT_TAG="v99.99.99-uplift-abcd1234"
RC_TAG="v99.99.99-rc1"
DEV_TAG="v99.99.99-dev20260515"

# Install a fake `docker` on PATH whose `manifest inspect` exit status is
# controlled by $FAKE_DOCKER_MISSING (1 -> exit 1, else exit 0). Records the
# full argv to $FAKE_DOCKER_ARGS so tests can assert on the image reference.
# Echoes the bindir; caller cleans up.
make_docker_mock() {
    local bindir
    bindir=$(mktemp -d)
    cat > "$bindir/docker" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${FAKE_DOCKER_ARGS:-}" ]]; then
    printf '%s\n' "$*" >> "$FAKE_DOCKER_ARGS"
fi
if [[ "$1" == "manifest" && "$2" == "inspect" ]]; then
    [[ "${FAKE_DOCKER_MISSING:-0}" == "1" ]] && exit 1
    exit 0
fi
exit 0
EOF
    chmod +x "$bindir/docker"
    echo "$bindir"
}

# Run the script with a mocked docker. Args: <tag> <missing 0|1>.
# Echoes "<rc>|<needs_rebuild value or empty>".
run_probe() {
    local tag="$1"
    local missing="$2"
    local bindir gh_out rc val
    bindir=$(make_docker_mock)
    gh_out=$(mktemp)
    set +e
    PATH="$bindir:$PATH" FAKE_DOCKER_MISSING="$missing" \
        GITHUB_OUTPUT="$gh_out" "$SCRIPT" "$tag" >/dev/null 2>&1
    rc=$?
    set -e
    val=$(grep '^needs_rebuild=' "$gh_out" 2>/dev/null | sed 's/^needs_rebuild=//' || true)
    rm -rf "$bindir" "$gh_out"
    echo "${rc}|${val}"
}

# --- Image present -> needs_rebuild=false, exit 0 ---
start_case "bare tag, image present"
assert_eq "$(run_probe "$BARE_TAG" 0)" "0|false"

start_case "uplift tag, image present"
assert_eq "$(run_probe "$UPLIFT_TAG" 0)" "0|false"

# --- Image missing, uplift form -> needs_rebuild=true, exit 0 ---
start_case "uplift tag, image missing"
assert_eq "$(run_probe "$UPLIFT_TAG" 1)" "0|true"

# --- Image missing, bare release form -> refuse with exit 1 ---
start_case "bare release tag, image missing -> refuse"
assert_eq "$(run_probe "$BARE_TAG" 1)" "1|"

start_case "rc release tag, image missing -> refuse"
assert_eq "$(run_probe "$RC_TAG" 1)" "1|"

start_case "dev release tag, image missing -> refuse"
assert_eq "$(run_probe "$DEV_TAG" 1)" "1|"

# --- Mock invoked with the expected image reference ---
start_case "probes the ird image at the given tag"
bindir=$(make_docker_mock)
gh_out=$(mktemp)
args_file=$(mktemp)
PATH="$bindir:$PATH" FAKE_DOCKER_MISSING=0 FAKE_DOCKER_ARGS="$args_file" \
    GITHUB_OUTPUT="$gh_out" "$SCRIPT" "$BARE_TAG" >/dev/null 2>&1
assert_matches "$(cat "$args_file")" \
    "manifest inspect ${IRD_IMAGE_BASE}:${BARE_TAG}" \
    "docker manifest inspect invoked with the ird image at $BARE_TAG"
rm -rf "$bindir" "$gh_out" "$args_file"

# --- Missing arg -> exit non-zero ---
start_case "no tag argument -> non-zero exit"
set +e
bindir=$(make_docker_mock)
GITHUB_OUTPUT=$(mktemp) PATH="$bindir:$PATH" "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
rm -rf "$bindir"
assert_neq "$rc" "0"

finish_tests
