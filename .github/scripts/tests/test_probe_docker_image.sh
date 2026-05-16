#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/probe-docker-image.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SCRIPT="$SCRIPTS_DIR/probe-docker-image.sh"

# Install a fake `docker` on PATH whose `manifest inspect` exit status is
# controlled by $FAKE_DOCKER_MISSING (1 -> exit 1, else exit 0). Echoes the
# bindir; caller cleans up.
make_docker_mock() {
    local bindir
    bindir=$(mktemp -d)
    cat > "$bindir/docker" <<'EOF'
#!/usr/bin/env bash
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
assert_eq "$(run_probe v1.2.0 0)" "0|false"

start_case "uplift tag, image present"
assert_eq "$(run_probe v1.2.0-uplift-abcd1234 0)" "0|false"

# --- Image missing, uplift form -> needs_rebuild=true, exit 0 ---
start_case "uplift tag, image missing"
assert_eq "$(run_probe v1.2.0-uplift-abcd1234 1)" "0|true"

# --- Image missing, bare release form -> refuse with exit 1 ---
start_case "bare release tag, image missing -> refuse"
assert_eq "$(run_probe v1.2.0 1)" "1|"

start_case "rc release tag, image missing -> refuse"
assert_eq "$(run_probe v1.2.0-rc1 1)" "1|"

start_case "dev release tag, image missing -> refuse"
assert_eq "$(run_probe v1.2.0-dev20260515 1)" "1|"

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
