#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Install and hardware-test the cp312 core wheel from a shared manylinux wheel
# artifact.

set -eu

dist_dir=dist
ttnn_dep_mode=""
python_bin=/opt/python/cp312-cp312/bin/python
repo_root=""
tutorial_script=""

usage() {
    echo "Usage: $0 --ttnn-dep-mode pypi|external [--dist-dir <dir>] [--python <path>] [--repo-root <dir>] [--tutorial-script <path>]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dist-dir)
            [ "$#" -ge 2 ] || usage
            dist_dir="$2"
            shift 2
            ;;
        --ttnn-dep-mode)
            [ "$#" -ge 2 ] || usage
            ttnn_dep_mode="$2"
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || usage
            python_bin="$2"
            shift 2
            ;;
        --repo-root)
            [ "$#" -ge 2 ] || usage
            repo_root="$2"
            shift 2
            ;;
        --tutorial-script)
            [ "$#" -ge 2 ] || usage
            tutorial_script="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

case "$ttnn_dep_mode" in
    pypi | external) ;;
    *) usage ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$repo_root" ]; then
    if [ -n "${GITHUB_WORKSPACE:-}" ]; then
        repo_root="$GITHUB_WORKSPACE"
    else
        repo_root="$(git rev-parse --show-toplevel)"
    fi
fi
tutorial_script="${tutorial_script:-$script_dir/run-tutorials.sh}"
core_wheel="$(find "$dist_dir" -maxdepth 1 -type f \
    -name 'tt_lang-*-cp312-cp312-manylinux_2_34_x86_64.whl' \
    -print -quit)"
if [ -z "$core_wheel" ]; then
    echo "cp312 manylinux_2_34 tt-lang wheel not found in $dist_dir" >&2
    exit 1
fi

test_venv="$(mktemp -d /tmp/test-manylinux-wheel.XXXXXX)"
trap 'rm -rf "$test_venv"' EXIT
"$python_bin" -m venv "$test_venv"
test_python="$test_venv/bin/python"

"$test_python" -m pip install \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "$core_wheel"
"$test_python" "$script_dir/check-installed-ttnn.py" --mode "$ttnn_dep_mode"
if [ "$ttnn_dep_mode" = pypi ]; then
    "$test_venv/bin/tt-lang-setup-sfpi"
fi
"$test_python" "$script_dir/smoke-test-wheel.py"

if [ ! -x "$test_venv/bin/tt-triage" ]; then
    echo "tt-triage was not installed by $core_wheel" >&2
    exit 1
fi

PATH="$test_venv/bin:$PATH" "$tutorial_script" "$repo_root"
