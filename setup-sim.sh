#!/usr/bin/env bash
# Personal sim-only setup for tt-lang repos (NOT part of any repo).
#
# Two ways to use it:
#
#   ./setup-sim.sh [REPO...]      # EXECUTE: for each repo, configure a sim-only
#                                 #   CMake build, build it, and install the dev
#                                 #   tools the sim workflow needs (pytest-xdist,
#                                 #   pre-commit) into the repo's venv. Does NOT
#                                 #   touch your interactive shell.
#
#   source setup-sim.sh [REPO]    # SOURCE: build REPO if it isn't built yet, then
#                                 #   activate its env (build/env/activate) in your
#                                 #   CURRENT shell. REPO defaults to the current
#                                 #   directory. This is how you "switch into" the
#                                 #   tt-lang sim environment.
#
# IMPORTANT: there is intentionally no top-level `set -e/-u/-o pipefail`. Those
# are shell-global and, when this file is *sourced*, would leak into and corrupt
# your interactive shell. In particular `set -u` (nounset) makes the generated
# build/env/activate abort on its `${TTLANG_ENV_ACTIVATED}` guard. Strict mode is
# therefore confined to the build subshell below, which never leaks to the caller.

# Build + install dev tools for one already-resolved repo path. Runs entirely in
# a subshell so `set -euo pipefail` and the venv activation stay contained.
_setup_sim_build() {
  local repo="$1"
  (
    set -euo pipefail
    cd "$repo"
    cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON
    cmake --build build
    # shellcheck disable=SC1091
    source build/env/activate
    pip install pytest-xdist pre-commit
    pre-commit install
  )
}

# Echo the absolute path of $1 if it looks like a tt-lang checkout, else fail.
_setup_sim_resolve() {
  local repo="$1"
  if [[ ! -d "$repo" ]]; then
    echo "SKIP $repo (not a directory)" >&2
    return 1
  fi
  repo="$(cd "$repo" && pwd)"
  if [[ ! -f "$repo/CMakeLists.txt" ]]; then
    echo "SKIP $repo (no CMakeLists.txt; not a tt-lang checkout)" >&2
    return 1
  fi
  printf '%s\n' "$repo"
}

# Detect whether this file is being sourced (activate in caller) or executed
# (build across repos). Works under both zsh and bash.
_setup_sim_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file) _setup_sim_sourced=1 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
  (return 0 2>/dev/null) && _setup_sim_sourced=1
fi

if [ "$_setup_sim_sourced" -eq 1 ]; then
  # SOURCED: activate a single repo's env in the caller's shell. Use `return`
  # (never `exit`) so a failure does not close the interactive shell.
  _setup_sim_repo="$(_setup_sim_resolve "${1:-.}")" || return 1
  echo "=== setup-sim (activate): ${_setup_sim_repo} ==="
  if [ ! -f "${_setup_sim_repo}/build/env/activate" ]; then
    echo "no build yet -- building first ..."
    _setup_sim_build "${_setup_sim_repo}" || {
      echo "FAIL build ${_setup_sim_repo}" >&2
      unset _setup_sim_repo _setup_sim_sourced
      return 1
    }
  fi
  # Activate at top level (not in a subshell) so it affects THIS shell.
  # shellcheck disable=SC1091
  source "${_setup_sim_repo}/build/env/activate"
  echo "activated: ${_setup_sim_repo}"
  unset _setup_sim_repo _setup_sim_sourced
  unset -f _setup_sim_build _setup_sim_resolve 2>/dev/null || true
else
  # EXECUTED: build + install across all given repos (default: current dir).
  _setup_sim_rc=0
  for _setup_sim_arg in "${@:-.}"; do
    if _setup_sim_abs="$(_setup_sim_resolve "$_setup_sim_arg")" \
      && _setup_sim_build "$_setup_sim_abs"; then
      echo "OK   $_setup_sim_abs"
    else
      echo "FAIL $_setup_sim_arg" >&2
      _setup_sim_rc=1
    fi
  done
  exit "$_setup_sim_rc"
fi
