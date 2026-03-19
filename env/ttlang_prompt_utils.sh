# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# tt-lang prompt utilities — sourced by activate after the venv is loaded.
#
# Prepends "(ᴛᴛʟᴀɴɢ) " to PS1 without clobbering the user's shell theme,
# and ensures the indicator disappears on `deactivate`.
#
# Usage (from activate):
#   VIRTUAL_ENV_DISABLE_PROMPT=1
#   source "<venv>/bin/activate"
#   unset VIRTUAL_ENV_DISABLE_PROMPT
#   source "<this-file>"

if [ -n "${ZSH_VERSION:-}" ]; then
  # --- zsh ---------------------------------------------------------------
  # With prompt_subst (enabled by oh-my-zsh, powerlevel10k, etc.), parameter
  # expansions inside single-quoted PROMPT are re-evaluated every prompt.
  # ${TTLANG_ENV_ACTIVATED:+(ᴛᴛʟᴀɴɢ) } expands to the tag when the var is
  # set and to nothing after `deactivate` unsets it.
  #
  # Guard against double-prepend on re-activation.
  case "$PROMPT" in
    *'${TTLANG_ENV_ACTIVATED:+'*) ;;
    *) PROMPT='${TTLANG_ENV_ACTIVATED:+(ᴛᴛʟᴀɴɢ) }'$PROMPT ;;
  esac

  # Wrap deactivate to also unset TTLANG_ENV_ACTIVATED (venv's deactivate
  # does not know about it).  The prompt tag disappears automatically on the
  # next prompt evaluation.
  _ttlang_orig_deactivate="$(functions deactivate)"
  deactivate() {
    unset TTLANG_ENV_ACTIVATED
    eval "${_ttlang_orig_deactivate}" && deactivate "$@"
    unset _ttlang_orig_deactivate
  }
else
  # --- bash / POSIX ------------------------------------------------------
  # bash does not re-evaluate PS1 parameter expansions, so save/restore.
  _TTLANG_OLD_PS1="${PS1:-}"
  PS1="(ᴛᴛʟᴀɴɢ) ${PS1:-}"
  export PS1

  _ttlang_orig_deactivate="$(type deactivate | tail -n +2)"
  eval "deactivate() {
    unset TTLANG_ENV_ACTIVATED
    PS1=\"\${_TTLANG_OLD_PS1}\"
    export PS1
    unset _TTLANG_OLD_PS1
    ${_ttlang_orig_deactivate}
    deactivate \"\$@\"
  }"
fi
