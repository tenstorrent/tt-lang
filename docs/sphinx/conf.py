# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from typing import Any

project = "tt-lang"
copyright = "2025 Tenstorrent AI ULC"
author = "TT-Lang Team"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "private-members": False,
}
autodoc_docstring_signature = True
autodoc_typehints = "description"
autodoc_member_order = "alphabetical"

napoleon_numpy_docstring = True

autosummary_generate = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"


def autodoc_skip_member(
    app: Any, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool:
    if hasattr(obj, "__autodoc_skip__") and obj.__autodoc_skip__:
        return True
    return skip


def setup(app: Any) -> None:
    app.connect("autodoc-skip-member", autodoc_skip_member)
