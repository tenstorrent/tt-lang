# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Any

project = "TT-Lang"
copyright = "2025 Tenstorrent AI ULC"
author = "TT-Lang Team"
release = ""

_docs_dir = Path(__file__).resolve().parent
_repo_root = _docs_dir.parent.parent

# Use Tenstorrent docsite shared theme when building inside tenstorrent.github.io.
_local_static = _docs_dir / "_static"
_shared_dir = _repo_root.parent.parent / "shared"
if _shared_dir.is_dir():
    _theme_static_paths = [str(_shared_dir / "_static"), str(_local_static)]
    _theme_templates = str(_shared_dir / "_templates")
    _theme_logo = str(_shared_dir / "images" / "tt_logo.svg")
    _theme_favicon = str(_shared_dir / "images" / "favicon.png")
else:
    _theme_static_paths = [str(_local_static)]
    _theme_templates = str(_docs_dir / "_templates")
    _theme_logo = str(_local_static / "images" / "tt_logo.svg")
    _theme_favicon = str(_local_static / "images" / "favicon.png")

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
html_theme_options = {
    "collapse_navigation": False,
    "titles_only": True,
    "navigation_depth": 2,
}
html_logo = _theme_logo
html_favicon = _theme_favicon
html_static_path = _theme_static_paths
templates_path = [_theme_templates]
html_last_updated_fmt = "%b %d, %Y"
html_css_files = ["https://docs.tenstorrent.com/_static/tt_theme.css"]

html_context = {
    "versions": None,
    "logo_link_url": os.environ.get("homepage", "https://docs.tenstorrent.com/"),
    "search_site_base_url": "https://docs.tenstorrent.com/tt-lang/",
}


def autodoc_skip_member(
    app: Any, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool:
    if hasattr(obj, "__autodoc_skip__") and obj.__autodoc_skip__:
        return True
    return skip


def setup(app: Any) -> None:
    app.add_css_file("tt_theme.css")
    app.connect("autodoc-skip-member", autodoc_skip_member)
