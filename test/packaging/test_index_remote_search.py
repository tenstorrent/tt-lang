# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path


def _load_index_remote_search():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "index_remote_search.py"
    )
    spec = importlib.util.spec_from_file_location("index_remote_search", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_documents_indexes_article_body_only(tmp_path):
    module = _load_index_remote_search()
    html_path = tmp_path / "index.html"
    html_path.write_text(
        """
        <html>
          <head><title>Compiler Options</title></head>
          <body>
            <nav>Repeated navigation text</nav>
            <div itemprop="articleBody">
              <h1>Compiler Options</h1>
              <p>Use --ttl-maximize-dst to configure DST scheduling.</p>
            </div>
            <footer>Repeated footer text</footer>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    docs = module._build_documents(
        tmp_path,
        "https://docs.example.com/tt-lang/",
        "docs-tenstorrent",
        "latest",
        "tt-lang",
    )

    assert docs == [
        {
            "id": "docs-tenstorrent:tt-lang:latest:index.html",
            "title": "Compiler Options",
            "body": "Compiler Options Use --ttl-maximize-dst to configure DST scheduling.",
            "url": "https://docs.example.com/tt-lang/index.html",
        }
    ]


def test_build_documents_excludes_chrome_after_void_elements(tmp_path):
    # Void elements get no HTMLParser end event; the body must still end at
    # </div itemprop="articleBody"> and not leak the trailing footer.
    module = _load_index_remote_search()
    html_path = tmp_path / "page.html"
    html_path.write_text(
        """
        <html>
          <head><title>Diagram Page</title></head>
          <body>
            <nav>Repeated navigation text</nav>
            <div itemprop="articleBody">
              <h1>Diagram Page</h1>
              <img src="diagram.png">
              <p>Text after the image.</p>
            </div>
            <footer>Repeated footer text</footer>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    (doc,) = module._build_documents(
        tmp_path,
        "https://docs.example.com/tt-lang/",
        "docs-tenstorrent",
        "latest",
        "tt-lang",
    )

    assert doc["body"] == "Diagram Page Text after the image."
    assert "navigation" not in doc["body"]
    assert "footer" not in doc["body"]
