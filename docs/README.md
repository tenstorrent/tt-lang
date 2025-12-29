# Documentation

## Prerequisites
- Docs are disabled by default. Enable with `-DTTLANG_ENABLE_DOCS=ON` when configuring CMake.
- On enable, CMake will `pip install` Sphinx, myst-parser, and sphinx-rtd-theme into the active Python.
- If you prefer manual install: `python -m pip install sphinx myst-parser sphinx-rtd-theme`.
- Activate the project env when building docs: `source build/env/activate`.

## Build and view
- Activate the build env if needed: `source build/env/activate`.
- Configure with docs enabled: `cmake -G Ninja -B build -DTTLANG_ENABLE_DOCS=ON ...`.
- Build HTML with CMake: `cmake --build build --target ttlang-docs` (outputs to `build/docs/sphinx/_build/html`).
- Serve locally from build output: `python -m http.server 8000 -d build/docs/sphinx/_build/html` and open http://localhost:8000/index.html.

## Add or update pages
- Write content in MyST markdown under `docs/sphinx/`.
- Update `docs/sphinx/index.rst` to include new pages in the appropriate toctree (User Guide or Contributor Guide).
- Place shared assets in `docs/sphinx/images/` and reference them with relative paths.
- Keep language concise and precise; end sentences with periods; avoid fluff.
- Rebuild with `cmake --build build --target ttlang-docs` to verify the navigation and formatting.
