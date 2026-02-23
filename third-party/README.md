# Third-party dependencies uplift instructions

Version pins for tt-lang's external dependencies.

## tt-mlir
`tt-mlir.commit` — pinned tt-mlir commit SHA. Used by CI to check out the
correct tt-mlir source when building tt-lang from source (see `call-build.yml`).

`tt-mlir-docker-tag` — pinned tag for the `tt-mlir-ci-ubuntu-22-04` base image
used in Docker builds and CI container jobs. Update this to switch the tt-mlir
image version across all build scripts and workflows at once.

To bump tt-mlir: update both files to the new commit/tag, then commit.
