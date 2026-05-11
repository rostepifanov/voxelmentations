# Improvements Backlog

This file tracks planned improvements for `voxelmentations`.

Priority levels:
- `P0`: correctness, API consistency, broken behavior
- `P1`: maintainability, packaging, tests, developer experience
- `P2`: performance, advanced features, long-term roadmap

## P0

- [ ] Finish `ElasticDistort`, because it currently does not perform a real distortion.
- [ ] Implement `apply_to_points` for `GridDistort`, or explicitly drop `points` support for it.
- [ ] Implement `apply_to_points` for `ElasticDistort`, or explicitly drop `points` support for it.
- [ ] Fix `PatchShuffle` so it shuffles all voxels inside a patch, not just along the first axis.
- [ ] Stop mutating the input dictionary in `from_dict`.
- [ ] Add a guard in `OneOf` for the case where the sum of transform probabilities is zero.
- [ ] Replace `assert`-based user-facing checks with explicit `ValueError` or `KeyError`.
- [ ] Validate `p` in all transforms and compositions to ensure it stays in `[0, 1]`.
- [ ] Validate `always_apply` as a real boolean.
- [ ] Validate that `nplanes` is not larger than the selected axis size.
- [ ] Validate that patch sizes are not larger than the input volume.
- [x] Rename `Tranpose` and `AxialPlaneTranpose` to `Transpose` and `AxialPlaneTranspose`.
- [ ] Update `README.md` so the documented transform list matches the real public API.
- [ ] Fix package description and metadata so the library is described as NumPy/SciPy-first instead of PyTorch-first.
- [ ] Restore a working local test environment so the project can be verified out of the box.

## P1

- [ ] Migrate packaging from `setup.py` to `pyproject.toml`.
- [ ] Add GitHub Actions for tests across multiple Python versions.
- [ ] Add `ruff` and baseline linting.
- [ ] Add type hints for the public API.
- [ ] Add `py.typed` once typing support becomes systematic.
- [ ] Create a dedicated document for shape conventions.
- [ ] Clearly document supported shapes such as `(H, W, D)` and `(H, W, D, C)`.
- [ ] Clearly document axis ordering and keep it consistent across code, docs, and examples.
- [ ] Document `mask` and `points` behavior for every transform.
- [ ] Add a "supported targets by transform" table.
- [ ] Define a deprecation policy for future public API changes.
- [ ] Make `from_dict` fail with a clear error when an unknown class name is encountered.
- [ ] Version the serialization format independently from the package version.
- [ ] Add round-trip serialization tests for all public transforms.
- [ ] Add deterministic tests for fixed random seeds.
- [x] Add inverse tests for `Flip`, `Rotate90`, and `Transpose`.
- [ ] Add edge-case tests for very small volumes.
- [ ] Add edge-case tests for single-channel and no-channel inputs.
- [ ] Add edge-case tests for anisotropic shapes such as `64x64x8`.
- [ ] Add edge-case tests for zero-width or degenerate parameter ranges.

## P2

- [ ] Relax validators so they accept NumPy scalar types such as `np.int64` and `np.float32`.
- [ ] Move common parameter checks into a more systematic validation layer.
- [ ] Add a `@contiguous`-style decorator for operations that benefit from C-contiguous arrays.
- [ ] Add a `@clipped`-style decorator for safe dtype-range clipping after arithmetic operations.
- [ ] Build a unified low-level arithmetic API for operations like `add`, `multiply`, `gamma`, `contrast`, and `normalize`.
- [ ] Build a unified low-level statistics API for `mean`, `std`, and percentile-based operations.
- [ ] Introduce explicit fast paths for `float32`.
- [ ] Define an explicit dtype conversion policy instead of relying on scattered implicit conversions.
- [ ] Reduce unnecessary array copies in the functional layer where safe.
- [ ] Add benchmarks for `64^3`, `128^3`, and `256x256x128` volumes in mono-channel and multi-channel configurations.
- [ ] Compare performance across NumPy, SciPy, and OpenCV backend paths.
- [ ] Add local random state support on the pipeline instead of relying only on global `np.random`.
- [ ] Add replay support for reusing the exact same sampled transform parameters on another sample.
- [ ] Add basic spatial transforms such as `CenterCrop3D`, `RandomCrop3D`, and `Resize3D`.
- [ ] Add medical-aware support for metadata such as `spacing`, starting with transforms like `ResampleSpacing` and `CropForeground`.
