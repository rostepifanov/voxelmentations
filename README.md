# Voxelmentations

![Python version support](https://img.shields.io/pypi/pyversions/voxelmentations)
[![PyPI version](https://badge.fury.io/py/voxelmentations.svg)](https://badge.fury.io/py/voxelmentations)
[![Downloads](https://pepy.tech/badge/voxelmentations/month)](https://pepy.tech/project/voxelmentations?versions=0.0.*)

Voxelmentations is a Python library for 3d image (voxel) augmentation. Voxel augmentation is used in deep learning to increase the quality of trained models. The purpose of voxel augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some augmentations from voxelmentations to create new voxel from the original one:

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [A simple example](#a-simple-example)
- [Supported shapes](#supported-shapes)
- [Additional targets](#additional-targets)
- [List of augmentations](#list-of-augmentations)
- [Citing](#citing)

## Authors
[**Rostislav Epifanov** — Researcher in Novosibirsk]()

## Installation
Installation from PyPI:

```
pip install voxelmentations
```

Installation from GitHub:

```
pip install git+https://github.com/rostepifanov/voxelmentations
```

## A simple example
```python
import numpy as np
import voxelmentations as V

# Declare an augmentation pipeline
transform = V.Sequential([
    V.Flip(p=0.5),
])

# Create example 3d image (height, width, depth, nchannels)
input = np.ones((32, 32, 32, 1))

# Augment voxel
transformed = transform(voxel=input)
output = transformed['voxel']
```

## Supported shapes

Voxelmentations currently uses channel-last tensors and named targets.

Supported target shapes:

| Target | Supported shapes | Notes |
| --- | --- | --- |
| `voxel` | `(H, W, D)` or `(H, W, D, C)` | `C` is the number of channels |
| `mask` | `(H, W, D)` or `(H, W, D, C)` | for segmentation masks and other dense labels |
| `points` | `(N, 4)` | first 3 values are spatial coordinates, the last value is the homogeneous coordinate |

Conventions:

- Spatial axis order is `(H, W, D)`.
- Channel axis, when present, is the last axis.
- Data should be passed as named arguments, for example: `transform(voxel=voxel, mask=mask)`.
- Batch dimensions are not supported directly.
- Channel-first layouts such as `(C, H, W, D)` are not supported directly.

## Additional targets

Voxelmentations supports additional named inputs through `add_keys(...)`.
This is useful when you want to transform multiple volumes or masks in the same way.

For example, the mapping below means:

- `voxel2` is transformed the same way as `voxel`
- `mask2` is transformed the same way as `mask`

```python
import voxelmentations as V

transform = V.Sequential([
    V.Flip(p=1.0),
]).add_keys({
    'voxel2': 'voxel',
    'mask2': 'mask',
})

transformed = transform(
    voxel=voxel1,
    voxel2=voxel2,
    mask=mask1,
    mask2=mask2,
)
```

`add_keys(...)` works both for single augmentations and for composed pipelines.

## List of augmentations

The list of transforms:

- [PadIfNeeded]()
- [Flip]()
- [AxialFlip]()
- [AxialPlaneFlip]()
- [Rotate90]()
- [AxialPlaneRotate90]()
- [Transpose]()
- [AxialPlaneTranspose]()
- [Affine]()
- [Scale]()
- [Translate]()
- [AxialPlaneAffine]()
- [AxialPlaneRotate]()
- [AxialPlaneScale]()
- [AxialPlaneTranslate]()
- [GridDistort]()
- [ElasticDistort]()
- [GaussNoise]()
- [GaussBlur]()
- [IntensityShift]()
- [IntensityScale]()
- [Contrast]()
- [Gamma]()
- [PlaneDropout]()
- [HorizontalPlaneDropout]()
- [VerticalPlaneDropout]()
- [AxialPlaneDropout]()
- [PatchDropout]()
- [PatchShuffle]()
- [Downscale]()

## Citing

If you find this library useful for your research, please consider citing:

```
@misc{epifanov2024voxelmentations,
  Author = {Rostislav Epifanov},
  Title = {voxelmentations},
  Year = {2024},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/rostepifanov/voxelmentations}}
}
```
