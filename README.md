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

## List of augmentations

The list of transforms:

- [Flip]()
- [AxialFlip]()
- [AxialPlaneFlip]()
- [AxialPlaneRotate]()
- [AxialPlaneScale]()
- [AxialPlaneAffine]()
- [GridDistort]()
- [ElasticDistort]()
- [RandomGamma]()
- [IntensityShift]()
- [GaussBlur]()
- [GaussNoise]()
- [PlaneDropout]()
- [HorizontalPlaneDropout]()
- [VerticalPlaneDropout]()
- [AxialPlaneDropout]()
- [PatchDropout]()
- [PatchShuffle]()

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
