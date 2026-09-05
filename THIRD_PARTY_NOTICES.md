# Third-Party Notices

## MLX-DLSS

This project contains Python source derived from the MLX-DLSS project:

- Project: https://github.com/iamwavecut/MLX-DLSS
- Reference revision: `771ccb3b3bc84477c99452b63b16cb724319a338`
- Upstream notice: `MLX-DLSS — Copyright 2026 MLX-DLSS contributors`
- License: Apache License 2.0

The derived/adapted implementation is located primarily under `dlss5/`.
Changes made here include packaging the inference code directly inside a ComfyUI custom node, removing the external `mlxdlss` package dependency, simplifying runtime weight validation, and adapting the public entry point for ComfyUI.

A copy of the Apache License 2.0 is provided at `licenses/MLX-DLSS-APACHE-2.0.txt`.

NVIDIA DLSS, NVIDIA, and related marks are trademarks of NVIDIA Corporation. No NVIDIA proprietary binaries or model weights are redistributed by this repository.
