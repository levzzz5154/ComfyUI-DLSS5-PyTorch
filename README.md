# ComfyUI-DLSS5-PyTorch

A **self-contained pure-PyTorch ComfyUI implementation** of the recovered DLSS 5 Neural Rendering network.

The reverse-engineering work this implementation is based on comes from [iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS). The required PyTorch model/runtime code is included directly in this repository as ordinary Python source under `dlss5/`.

## What “self-contained” means here

This repository does **not** depend on the `mlxdlss` Python package and does not download MLX-DLSS at install or runtime.

It also does **not** use or bundle:

- `nvngx_dlssnr.dll`
- NVIDIA NGX
- D3D12 runtime bridges
- VapourSynth/VapourKit
- custom `.so` / `.dll` native extensions
- precompiled CUDA binaries
- opaque executable blobs

The inference implementation is visible Python/PyTorch source:

```text
ComfyUI-DLSS5-PyTorch/
├── __init__.py
├── nodes.py
└── dlss5/
    ├── __init__.py
    ├── model.py
    ├── pipeline.py
    ├── features.py
    └── composition.py
```

The only thing not bundled is the **model weight data**. You still need a compatible fully-logical DLSS 5 `.safetensors` file; proprietary model weights are not redistributed here.

## Architecture/runtime

`dlss5/model.py` contains the recovered 71-block transformer graph in PyTorch, including the E4M3 publication emulation, custom polynomial gate, cosine attention, shifted 8×8 windows, global bottleneck attention, hierarchical pooling/upsampling, and output head.

`dlss5/features.py` builds the recovered 16-channel first-frame input features. `dlss5/composition.py` performs output composition/resampling. `dlss5/pipeline.py` ties weight loading, preprocessing, PyTorch inference, and postprocessing together.

There is no hidden runtime behind the ComfyUI node.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/levzzz5154/ComfyUI-DLSS5-PyTorch.git
cd ComfyUI-DLSS5-PyTorch
pip install -r requirements.txt
```

Restart ComfyUI.

The requirements are only normal Python libraries used by the implementation:

```text
numpy
Pillow
safetensors
```

PyTorch itself is supplied by ComfyUI.

## Weights

Place a compatible **fully-logical** DLSS 5 safetensors file in:

```text
ComfyUI/models/dlss5/
```

The loader currently accepts recovered logical formats `dlssnr-logical-v8` through `dlssnr-logical-v18` with metadata `fully_logical=true`.

No weights are included in this repository.

## Nodes

### DLSS 5 PyTorch Model Loader

Loads the logical safetensors directly into the local PyTorch implementation.

- `fast`: float16 model execution on GPU while preserving recovered E4M3 publication points.
- `reference`: float32/reference execution.
- `auto`: CUDA first, then MPS, then CPU.

### DLSS 5 PyTorch Neural Rendering

Takes a normal ComfyUI `IMAGE` and exposes:

- profile: standard / natural / cinematic / neutral
- processing scale
- intensity
- detail strength
- colour strength
- detail radius
- deterministic frame index
- custom style/tone/structure controls
- optional RGB control image
- image batches

`sequence (advance frame index)` only advances deterministic noise across a batch; it is not full temporal accumulation yet.

### DLSS 5 PyTorch Clear Cache

Releases the cached model and empties the CUDA cache when available.

## Current limitations

This is a correctness-first PyTorch implementation, not NVIDIA's fused production runtime. It is expected to be much slower than native DLSS 5 until the expensive operations are replaced with optimized kernels.

v0.2 currently exposes the first-frame/still-image path. Temporal history + motion-vector inference is planned separately.

## Why not use the native DLL?

There are already ComfyUI projects wrapping the native NVIDIA runtime. This project has a different goal: make the recovered model graph directly inspectable and modifiable in PyTorch so it can later serve as the baseline for CUDA/FP8/INT8 work.

## Credits

The architecture recovery, tensor layouts, feature reconstruction, and reference implementation this project is derived from were produced by the **MLX-DLSS contributors**:

https://github.com/iamwavecut/MLX-DLSS

Vendored/adapted portions retain the upstream Apache-2.0 licensing requirements. See `THIRD_PARTY_NOTICES.md` and `licenses/MLX-DLSS-APACHE-2.0.txt`.

This project is not affiliated with or endorsed by NVIDIA, Comfy Org, or the MLX-DLSS contributors.

## Development check

```bash
python tests/smoke.py
```

The smoke test checks the ComfyUI-facing plumbing and explicitly verifies that the repository has no `mlxdlss` package dependency/import.

## License

The original ComfyUI wrapper code in this repository is MIT licensed. Code derived from MLX-DLSS remains subject to Apache License 2.0; see the included third-party notices and license copy.
