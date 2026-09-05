# ComfyUI-DLSS5-PyTorch

A **self-contained pure-PyTorch ComfyUI implementation** of the recovered DLSS 5 Neural Rendering network.

The reverse-engineering work this implementation is based on comes from [iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS). The required model/runtime code is included directly in this repository as ordinary Python source under `dlss5/`.

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

The inference implementation is visible source:

```text
ComfyUI-DLSS5-PyTorch/
├── __init__.py
├── nodes.py
└── dlss5/
    ├── __init__.py
    ├── model.py
    ├── pipeline.py
    ├── features.py
    ├── temporal.py
    └── composition.py
```

The only thing not bundled is the **model weight data**. You still need a compatible fully-logical DLSS 5 `.safetensors` file; proprietary model weights are not redistributed here.

## Architecture/runtime

`dlss5/model.py` contains the recovered 71-block transformer graph in PyTorch, including E4M3 publication emulation, the custom polynomial gate, cosine attention, shifted 8×8 windows, global bottleneck attention, hierarchical pooling/upsampling, and the four-channel output head.

`dlss5/features.py` builds the recovered 16-channel first-frame input. `dlss5/temporal.py` implements motion reprojection, five-tap history reconstruction, the optional recovered closest-depth guide, and learned temporal composition. `dlss5/composition.py` handles display composition/resampling. `dlss5/pipeline.py` ties weight loading and still-image inference together.

There is no hidden runtime behind the ComfyUI nodes.

## Installation

For a private clone, GitHub CLI is convenient:

```bash
cd ComfyUI/custom_nodes
gh repo clone levzzz5154/ComfyUI-DLSS5-PyTorch
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

The still/first-frame node. It intentionally contains **all recovered non-temporal controls** and none of the motion/history-specific controls.

Inputs include:

- current `IMAGE`
- profile: standard / natural / cinematic / neutral
- processing scale
- intensity
- detail strength / colour strength / detail radius
- deterministic frame index
- custom style index
- local tone strength
- local structure strength
- automatic-mask enable
- skin structure strength
- automatic-mask structure strength
- optional RGB `control_image`
- image batches

The RGB control image follows the recovered contract:

- R: final effect blend
- G: local tone multiplier
- B: local structure multiplier

An explicit control image takes precedence over automatic masking. `sequence (advance frame index)` only changes deterministic noise across a batch; it does **not** turn the still node into a temporal sequence.

### DLSS 5 PyTorch Video Neural Rendering

The temporal node. It takes an ordered ComfyUI `IMAGE` batch and carries the previous neural-rendered display history internally from one frame to the next.

It has the same art-direction/effect controls as the still node, plus:

- `motion_vectors` — RG are X/Y current-to-previous motion
- `motion_format` — pixel or normalized UV
- `motion_encoding` — raw signed RG, or 0.5-centered RG
- `motion_value_scale` — decoding scale for the supplied motion values
- `motion_scale_x/y` — recovered host motion scale/sign controls
- `jitter_delta_x/y` — previous jitter minus current jitter, in pixels
- `blend_scale` — recovered temporal blend cap (default `0.73974609375`)
- optional `depth_image`
- `depth_guide`
- `depth_inverted`
- optional scene-cut history reset threshold
- optional RGB control image

The first frame is evaluated as the normal first-frame path. Every later frame reprojects the retained previous output using the supplied current-to-previous motion and writes the reconstructed history into network feature channels 7–9.

`motion_vectors` may have:

- one frame, broadcast to every transition;
- the same batch size as the video (frame 0 is ignored); or
- `video_batch_size - 1`, one field per actual transition.

For `motion_encoding = signed RG`, values are used directly. For `0.5-centered RG`, `0.5` means zero and `[0,1]` maps to `[-1,1]` before `motion_value_scale` is applied.

For pixel motion, the recovered conversion is:

```text
normalizedX = (motionX * motionScaleX + jitterDeltaX) / width
normalizedY = (motionY * motionScaleY + jitterDeltaY) / height
```

Positive normalized offsets sample history to the right/down (`historyUV = currentUV + motion`).

#### Depth behavior

The surfaced DLL reads depth-related parameters, but the observed shipping path binds a null depth texture, so its normal behavior samples motion at the current pixel. Therefore:

- `observed (matches DLL)` ignores depth and matches that surfaced path;
- `closest-depth (experimental)` enables the separately recovered dormant branch that chooses the closest of the current pixel and four diagonals before sampling motion.

Depth is not a direct transformer feature channel.

Temporal mode currently requires `processing_scale = 1.0`.

### DLSS 5 PyTorch Clear Cache

Releases the cached model and empties the CUDA cache when available.

## What is *not* a runtime input

Albedo, normals, roughness, metallic/specular buffers and similar renderer G-buffer attributes are **not inputs to this recovered checkpoint's deployed transformer**. They are associated with NVIDIA's 3D-guided training/supervision story, not extra sockets that should be invented in this ComfyUI implementation.

The actual recovered transformer feature layout is:

```text
0-2   deterministic Gaussian noise
3     constant 1
4-6   current RGB
7-9   reprojected history RGB (current RGB on first frame)
10    style
11    local tone
12    local structure
13    skin structure
14    automatic-mask structure
15    zero
```

Motion and the optional depth guide are preprocessing inputs used to construct channels 7–9; they are not concatenated directly into the transformer.

## Current limitations

This is a correctness-first PyTorch implementation, not NVIDIA's fused production runtime. It is expected to be much slower than native DLSS 5 until expensive operations are replaced with optimized CUDA/FP8/INT8 kernels.

The video node currently expects motion fields to be supplied by the workflow; it does not generate optical flow itself.

## Why not use the native DLL?

There are already ComfyUI projects wrapping the native NVIDIA runtime. This project has a different goal: make the recovered model graph directly inspectable and modifiable in PyTorch so it can serve as a baseline for CUDA/FP8/INT8 work.

## Credits

The architecture recovery, tensor layouts, feature reconstruction, temporal reconstruction, and reference implementation this project is derived from were produced by the **MLX-DLSS contributors**:

https://github.com/iamwavecut/MLX-DLSS

Vendored/adapted portions retain the upstream Apache-2.0 licensing requirements. See `THIRD_PARTY_NOTICES.md` and `licenses/MLX-DLSS-APACHE-2.0.txt`.

This project is not affiliated with or endorsed by NVIDIA, Comfy Org, or the MLX-DLSS contributors.

## Development check

```bash
python tests/smoke.py
```

The smoke test checks both still and temporal node plumbing and explicitly verifies that the repository has no `mlxdlss` package dependency/import.

## License

The original ComfyUI wrapper code in this repository is MIT licensed. Code derived from MLX-DLSS remains subject to Apache License 2.0; see the included third-party notices and license copy.
