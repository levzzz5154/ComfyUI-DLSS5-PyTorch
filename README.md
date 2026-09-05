# ComfyUI-DLSS5-PyTorch

Experimental ComfyUI nodes for running the reverse-engineered **DLSS 5 Neural Rendering** model from [iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS) directly through its PyTorch implementation.

This project is intentionally different from existing DLSS 5 ComfyUI integrations that call NVIDIA NGX through D3D12/VapourSynth. It does **not** call `nvngx_dlssnr.dll` during inference. Instead, it loads MLX-DLSS **fully-logical safetensors** and executes the recovered transformer graph in PyTorch.

> Experimental / research software. This is not affiliated with or endorsed by NVIDIA, Comfy Org, or the MLX-DLSS authors.

## Why this exists

- Direct PyTorch execution of the recovered model.
- CUDA/Linux should be possible because inference is not tied to D3D12/NGX.
- Useful baseline before trying custom FP8/INT8/CUTLASS kernels.
- Keeps the ComfyUI wrapper small by depending on upstream `mlxdlss` rather than copying the recovered model code.

## Current status

v0.1 supports **first-frame / still-image inference** and IMAGE batches.

MLX-DLSS also contains a separate recovered `TemporalSession` with display-history reprojection, motion input/optical-flow support, and the learned temporal blend. **This node does not expose that temporal API yet**; its batch sequence mode only advances the deterministic noise frame index. Therefore v0.1 should be treated as still-image/batch neural-rendering inference, not full temporal DLSS 5 parity.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/levzzz5154/ComfyUI-DLSS5-PyTorch.git
cd ComfyUI-DLSS5-PyTorch
pip install -r requirements.txt
```

Restart ComfyUI.

The dependency is pinned to a known MLX-DLSS commit for API stability.

## Weights

This node does not redistribute NVIDIA model weights or runtime binaries.

It expects the **fully-logical safetensors** format accepted by MLX-DLSS (`dlssnr-logical-v8` through the formats supported by the pinned upstream package, with `fully_logical=true`).

Put the file in:

```text
ComfyUI/models/dlss5/
```

Then choose it in **DLSS 5 PyTorch Model Loader**.

Use the MLX-DLSS weight tooling to produce the logical safetensors from a source you are legally permitted to use. See upstream documentation for the current extraction workflow.

## Nodes

### DLSS 5 PyTorch Model Loader

Loads and caches one logical model.

- **precision = fast**: upstream FP16 path on GPU while preserving the recovered E4M3 publication behavior.
- **precision = reference**: upstream float32/reference path; much slower and intended for correctness comparisons.
- **device = auto**: CUDA first, then MPS, then CPU according to MLX-DLSS.

### DLSS 5 PyTorch Neural Rendering

Inputs:

- ComfyUI `IMAGE` (single image or batch)
- profile: `standard`, `natural`, `cinematic`, `neutral`
- processing scale: 1.0–4.0
- intensity
- detail / colour strengths and detail radius
- deterministic noise frame index
- optional custom style/tone/structure controls
- optional RGB control image

The upstream control-image channel semantics are:

- R: final blend strength
- G: local tone strength
- B: local structure strength

When a control image is supplied, upstream currently requires `processing_scale = 1.0`.

### DLSS 5 PyTorch Clear Cache

Releases the cached model and asks PyTorch to empty CUDA cache.

## Batch behavior

`independent (same frame index)` uses the same deterministic noise index for every image in the batch.

`sequence (advance frame index)` increments the noise frame index across the batch. This is **not temporal accumulation**. Upstream MLX-DLSS has a separate `TemporalSession`; exposing that stateful path in ComfyUI is future work here.

## Performance notes

This is the reference PyTorch implementation, not NVIDIA's fused runtime. Expect it to be substantially slower than native DLSS 5. The point of this node is to establish a correct, inspectable ComfyUI baseline first.

The MLX-DLSS project has already added bounded/chunked PyTorch evaluation to reduce activation memory. Future work here can target native FP8 and then INT8 Tensor Core kernels without changing the ComfyUI-facing workflow.

## Existing DLSS 5 ComfyUI projects

Other projects already integrate DLSS 5 by driving native/runtime paths, including:

- `lisitskyaa/ComfyUI-DLSS5-NR` — in-process D3D12/NGX bridge.
- `Blueforcer/ComfyUI-DLSS5-Enhancer` — native worker / runtime integration, including video tooling.
- `HECer/ComfyUI-DLSS5` — VapourSynth/VapourKit-based runtime path.

This repository's niche is specifically **recovered PyTorch model inference**.

## Roadmap

- [x] Load fully-logical MLX-DLSS safetensors
- [x] Still-image inference
- [x] IMAGE batches
- [x] ComfyUI model-folder integration
- [x] Optional RGB control image
- [ ] Expose upstream `TemporalSession` (history + motion-vector/optical-flow path)
- [ ] CUDA-only path that avoids CPU/NumPy staging
- [ ] Native FP8 kernels
- [ ] Experimental INT8 W8A8 backend
- [ ] ComfyUI Registry metadata / packaged release

## Development check

The repository includes a small smoke test that mocks ComfyUI and the heavy MLX-DLSS network while exercising model discovery, loader caching, IMAGE batches, frame-index sequencing, custom controls, and control-mask validation:

```bash
python tests/smoke.py
```

It does not replace an end-to-end inference test with real logical weights on a ComfyUI installation.

## License

This ComfyUI wrapper is MIT licensed. MLX-DLSS is a separate project under its own license. NVIDIA software, model weights, trademarks, and proprietary binaries are not included in this repository and are governed by their respective terms.
