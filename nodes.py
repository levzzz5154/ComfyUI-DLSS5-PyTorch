from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import folder_paths


CATEGORY = "DLSS 5/PyTorch (experimental)"
MODEL_FOLDER_NAME = "dlss5"
MODEL_EXTENSIONS = {".safetensors"}
NO_MODEL_SENTINEL = "[no logical DLSS 5 .safetensors found]"

_MODEL_DIR = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Modern ComfyUI API. The fallback keeps the node usable on older builds.
try:
    folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(_MODEL_DIR))
except Exception:
    if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = ([str(_MODEL_DIR)], MODEL_EXTENSIONS)


@dataclass
class DLSS5ModelHandle:
    pipeline: Any
    model_path: str
    precision: str
    device: str


_PIPELINE_CACHE: dict[tuple[str, int, int, str, str], DLSS5ModelHandle] = {}


def _import_pipeline_class():
    """Import the runtime vendored inside this custom-node repository."""
    try:
        from .dlss5.pipeline import NeuralRenderingPipeline
    except (ImportError, ValueError):
        # Allows focused direct loading of nodes.py by tests/tools while the
        # normal ComfyUI package import uses the relative path above.
        from dlss5.pipeline import NeuralRenderingPipeline
    return NeuralRenderingPipeline


def _model_names() -> list[str]:
    try:
        names = folder_paths.get_filename_list(MODEL_FOLDER_NAME)
    except Exception:
        names = []

    names = [name for name in names if Path(name).suffix.lower() in MODEL_EXTENSIONS]
    return sorted(names) or [NO_MODEL_SENTINEL]


def _resolve_model_path(model_name: str) -> str:
    if model_name == NO_MODEL_SENTINEL:
        raise FileNotFoundError(
            f"No DLSS 5 logical safetensors were found. Put a fully-logical "
            f".safetensors file in: {_MODEL_DIR}"
        )

    full_path = None
    try:
        full_path = folder_paths.get_full_path(MODEL_FOLDER_NAME, model_name)
    except Exception:
        pass

    if not full_path:
        candidate = (_MODEL_DIR / model_name).resolve()
        if candidate.is_file():
            full_path = str(candidate)

    if not full_path or not Path(full_path).is_file():
        raise FileNotFoundError(f"DLSS 5 model not found: {model_name}")
    return str(Path(full_path).resolve())


def _clear_pipeline_cache() -> None:
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pipeline(model_path: str, precision: str, device: str) -> DLSS5ModelHandle:
    stat = os.stat(model_path)
    key = (model_path, stat.st_size, stat.st_mtime_ns, precision, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    # Keep only one large neural-rendering model resident by default.
    _clear_pipeline_cache()
    NeuralRenderingPipeline = _import_pipeline_class()
    pipeline = NeuralRenderingPipeline.from_safetensors(
        model_path,
        device=device,
        precision=precision,
    )
    handle = DLSS5ModelHandle(
        pipeline=pipeline,
        model_path=model_path,
        precision=precision,
        device=str(pipeline.device),
    )
    _PIPELINE_CACHE[key] = handle
    return handle


def _image_batch_to_numpy(image: torch.Tensor) -> np.ndarray:
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE input must be a torch.Tensor")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Expected ComfyUI IMAGE as [B,H,W,C>=3], got {tuple(image.shape)}")
    return image[..., :3].detach().to(device="cpu", dtype=torch.float32).numpy()


def _matching_control_frame(control: np.ndarray | None, index: int, batch: int) -> np.ndarray | None:
    if control is None:
        return None
    control_batch = control.shape[0]
    if control_batch == 1:
        return control[0]
    if control_batch != batch:
        raise ValueError(
            f"control_image batch must be 1 or match the input batch ({batch}); got {control_batch}"
        )
    return control[index]


class DLSS5PyTorchModelLoader:
    """Load fully-logical DLSS 5 safetensors into the recovered PyTorch model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_names(),),
                "precision": (["fast", "reference"], {"default": "fast"}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DLSS5_MODEL",)
    RETURN_NAMES = ("dlss5_model",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    def load_model(self, model_name: str, precision: str, device: str):
        model_path = _resolve_model_path(model_name)
        if precision not in {"fast", "reference"}:
            raise ValueError("precision must be 'fast' or 'reference'")
        if device not in {"auto", "cuda", "cpu", "mps"}:
            raise ValueError("unsupported device")
        return (_load_pipeline(model_path, precision, device),)


class DLSS5PyTorchEnhance:
    """Run recovered DLSS 5 Neural Rendering on ComfyUI IMAGE tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dlss5_model": ("DLSS5_MODEL",),
                "image": ("IMAGE",),
                "profile": (["standard", "natural", "cinematic", "neutral"], {"default": "standard"}),
                "processing_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05},
                ),
                "intensity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "detail_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "colour_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "detail_radius": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 32.0, "step": 0.1},
                ),
                "frame_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "step": 1},
                ),
                "batch_noise_mode": (
                    ["independent (same frame index)", "sequence (advance frame index)"],
                    {"default": "independent (same frame index)"},
                ),
                "use_custom_controls": ("BOOLEAN", {"default": False}),
                "style_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1},
                ),
                "local_tone_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "local_structure_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
            },
            "optional": {
                "control_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    def enhance(
        self,
        dlss5_model: DLSS5ModelHandle,
        image: torch.Tensor,
        profile: str,
        processing_scale: float,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        frame_index: int,
        batch_noise_mode: str,
        use_custom_controls: bool,
        style_index: int,
        local_tone_strength: float,
        local_structure_strength: float,
        control_image: torch.Tensor | None = None,
    ):
        if not isinstance(dlss5_model, DLSS5ModelHandle):
            raise TypeError("dlss5_model must come from the DLSS 5 PyTorch Model Loader node")
        if batch_noise_mode not in {
            "independent (same frame index)",
            "sequence (advance frame index)",
        }:
            raise ValueError("unsupported batch_noise_mode")

        source = _image_batch_to_numpy(image)
        control = _image_batch_to_numpy(control_image) if control_image is not None else None
        batch = source.shape[0]

        if batch == 0:
            raise ValueError("IMAGE batch must contain at least one frame")
        if control is not None and control.shape[1:3] != source.shape[1:3]:
            raise ValueError(
                "control_image height/width must match the input image; "
                f"got {tuple(control.shape[1:3])} vs {tuple(source.shape[1:3])}"
            )
        if control is not None and processing_scale != 1.0:
            raise ValueError("DLSS 5 control masks require processing_scale=1.0")

        progress = None
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(batch)
        except Exception:
            pass

        outputs: list[np.ndarray] = []
        for index in range(batch):
            current_frame_index = frame_index
            if batch_noise_mode == "sequence (advance frame index)":
                current_frame_index += index

            kwargs: dict[str, Any] = {
                "profile": profile,
                "processing_scale": processing_scale,
                "detail_strength": detail_strength,
                "colour_strength": colour_strength,
                "detail_radius": detail_radius,
                "intensity": intensity,
                "frame_index": current_frame_index,
                "control_mask": _matching_control_frame(control, index, batch),
            }
            if use_custom_controls:
                # Recovered feature channel 10 is style_index / 128.
                kwargs.update(
                    normalized_style=float(style_index) / 128.0,
                    local_tone_strength=float(local_tone_strength),
                    local_structure_strength=float(local_structure_strength),
                )

            result = dlss5_model.pipeline.enhance(source[index], **kwargs)
            outputs.append(np.asarray(result.image, dtype=np.float32))
            if progress is not None:
                progress.update(1)

        stacked = np.stack(outputs, axis=0)
        out = torch.from_numpy(stacked).clamp_(0.0, 1.0)
        return (out.to(device=image.device, dtype=image.dtype),)


class DLSS5PyTorchClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clear": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ()
    FUNCTION = "clear_cache"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def clear_cache(self, clear: bool):
        if clear:
            _clear_pipeline_cache()
        return ()


NODE_CLASS_MAPPINGS = {
    "DLSS5PyTorchModelLoader": DLSS5PyTorchModelLoader,
    "DLSS5PyTorchEnhance": DLSS5PyTorchEnhance,
    "DLSS5PyTorchClearCache": DLSS5PyTorchClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSS5PyTorchModelLoader": "DLSS 5 PyTorch Model Loader",
    "DLSS5PyTorchEnhance": "DLSS 5 PyTorch Neural Rendering",
    "DLSS5PyTorchClearCache": "DLSS 5 PyTorch Clear Cache",
}
