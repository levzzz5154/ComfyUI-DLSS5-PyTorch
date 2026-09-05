"""Self-contained PyTorch inference pipeline for the recovered DLSS 5 graph."""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import model as reference
from .composition import compose_detail, compose_head, resample
from .features import PROFILES, AutomaticMask, NetworkGeometry, make_features

PRECISIONS = ("reference", "fast")
WEIGHT_FORMATS = tuple(f"dlssnr-logical-v{v}" for v in range(8, 19))


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_weights(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load fully-logical DLSS 5 safetensors without any external runtime package."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        fmt = metadata.get("format")
        if fmt not in WEIGHT_FORMATS:
            raise ValueError(
                f"unsupported weight format {fmt!r}; expected {WEIGHT_FORMATS[0]} through {WEIGHT_FORMATS[-1]}"
            )
        if metadata.get("fully_logical") != "true":
            raise ValueError("weights must declare fully_logical=true")
        weights = {name: source.get_tensor(name) for name in source.keys()}

    required = {
        "block0.layer0.input_adapter_weight",
        "block0.layer0.qkv_weight",
        "block30.layer4.weight",
        "block39.layer0.conv_weight",
        "block70.layer0.out_gain",
        "block70.layer0.out_conv_weight",
    }
    missing = sorted(required.difference(weights))
    if missing:
        raise ValueError(f"weights are missing required tensors: {missing}")
    return weights


@dataclass
class EnhanceResult:
    image: np.ndarray
    network_extent: tuple[int, int]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PreparedFrame:
    source: np.ndarray
    processing: np.ndarray
    features: np.ndarray
    geometry: NetworkGeometry
    control_mask: np.ndarray | None
    preprocess_seconds: float


class NeuralRenderingPipeline:
    """Runs the recovered 71-block PyTorch model on one RGB frame."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        *,
        device: str | torch.device = "auto",
        precision: str = "reference",
    ):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.device = resolve_device(device)
        self.precision = precision
        self.model = reference.NeuralRenderingModel(weights).eval()
        self.dtype = torch.float32

        if precision == "fast" and self.device.type != "cpu":
            self.dtype = torch.float16
            self.model = self.model.to(torch.float16)
        self.model = self.model.to(self.device)

    @classmethod
    def from_safetensors(
        cls,
        path: str | pathlib.Path,
        *,
        device: str | torch.device = "auto",
        precision: str = "reference",
    ) -> "NeuralRenderingPipeline":
        return cls(load_weights(path), device=device, precision=precision)

    @torch.no_grad()
    def run_features(self, features: np.ndarray) -> np.ndarray:
        return self.run_features_batch(np.asarray(features, dtype=np.float32)[None])[0]

    @torch.no_grad()
    def run_features_batch(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 4 or features.shape[-1] != 16:
            raise ValueError("features must be [B,H,W,16]")
        if features.shape[1] % 64 or features.shape[2] % 64:
            raise ValueError("feature extent must be a multiple of 64")
        tensor = torch.from_numpy(np.ascontiguousarray(features)).to(self.device, self.dtype)
        head = self.model(tensor)
        return head.to(torch.float32).cpu().numpy()

    def _controls(
        self,
        profile: str,
        normalized_style: float | None,
        local_tone_strength: float | None,
        local_structure_strength: float | None,
    ) -> dict[str, float]:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {tuple(PROFILES)}")
        controls = dict(PROFILES[profile])
        if normalized_style is not None:
            controls["normalized_style"] = float(normalized_style)
        if local_tone_strength is not None:
            controls["local_tone_strength"] = float(local_tone_strength)
        if local_structure_strength is not None:
            controls["local_structure_strength"] = float(local_structure_strength)
        return controls

    def prepare(
        self,
        image: np.ndarray,
        *,
        profile: str = "standard",
        processing_scale: float = 1.0,
        frame_index: int = 0,
        control_mask: np.ndarray | None = None,
        automatic_mask: AutomaticMask | None = None,
        normalized_style: float | None = None,
        local_tone_strength: float | None = None,
        local_structure_strength: float | None = None,
    ) -> PreparedFrame:
        if not 1.0 <= processing_scale <= 4.0:
            raise ValueError("processing_scale must be within [1, 4]")
        if control_mask is not None and processing_scale != 1.0:
            raise ValueError("a control mask requires processing_scale=1")

        source = np.asarray(image, dtype=np.float32)
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError("image must be [H,W,3]")

        started = time.perf_counter()
        processing = source
        if processing_scale != 1.0:
            processing = resample(
                source,
                int(round(source.shape[1] * processing_scale)),
                int(round(source.shape[0] * processing_scale)),
            )

        geometry = NetworkGeometry.vendor_aligned(processing.shape[1], processing.shape[0])
        controls = self._controls(
            profile,
            normalized_style,
            local_tone_strength,
            local_structure_strength,
        )
        features = make_features(
            processing,
            frame_index=frame_index,
            geometry=geometry,
            automatic_mask=automatic_mask,
            control_mask=control_mask,
            **controls,
        )
        return PreparedFrame(
            source=source,
            processing=processing,
            features=features,
            geometry=geometry,
            control_mask=control_mask,
            preprocess_seconds=time.perf_counter() - started,
        )

    def finish(
        self,
        prepared: PreparedFrame,
        head: np.ndarray,
        *,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
        intensity: float = 1.0,
        network_seconds: float = 0.0,
    ) -> EnhanceResult:
        started = time.perf_counter()
        composed = compose_head(
            prepared.geometry.crop(head),
            prepared.processing,
            control_mask=prepared.control_mask,
            intensity=intensity,
        )
        if composed.shape[:2] != prepared.source.shape[:2]:
            composed = resample(composed, prepared.source.shape[1], prepared.source.shape[0])
        output = compose_detail(
            prepared.source,
            composed,
            detail_strength=detail_strength,
            colour_strength=colour_strength,
            radius=detail_radius,
        )
        return EnhanceResult(
            image=output,
            network_extent=(prepared.geometry.network_height, prepared.geometry.network_width),
            timings={
                "preprocess": prepared.preprocess_seconds,
                "network": network_seconds,
                "postprocess": time.perf_counter() - started,
            },
        )

    def enhance(
        self,
        image: np.ndarray,
        *,
        profile: str = "standard",
        processing_scale: float = 1.0,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
        intensity: float = 1.0,
        frame_index: int = 0,
        control_mask: np.ndarray | None = None,
        automatic_mask: AutomaticMask | None = None,
        normalized_style: float | None = None,
        local_tone_strength: float | None = None,
        local_structure_strength: float | None = None,
    ) -> EnhanceResult:
        prepared = self.prepare(
            image,
            profile=profile,
            processing_scale=processing_scale,
            frame_index=frame_index,
            control_mask=control_mask,
            automatic_mask=automatic_mask,
            normalized_style=normalized_style,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
        )
        started = time.perf_counter()
        head = self.run_features(prepared.features)
        return self.finish(
            prepared,
            head,
            detail_strength=detail_strength,
            colour_strength=colour_strength,
            detail_radius=detail_radius,
            intensity=intensity,
            network_seconds=time.perf_counter() - started,
        )


class NeuralRenderingSession:
    """Simple frame-indexing wrapper; this is not temporal accumulation."""

    def __init__(self, pipeline: NeuralRenderingPipeline, **enhance_options: Any):
        self.pipeline = pipeline
        self.options = enhance_options
        self.frame_index = 0

    def reset(self) -> None:
        self.frame_index = 0

    def process(self, image: np.ndarray) -> EnhanceResult:
        result = self.pipeline.enhance(image, frame_index=self.frame_index, **self.options)
        self.frame_index += 1
        return result
