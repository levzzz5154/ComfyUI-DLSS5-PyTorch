"""Dependency-light smoke tests for the ComfyUI wrapper.

Run from the repository root with: python tests/smoke.py
The MLX-DLSS network is mocked; this validates ComfyUI-facing plumbing without
requiring proprietary weights or a full ComfyUI checkout.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]


class FakeFolderPaths(types.ModuleType):
    def __init__(self, model_root: Path):
        super().__init__("folder_paths")
        self.models_dir = str(model_root)
        self.folder_names_and_paths = {}

    def add_model_folder_path(self, name, path, is_default=False):
        del is_default
        self.folder_names_and_paths.setdefault(name, ([path], set()))

    def get_filename_list(self, name):
        paths, _ = self.folder_names_and_paths[name]
        out = []
        for base in paths:
            base = Path(base)
            if not base.exists():
                continue
            out.extend(
                str(p.relative_to(base)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file()
            )
        return sorted(out)

    def get_full_path(self, name, filename):
        paths, _ = self.folder_names_and_paths[name]
        for base in paths:
            candidate = Path(base) / filename
            if candidate.is_file():
                return str(candidate)
        return None


class FakePipeline:
    loads = []

    def __init__(self, model_path, device, precision):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.precision = precision
        self.calls = []

    @classmethod
    def from_safetensors(cls, model_path, *, device="auto", precision="reference"):
        cls.loads.append((str(model_path), device, precision))
        return cls(str(model_path), device, precision)

    def enhance(self, image, **kwargs):
        self.calls.append(kwargs)
        result = np.clip(np.asarray(image, dtype=np.float32) + 0.1, 0.0, 1.0)
        return types.SimpleNamespace(image=result)


def load_nodes(model_root: Path):
    fp = FakeFolderPaths(model_root)
    sys.modules["folder_paths"] = fp

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.value = 0

        def update(self, amount):
            self.value += amount

    comfy_utils.ProgressBar = ProgressBar
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils

    mlxdlss = types.ModuleType("mlxdlss")
    pipeline = types.ModuleType("mlxdlss.pipeline")
    pipeline.NeuralRenderingPipeline = FakePipeline
    sys.modules["mlxdlss"] = mlxdlss
    sys.modules["mlxdlss.pipeline"] = pipeline

    # Load nodes.py directly for focused tests.
    spec = importlib.util.spec_from_file_location("dlss5_nodes_smoke", REPO / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Also verify the package-style __init__.py load ComfyUI uses for custom nodes.
    package_name = "dlss5_custom_node_smoke"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        REPO / "__init__.py",
        submodule_search_locations=[str(REPO)],
    )
    package = importlib.util.module_from_spec(package_spec)
    assert package_spec.loader is not None
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    assert "DLSS5PyTorchModelLoader" in package.NODE_CLASS_MAPPINGS
    return module


def make_model(nodes, name="model.safetensors"):
    path = Path(nodes._MODEL_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        nodes = load_nodes(Path(td) / "models")
        FakePipeline.loads.clear()

        # Registration and extension filtering.
        make_model(nodes, "a.safetensors")
        (Path(nodes._MODEL_DIR) / "ignore.txt").write_text("x")
        assert nodes._model_names() == ["a.safetensors"]

        # Cache behavior.
        loader = nodes.DLSS5PyTorchModelLoader()
        first = loader.load_model("a.safetensors", "fast", "auto")[0]
        second = loader.load_model("a.safetensors", "fast", "auto")[0]
        assert first is second
        assert len(FakePipeline.loads) == 1

        # Batch conversion and frame-index sequencing.
        image = torch.zeros((2, 8, 9, 3), dtype=torch.float32)
        out = nodes.DLSS5PyTorchEnhance().enhance(
            first, image, "standard", 1.0, 1.0, 1.0, 1.0, 4.0, 7,
            "sequence (advance frame index)", False, 0, 1.0, 1.0,
        )[0]
        assert out.shape == image.shape
        assert torch.allclose(out, torch.full_like(out, 0.1))
        assert [c["frame_index"] for c in first.pipeline.calls[-2:]] == [7, 8]

        # Custom control forwarding.
        nodes.DLSS5PyTorchEnhance().enhance(
            first, image[:1], "natural", 1.0, 0.75, 0.5, 0.6, 3.0, 0,
            "independent (same frame index)", True, 64, 1.25, 0.8,
        )
        call = first.pipeline.calls[-1]
        assert abs(call["normalized_style"] - 0.5) < 1e-9
        assert abs(call["local_tone_strength"] - 1.25) < 1e-9
        assert abs(call["local_structure_strength"] - 0.8) < 1e-9

        # Control-image broadcasting.
        control = torch.ones((1, 8, 9, 3))
        nodes.DLSS5PyTorchEnhance().enhance(
            first, image, "standard", 1.0, 1.0, 1.0, 1.0, 4.0, 0,
            "independent (same frame index)", False, 0, 1.0, 1.0, control,
        )
        assert all(c["control_mask"].shape == (8, 9, 3) for c in first.pipeline.calls[-2:])

        # Geometry validation should fail before upstream invocation.
        bad_control = torch.zeros((1, 7, 9, 3))
        try:
            nodes.DLSS5PyTorchEnhance().enhance(
                first, image[:1], "standard", 1.0, 1.0, 1.0, 1.0, 4.0, 0,
                "independent (same frame index)", False, 0, 1.0, 1.0, bad_control,
            )
        except ValueError as exc:
            assert "height/width" in str(exc)
        else:
            raise AssertionError("mismatched control geometry was accepted")

        # Control masks must stay at processing scale 1.0 in current upstream.
        try:
            nodes.DLSS5PyTorchEnhance().enhance(
                first, image[:1], "standard", 2.0, 1.0, 1.0, 1.0, 4.0, 0,
                "independent (same frame index)", False, 0, 1.0, 1.0, control,
            )
        except ValueError as exc:
            assert "processing_scale=1.0" in str(exc)
        else:
            raise AssertionError("control mask with processing_scale != 1 was accepted")

    print("smoke tests: PASS")


if __name__ == "__main__":
    main()
