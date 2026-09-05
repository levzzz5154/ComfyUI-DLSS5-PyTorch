"""Audit regressions: python tests/regression.py (no checkpoint or GPU needed)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss5.pipeline import NeuralRenderingPipeline, load_weights, validate_weights
from dlss5.model import NeuralRenderingModel
from smoke import load_nodes, make_model


class WeightValidationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "dlss5/weight_spec.json"
        self.spec = json.loads(path.read_text())["tensors"]
        self.weights = {
            name: torch.empty(entry["shape"], device="meta")
            for name, entry in self.spec.items()
        }

    def test_complete_shapes_accepted(self):
        validate_weights(self.weights)

    def test_missing_non_sentinel_tensor_rejected(self):
        del self.weights["block1.layer0.weight1"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_weights(self.weights)

    def test_shape_and_dtype_rejected(self):
        name = next(iter(self.weights))
        self.weights[name] = torch.empty(1, device="meta")
        with self.assertRaisesRegex(ValueError, "expected shape"):
            validate_weights(self.weights)
        self.weights[name] = torch.empty(self.spec[name]["shape"], device="meta", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "floating-point"):
            validate_weights(self.weights)

    def test_file_loader_rejects_old_six_tensor_loophole(self):
        names = ["block0.layer0.input_adapter_weight", "block0.layer0.qkv_weight",
                 "block30.layer4.weight", "block39.layer0.conv_weight",
                 "block70.layer0.out_gain", "block70.layer0.out_conv_weight"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.safetensors"
            save_file({name: torch.zeros(1) for name in names}, str(path),
                      metadata={"format": "dlssnr-logical-v18", "fully_logical": "true"})
            with self.assertRaisesRegex(ValueError, "missing"):
                load_weights(path)

    def test_direct_pipeline_validates_before_constructing_model(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            NeuralRenderingPipeline({}, device="cpu")


class NodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.nodes = load_nodes(Path(self.temp.name))
        self.path = make_model(self.nodes)
        self.handle = self.nodes.DLSS5PyTorchModelLoader().load_model(
            "model.safetensors", "fast", "auto")[0]
        self.moves = []
        self.handle.pipeline.model.to = lambda device: self.moves.append(str(device))
        # Movement is recorded, so lifecycle tests can exercise a CUDA target
        # without requiring or allocating on a physical GPU.
        self.handle.pipeline.device = torch.device("cuda")

    def test_offloads_after_success_and_failure(self):
        for fail in (False, True):
            self.moves.clear()

            @self.nodes._with_model_on_device
            def render(instance, dlss5_model, image, processing_scale):
                if fail:
                    raise RuntimeError("simulated OOM")
                return image

            image = torch.zeros(1, 8, 9, 3)
            if fail:
                with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
                    render(None, self.handle, image, 1.0)
            else:
                self.assertIs(render(None, self.handle, image, 1.0), image)
            self.assertEqual(self.moves, ["cuda", "cpu"])
            self.assertIsNone(self.handle.pipeline.model.interrupt_check)

    def test_clear_offloads_handle_retained_outside_private_cache(self):
        self.nodes._PIPELINE_CACHE.clear()
        self.nodes.DLSS5PyTorchClearCache().clear_cache(True)
        self.assertEqual(self.moves, ["cpu"])
        self.assertTrue(math.isnan(self.nodes.DLSS5PyTorchClearCache.IS_CHANGED(True)))

    def test_loader_invalidates_replaced_file(self):
        loader = self.nodes.DLSS5PyTorchModelLoader
        old = loader.IS_CHANGED("model.safetensors", "fast", "auto")
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(old, loader.IS_CHANGED("model.safetensors", "fast", "auto"))

    def test_intensity_range_matches_blend_contract(self):
        for cls in (self.nodes.DLSS5PyTorchEnhance, self.nodes.DLSS5PyTorchVideoEnhance):
            self.assertEqual(cls.INPUT_TYPES()["required"]["intensity"][1]["max"], 1.0)

    def test_model_checks_cancellation_before_each_block_family(self):
        model = NeuralRenderingModel({})
        def interrupt():
            raise RuntimeError("cancelled")
        model.interrupt_check = interrupt
        for call in (lambda: model._window(None, 0, head_count=1),
                     lambda: model._split_window(None, 23), lambda: model._global(None, 31)):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                call()


if __name__ == "__main__":
    unittest.main()
