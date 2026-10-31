"""CPU tests, isolated from solution's CUDA/Triton package imports.

Run: python -m unittest discover -s tools -p test_frozen_head.py -v
"""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location(
        "frozen_head", Path(__file__).resolve().parents[1] / "solution" / "frozen_head.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compact_frozen_head = module.compact_frozen_head

    class ToyLM(torch.nn.Module):
        def __init__(self, tied=True):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 16)
            self.head = torch.nn.Linear(16, 32, bias=False)
            if tied:
                self.head.weight = self.embedding.weight
            self.requires_grad_(False)
            # Simulate an FP16 checkpoint upcast by k-bit preparation.
            with torch.no_grad():
                for param in self.parameters():
                    param.copy_(param.half().float())

        def get_output_embeddings(self):
            return self.head

        def get_input_embeddings(self):
            return self.embedding


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FrozenHeadTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SOUP_COMPACT_HEAD": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(42)

    def test_tied_embeddings_keep_values_dtype_and_parameter_identity(self):
        model = ToyLM()
        tokens = torch.tensor([[0, 3, 31]])
        expected = model.embedding(tokens)
        parameter = model.head.weight
        saved = compact_frozen_head(model)
        self.assertEqual(saved, parameter.numel() * 2)
        self.assertIs(model.head.weight, parameter)
        self.assertIs(model.embedding.weight, parameter)
        self.assertEqual(parameter.dtype, torch.float16)
        actual = model.embedding(tokens)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(compact_frozen_head(model), 0)

    def test_fp32_head_output_and_input_gradient_are_unchanged(self):
        model = ToyLM()
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        expected = model.head(hidden)
        expected_grad = torch.autograd.grad(expected.square().sum(), hidden)[0]
        compact_frozen_head(model)
        actual = model.head(hidden)
        actual_grad = torch.autograd.grad(actual.square().sum(), hidden)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_autocast_head_output_is_unchanged(self):
        model = ToyLM()
        hidden = torch.randn(2, 5, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model.head(hidden)
        compact_frozen_head(model)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.head(hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_untied_embeddings_are_untouched(self):
        model = ToyLM(tied=False)
        compact_frozen_head(model)
        self.assertEqual(model.head.weight.dtype, torch.float16)
        self.assertEqual(model.embedding.weight.dtype, torch.float32)

    def test_trainable_head_is_untouched(self):
        model = ToyLM()
        model.head.weight.requires_grad_(True)
        self.assertEqual(compact_frozen_head(model), 0)
        self.assertEqual(model.head.weight.dtype, torch.float32)

    def test_lossy_or_nonfinite_weights_are_untouched(self):
        for value in (0.1, float("inf"), float("nan"), 100000.0):
            with self.subTest(value=value):
                model = ToyLM()
                model.head.weight[0, 0] = value
                self.assertEqual(compact_frozen_head(model), 0)
                self.assertEqual(model.head.weight.dtype, torch.float32)

    def test_distinct_storage_alias_is_untouched(self):
        model = ToyLM(tied=False)
        model.embedding.weight = torch.nn.Parameter(model.head.weight.detach(), requires_grad=False)
        self.assertEqual(compact_frozen_head(model), 0)

    def test_environment_disables_compaction(self):
        model = ToyLM()
        with patch.dict(os.environ, {"SOUP_COMPACT_HEAD": "0"}):
            self.assertEqual(compact_frozen_head(model), 0)


if __name__ == "__main__":
    unittest.main()
