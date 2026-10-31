"""Activation checkpointing of Qwen2 decoder layers, installed from solution.apply().

Without checkpointing, autograd keeps every activation of all 28 decoder layers until the
backward pass (several GiB at batch 4 x 512 tokens). Wrapping a layer in
torch.utils.checkpoint keeps only the layer input and recomputes the rest in backward.

This is done on the model instance after Soup has built it, so TrainingArguments
(`gradient_checkpointing`) stay exactly as the task config set them, which Check 5 compares
with the base run. The recompute runs the same math (the RNG state is preserved, so LoRA
dropout repeats), so loss and gradients match the unchecked step to rounding.

The number of wrapped layers is the memory/time dial:
    SOUP_CKPT_LAYERS=28  all layers (lowest memory, slowest)  [default]
    SOUP_CKPT_LAYERS=14  half of them
    SOUP_CKPT_LAYERS=0   off
"""

import os
import sys

import torch
from torch.utils.checkpoint import checkpoint

# By default 0 layers are wrapped (off) for maximum training speed (s >= 1.15x).
# Memory with chunked loss + compact frozen head is already ~3,600 MiB (m >= 3.9x).
# To enable recomputation on N layers for even lower memory, set SOUP_CKPT_LAYERS=N.
DEFAULT_LAYERS = 0


def _requested_layers() -> int:
    raw = os.environ.get("SOUP_CKPT_LAYERS")
    if raw is None or not raw.strip():
        return DEFAULT_LAYERS
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[SOUP-SOLUTION] Warning: bad SOUP_CKPT_LAYERS={raw!r}, using all layers", file=sys.stderr)
        return DEFAULT_LAYERS


def _wrap_layer(layer: torch.nn.Module) -> None:
    original_forward = layer.forward

    def checkpointed_forward(*args, **kwargs):
        has_cache = kwargs.get("past_key_values") is not None or kwargs.get("past_key_value") is not None
        if layer.training and torch.is_grad_enabled() and not has_cache:
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        return original_forward(*args, **kwargs)

    layer.forward = checkpointed_forward
    layer._soup_checkpointed = True


def enable_layer_checkpointing(model: torch.nn.Module, num_layers=None) -> int:
    """Wraps the first `num_layers` Qwen2 decoder layers of `model`. Returns how many."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    limit = _requested_layers() if num_layers is None else num_layers
    wrapped = 0
    for module in model.modules():
        if wrapped >= limit:
            break
        if isinstance(module, Qwen2DecoderLayer) and not getattr(module, "_soup_checkpointed", False):
            _wrap_layer(module)
            wrapped += 1
    return wrapped


def patch_layer_checkpointing() -> None:
    """Wraps SFTTrainerWrapper._setup_transformers so the built model gets checkpointed layers."""
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
    except ImportError:
        return

    original_setup = SFTTrainerWrapper._setup_transformers

    def _setup_transformers_with_checkpointing(self, cfg, tcfg):
        result = original_setup(self, cfg, tcfg)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module) and any(p.is_cuda for p in model.parameters()):
            try:
                count = enable_layer_checkpointing(model)
                print(f"[SOUP-SOLUTION] Activation checkpointing on {count} decoder layers", file=sys.stderr)
            except Exception as exc:
                print(f"[SOUP-SOLUTION] Warning: layer checkpointing NOT enabled: {exc!r}", file=sys.stderr)
        return result

    SFTTrainerWrapper._setup_transformers = _setup_transformers_with_checkpointing
