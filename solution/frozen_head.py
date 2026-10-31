"""Remove a redundant FP32 frozen head without rounding its stored values.

PEFT's k-bit preparation upcasts non-quantized weights to FP32. The chunked
loss then keeps a second FP16 head. Compacting an exactly FP16-representable
head avoids both the FP32 storage and that shadow copy. If input embeddings
share the Parameter, keep their output dtype unchanged for the decoder.
"""

import os
import sys

import torch
import torch.nn.functional as F


def compact_frozen_head(model: torch.nn.Module) -> int:
    """Return bytes saved in parameter storage; skip unsupported or lossy casts.

    The separate FP16 loss cache avoided by this change is not included.
    Call after k-bit preparation/PEFT and before the first labelled forward.
    """
    if os.environ.get("SOUP_COMPACT_HEAD", "1") == "0":
        return 0
    get_base = getattr(model, "get_base_model", None)
    causal_lm = get_base() if callable(get_base) else model
    get_head = getattr(causal_lm, "get_output_embeddings", None)
    if not callable(get_head):
        return 0
    head = get_head()
    # Quantized heads, adapter wrappers and custom projections need their own path.
    if type(head) is not torch.nn.Linear or head.bias is not None:
        return 0
    weight = head.weight
    if (weight.requires_grad or weight.dtype != torch.float32 or not weight.is_contiguous()
            or weight.numel() == 0 or weight.device.type == "meta"):
        return 0

    get_embed = getattr(causal_lm, "get_input_embeddings", None)
    embedding = get_embed() if callable(get_embed) else None
    tied = embedding is not None and getattr(embedding, "weight", None) is weight
    if tied and (type(embedding) is not torch.nn.Embedding or embedding.max_norm is not None):
        return 0
    # Distinct Parameter views into the same storage cannot safely be retied here.
    for other in model.parameters():
        if other is not weight and other.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr():
            return 0

    with torch.no_grad():
        flat = weight.view(-1)
        # Bound verification temporaries instead of making another full FP32 head.
        for start in range(0, flat.numel(), 1 << 20):
            part = flat[start:start + (1 << 20)]
            half = part.to(torch.float16)
            if not torch.isfinite(half).all().item() or not torch.equal(part, half.float()):
                print("[SOUP-SOLUTION] Frozen head kept in FP32: cast is not lossless", file=sys.stderr)
                return 0
        del flat, part, half
        saved = weight.numel() * 2
        weight.data = weight.data.to(torch.float16)

    original_head_forward = head.forward

    def head_forward(hidden):
        if torch.is_autocast_enabled(hidden.device.type) or hidden.dtype == head.weight.dtype:
            return original_head_forward(hidden)
        # Keep un-autocast FP32 calls valid, including generation/debugging.
        return F.linear(hidden, head.weight.to(hidden.dtype))

    head.forward = head_forward
    if tied:
        original_embedding_forward = embedding.forward

        def embedding_forward(*args, **kwargs):
            return original_embedding_forward(*args, **kwargs).to(torch.float32)

        embedding.forward = embedding_forward

    print(
        f"[SOUP-SOLUTION] Lossless frozen head FP16: saved {saved / 2**20:.1f} MiB "
        f"of parameter storage; tied_embedding={tied}; no FP16 shadow needed",
        file=sys.stderr,
    )
    return saved
