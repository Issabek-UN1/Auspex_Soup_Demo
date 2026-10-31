"""High-Performance Fused Token-Chunked Cross-Entropy Loss + Online Metrics for Qwen2.5.
Computes Online Softmax, Cross-Entropy Loss, Shannon Entropy, Top-1 Accuracy,
and in-place backward gradients in Triton.

Key performance optimizations for Tesla T4 (Turing SM 7.5):
1. Cached FP16 Head Weights:
   Caches lm_head.weight in fp16 once. Eliminates casting a 933 MB weight tensor
   on every chunk.
2. Optimal Chunk Size (CHUNK_SIZE = 128):
   Processes 128 tokens per chunk (512 tokens for batch size 4). Exactly 16 chunks
   per step instead of 64 chunks, cutting Python dispatch overhead by 4x.
3. In-Place Logits Gradients & Immediate Memory Reclamation:
   Overwrites chunk logits buffer directly in Pass 2 with zero extra gradient allocations,
   multiplies by W_head immediately, and frees the logits chunk before the next iteration.
4. Turing SM 7.5 Kernel Tuning:
   num_stages=2, num_warps=4, BLOCK_V=4096.

Clean-room author implementation for Soup Kernel Challenge (Decentrathon 6.0).
"""

import os
import sys
from typing import Any, Optional, Tuple
import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100

_CACHED_FP16_WEIGHT: Optional[torch.Tensor] = None
_CACHED_FP16_WEIGHT_ID: Optional[int] = None


@triton.jit
def _chunked_cross_entropy_fused_kernel(
    logits_ptr,          # [N_rows, V] fp16
    targets_ptr,         # [N_rows] int64
    losses_ptr,          # [N_rows] fp32 out
    entropies_ptr,       # [N_rows] fp32 out
    correct_ptr,         # [N_rows] int32 out
    stride_logits_row,   # Stride between rows in logits
    V,                   # Vocabulary size (e.g. 151936)
    denominator,         # Normalizing denominator (float)
    ignore_index: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused Cross-Entropy, Online Softmax, Entropy, Accuracy, and In-place Gradient."""
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_row

    # Online Softmax running statistics in FP32 registers
    m = -10000.0
    d = 0.0
    w = 0.0
    max_val = -10000.0
    max_idx = -1
    z_target = 0.0

    n_tiles = tl.cdiv(V, BLOCK_V)

    # -----------------------------------------------------------------
    # PASS 1: Online Softmax, Argmax, Cross-Entropy Loss & Entropy
    # -----------------------------------------------------------------
    for tile_k in range(0, n_tiles):
        col_offsets = tile_k * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = col_offsets < V

        # Load tile in fp16, cast to fp32 accumulator
        z = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-10000.0).to(tl.float32)

        tile_max = tl.max(z, axis=0)
        tile_argmax_local = tl.argmax(z, axis=0)
        tile_argmax = tile_k * BLOCK_V + tile_argmax_local

        new_m = tl.maximum(m, tile_max)
        alpha = tl.exp(m - new_m)

        # Zero out-of-bounds elements explicitly to prevent (-inf * 0.0 = NaN)
        exp_z = tl.where(mask, tl.exp(z - new_m), 0.0)
        z_safe = tl.where(mask, z, 0.0)

        d = d * alpha + tl.sum(exp_z, axis=0)
        w = w * alpha + tl.sum(z_safe * exp_z, axis=0)

        # Update global argmax
        is_new_max = tile_max > max_val
        max_idx = tl.where(is_new_max, tile_argmax, max_idx)
        max_val = tl.maximum(max_val, tile_max)
        m = new_m

        # Capture target logit if target is within current tile
        is_target = (col_offsets == target)
        z_target += tl.sum(tl.where(is_target & mask, z, 0.0), axis=0)

    # Compute row-level metrics
    is_ignored = (target == ignore_index)
    safe_d = tl.maximum(d, 1e-12)
    log_d = tl.log(safe_d)

    raw_loss = log_d + m - z_target
    # Shannon entropy: H(p) = -sum(p * log p) = log_d + m - (w / d)
    raw_entropy = tl.maximum(0.0, log_d + m - (w / safe_d))
    raw_correct = tl.where(max_idx == target, 1, 0)

    loss = tl.where(is_ignored, 0.0, raw_loss)
    correct = tl.where(is_ignored, 0, raw_correct)

    tl.store(losses_ptr + row_idx, loss)
    tl.store(entropies_ptr + row_idx, raw_entropy)
    tl.store(correct_ptr + row_idx, correct)

    # -----------------------------------------------------------------
    # PASS 2: In-place Gradient Calculation directly inside logits_ptr
    # -----------------------------------------------------------------
    for tile_k in range(0, n_tiles):
        col_offsets = tile_k * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = col_offsets < V

        z = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-10000.0).to(tl.float32)
        prob = tl.where(mask, tl.exp(z - m) / safe_d, 0.0)
        is_target = tl.where(col_offsets == target, 1.0, 0.0)

        # Softmax cross-entropy gradient: (p - y) / denominator
        grad = tl.where(is_ignored, 0.0, (prob - is_target) / denominator)

        # Store in-place back to logits buffer
        tl.store(logits_row_ptr + col_offsets, grad.to(logits_ptr.dtype.element_ty), mask=mask)


def _extract_head_weight(head: Any) -> Optional[torch.Tensor]:
    """Safely extracts weight tensor from head callable, module, or bound method."""
    w = getattr(head, "weight", None)
    if isinstance(w, torch.Tensor):
        return w

    self_obj = getattr(head, "__self__", None)
    if self_obj is not None:
        w = getattr(self_obj, "weight", None)
        if isinstance(w, torch.Tensor):
            return w
        if hasattr(self_obj, "get_output_embeddings"):
            try:
                emb = self_obj.get_output_embeddings()
                w = getattr(emb, "weight", None)
                if isinstance(w, torch.Tensor):
                    return w
            except Exception:
                pass

    closure = getattr(head, "__closure__", None)
    if closure:
        for cell in closure:
            val = getattr(cell, "cell_contents", None)
            w = getattr(val, "weight", None)
            if isinstance(w, torch.Tensor):
                return w
            if hasattr(val, "get_output_embeddings"):
                try:
                    emb = val.get_output_embeddings()
                    w = getattr(emb, "weight", None)
                    if isinstance(w, torch.Tensor):
                        return w
                except Exception:
                    pass

    return None


def _get_fp16_head_weight(weight: torch.Tensor) -> torch.Tensor:
    """Caches frozen lm_head weight in fp16 to avoid casting 933MB on every chunk."""
    global _CACHED_FP16_WEIGHT, _CACHED_FP16_WEIGHT_ID
    w_id = id(weight)
    if _CACHED_FP16_WEIGHT is None or _CACHED_FP16_WEIGHT_ID != w_id:
        _CACHED_FP16_WEIGHT_ID = w_id
        if weight.dtype != torch.float16:
            _CACHED_FP16_WEIGHT = weight.detach().to(torch.float16)
        else:
            _CACHED_FP16_WEIGHT = weight
    return _CACHED_FP16_WEIGHT


def chunk_cross_entropy_triton(
    head: Any,
    piece: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    denominator: float = 1.0,
    with_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Evaluates cross-entropy loss and metrics on a single sequence chunk with fast Tensor Core GEMMs."""
    if isinstance(piece, tuple):
        piece = piece[0]
    if isinstance(target, tuple):
        target = target[0]

    weight = _extract_head_weight(head)
    orig_piece_shape = piece.shape

    if weight is not None:
        head_weight_fp16 = _get_fp16_head_weight(weight)
        flat_piece = piece.contiguous().view(-1, orig_piece_shape[-1]).to(torch.float16)
        # Fast FP16 Tensor Core GEMM: [N_rows, D] @ [D, V] -> [N_rows, V]
        flat_logits = torch.matmul(flat_piece, head_weight_fp16.t())
        piece_in = None
    else:
        head_weight_fp16 = None
        if with_grad:
            piece_in = piece.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = head(piece_in)
        else:
            piece_in = None
            logits = head(piece)
        flat_logits = logits.contiguous().view(-1, logits.shape[-1])

    V = flat_logits.shape[-1]
    N_rows = flat_logits.shape[0]
    flat_targets = target.contiguous().view(-1)

    losses = torch.empty(N_rows, device=piece.device, dtype=torch.float32)
    entropies = torch.empty(N_rows, device=piece.device, dtype=torch.float32)
    correct = torch.empty(N_rows, device=piece.device, dtype=torch.int32)

    BLOCK_V = 4096
    num_warps = 4
    num_stages = 2

    _chunked_cross_entropy_fused_kernel[(N_rows,)](
        flat_logits,
        flat_targets,
        losses,
        entropies,
        correct,
        stride_logits_row=flat_logits.stride(0),
        V=V,
        denominator=float(denominator),
        ignore_index=ignore_index,
        BLOCK_V=BLOCK_V,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
    entropies = torch.nan_to_num(entropies, nan=0.0, posinf=0.0, neginf=0.0)

    total_loss = losses.sum()
    entropy = entropies.view(orig_piece_shape[:-1])
    token_loss = losses.view(orig_piece_shape[:-1])
    hits = correct.sum()

    if with_grad:
        if head_weight_fp16 is not None:
            # Fast direct Tensor Core GEMM backward: piece_grad = grad_logits @ W_fp16
            flat_piece_grad = torch.matmul(flat_logits, head_weight_fp16)
            piece_grad = flat_piece_grad.view(orig_piece_shape)
            del flat_logits
        elif piece_in is not None:
            grads = torch.autograd.grad(
                logits,
                piece_in,
                grad_outputs=flat_logits.view(logits.shape),
                retain_graph=False,
            )
            piece_grad = grads[0]
            del flat_logits
        else:
            piece_grad = torch.zeros_like(piece)
            del flat_logits
    else:
        piece_grad = None
        del flat_logits

    return total_loss, entropy, token_loss, hits, piece_grad


def make_triton_chunked_loss_function():
    """Builds a torch.autograd.Function integrating chunked head loss."""
    class TritonChunkedHeadLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
            # Avoid a device-to-host synchronization for the same denominator on every chunk.
            denominator = max(float(denominator), 1.0)
            grad = torch.zeros_like(hidden) if with_grad else None
            total = torch.zeros((), dtype=torch.float32, device=hidden.device)
            entropies, losses = [], []
            correct = torch.zeros((), dtype=torch.long, device=hidden.device)

            for start in range(0, hidden.shape[1], chunk):
                stop = start + chunk
                piece = hidden[:, start:stop]
                labels = target[:, start:stop]

                part, entropy, token_loss, hits, piece_grad = chunk_cross_entropy_triton(
                    head, piece, labels, ignore_index=ignore_index,
                    denominator=denominator, with_grad=with_grad
                )

                if with_grad:
                    grad[:, start:stop] = piece_grad.to(hidden.dtype)
                total = total + part
                entropies.append(entropy)
                losses.append(token_loss)
                correct = correct + hits

            token_entropy = torch.cat(entropies, dim=1)
            token_entropy = torch.nan_to_num(token_entropy, nan=0.0, posinf=0.0, neginf=0.0)
            token_loss = torch.cat(losses, dim=1)
            token_loss = torch.nan_to_num(token_loss, nan=0.0, posinf=0.0, neginf=0.0)

            ctx.with_grad = with_grad
            if with_grad:
                ctx.save_for_backward(grad)
            ctx.mark_non_differentiable(token_entropy, token_loss, correct)
            den = float(denominator) if float(denominator) > 0 else 1.0
            return total / den, token_entropy, token_loss, correct

        @staticmethod
        def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
            if not ctx.with_grad:
                return (None,) * 7
            (grad,) = ctx.saved_tensors
            return ((grad * grad_loss).to(grad.dtype), None, None, None, None, None, None)

    return TritonChunkedHeadLoss


def patch_soup_chunked_loss(chunk_size: int = 128):
    """Integrates Triton chunked head loss into Soup runtime with optimal chunk_size=128."""
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils.stream_loss_chunk import (
            ChunkedHeadLoss,
            causal_lm_of,
        )
        import soup_cli.utils.stream_loss_chunk as slc
    except ImportError:
        return

    chunk_size = int(os.environ.get("SOUP_LOSS_CHUNK", chunk_size))
    if chunk_size < 1:
        raise ValueError("SOUP_LOSS_CHUNK must be a positive integer")

    # Patch inner autograd function
    slc._chunked_function = make_triton_chunked_loss_function

    # Patch SFTTrainerWrapper setup hook with optimal chunk_size (default 128)
    original_setup = SFTTrainerWrapper._setup_transformers

    def _setup_transformers_with_custom_chunk(self, cfg, tcfg):
        result = original_setup(self, cfg, tcfg)
        model = getattr(self, "model", None)
        if model is None or not isinstance(model, torch.nn.Module):
            return result
        has_cuda = any(p.is_cuda for p in model.parameters()) if hasattr(model, "parameters") else False
        if not has_cuda:
            return result

        from ..frozen_head import compact_frozen_head

        compact_frozen_head(model)
        if getattr(self, "_stream_loss_chunk_hook", None) is None:
            try:
                self._stream_loss_chunk_hook = ChunkedHeadLoss(causal_lm_of(model), chunk_size)
                print(f"[SOUP-SOLUTION] Chunked head loss hook installed (chunk={chunk_size})", file=sys.stderr)
            except Exception as exc:
                print(f"[SOUP-SOLUTION] Warning: chunked head loss hook NOT installed: {exc!r}", file=sys.stderr)
        return result

    SFTTrainerWrapper._setup_transformers = _setup_transformers_with_custom_chunk
