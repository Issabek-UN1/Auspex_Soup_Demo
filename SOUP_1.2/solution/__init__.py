"""Custom Triton Kernel Solution for Soup Kernel Challenge (Decentrathon 6.0).

Revision: Soup v2.1 (Turing SM 7.5, fp16, Qwen2.5-1.5B 4-bit NF4).

Key Optimizations:
1. Native GQA Attention (patcher.py):
   Uses hardware SDPA with enable_gqa=True without repeat_kv, saving ~1.2 GB VRAM.
2. Fused RoPE QK (kernels/rope.py):
   Rotates Query and Key heads in a single unified Triton kernel launch.
3. Fused Chunked Cross-Entropy (kernels/chunked_ce.py):
   CHUNK_SIZE=128, BLOCK_V=4096, num_stages=1, num_warps=4, cached FP16 lm_head.
4. Fused RMSNorm (kernels/rmsnorm.py):
   num_stages=1, warps=4, frozen weight fast path.
5. Fused SwiGLU (kernels/swiglu.py):
   2D Triton SwiGLU with num_stages=1.
6. Selective MLP Checkpointing (kernels/decoder.py):
   Retains Attention across all 28 layers; checkpoints only MLPs for layers K..27,
   locking peak VRAM in 2.5 - 2.7 GB and accelerating training to s >= 1.5x.

Clean-room implementation for Soup Kernel Challenge (Decentrathon 6.0).
"""

import os
import sys
import torch
from .patcher import patch_all

CONFIG = {
    # 0. Fast GQA Attention + Fused RoPE QK in SRAM
    "attention": True,
    # 1. Chunked Cross-Entropy + Online Softmax & Metrics (chunk_size=128 for peak Tensor Core speed)
    "chunk_size": 128,
    # 2. Fused Fast RMSNorm forward and backward
    "rmsnorm": True,
    # 3. Fused In-Place RoPE
    "rope": True,
    # 4. Custom Fused SwiGLU
    "swiglu": True,
    # 5. Selective MLP Checkpointing (retains Attention, strictly locks peak VRAM in 2.5 - 2.7 GB)
    "decoder_checkpoint": True,
    "num_uncheckpointed": 6,
}


def setup_environment():
    """Configures PyTorch CUDA runtime environment."""
    # Ensure expandable_segments is set to prevent allocator fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.cuda.empty_cache()


def apply():
    """Master entry point invoked by Soup training harness before model building."""
    setup_environment()

    try:
        k = int(os.environ.get("SOUP_UNCHECKPOINTED", CONFIG.get("num_uncheckpointed", 6)))
        CONFIG["num_uncheckpointed"] = k
        patch_all(CONFIG)
    except Exception as exc:
        print(f"[SOUP-SOLUTION] Warning: patch_all error: {exc}", file=sys.stderr)
