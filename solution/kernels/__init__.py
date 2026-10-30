"""Triton Kernels package for Soup Kernel Challenge (Decentrathon 6.0)."""

from .chunked_ce import (
    _chunked_cross_entropy_fused_kernel,
    chunk_cross_entropy_triton,
    make_triton_chunked_loss_function,
    patch_soup_chunked_loss,
)
from .rmsnorm import (
    _rms_norm_fwd_kernel,
    _rms_norm_bwd_dx_kernel,
    FastRMSNormFunction,
    fast_rms_norm,
    patch_qwen2_rmsnorm,
)
from .rope import (
    _fused_rope_2d_fwd_kernel,
    _fused_rope_2d_bwd_kernel,
    FastRoPEFunction,
    apply_fast_rope,
    patch_qwen2_rope,
)
from .swiglu import (
    _custom_swiglu_fwd_kernel,
    _custom_swiglu_bwd_kernel,
    FastSwiGLUFunction,
    fast_swiglu,
    patch_qwen2_mlp,
)
from .attention import patch_qwen2_attention
from .decoder import patch_qwen2_decoder

__all__ = [
    "_chunked_cross_entropy_fused_kernel",
    "chunk_cross_entropy_triton",
    "make_triton_chunked_loss_function",
    "patch_soup_chunked_loss",
    "_rms_norm_fwd_kernel",
    "_rms_norm_bwd_dx_kernel",
    "FastRMSNormFunction",
    "fast_rms_norm",
    "patch_qwen2_rmsnorm",
    "_fused_rope_2d_fwd_kernel",
    "_fused_rope_2d_bwd_kernel",
    "FastRoPEFunction",
    "apply_fast_rope",
    "patch_qwen2_rope",
    "_custom_swiglu_fwd_kernel",
    "_custom_swiglu_bwd_kernel",
    "FastSwiGLUFunction",
    "fast_swiglu",
    "patch_qwen2_mlp",
    "patch_qwen2_attention",
    "patch_qwen2_decoder",
]
