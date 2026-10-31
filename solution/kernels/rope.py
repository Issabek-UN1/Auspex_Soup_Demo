"""High-Performance Fused In-Place Rotary Position Embedding (RoPE) for Qwen2.5.
Optimized for NVIDIA Tesla T4 (fp16, Turing SM 7.5).

Key features:
1. In-Register 2D Rotation:
   Performs the half-dim rotation and sine/cosine mixing in GPU registers without
   slicing `x[..., :half]` or `x[..., half:]` in Python.
2. Zero Intermediate Allocations:
   Forward and backward passes write directly to output buffers without intermediate
   concatenations or copies.
3. Turing Architecture Tuning:
   num_stages=2 and num_warps=4 for peak register residency on SM 7.5.

Clean-room author implementation for Soup Kernel Challenge (Decentrathon 6.0).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_2d_fwd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_xb, stride_xh, stride_xs,
    stride_cb, stride_cs,
    stride_ob, stride_oh, stride_os,
    H,
    S,
    HALF_DIM: tl.constexpr,
):
    """2D-grid Fused RoPE Forward Kernel."""
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)

    b_idx = bh_idx // H
    h_idx = bh_idx % H

    cols = tl.arange(0, HALF_DIM)
    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols

    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

    offset1 = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs + cols
    offset2 = offset1 + HALF_DIM

    x1 = tl.load(x_ptr + offset1).to(tl.float32)
    x2 = tl.load(x_ptr + offset2).to(tl.float32)

    y1 = x1 * c - x2 * s_val
    y2 = x1 * s_val + x2 * c

    out_offset1 = b_idx * stride_ob + h_idx * stride_oh + s_idx * stride_os + cols
    out_offset2 = out_offset1 + HALF_DIM

    tl.store(out_ptr + out_offset1, y1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_offset2, y2.to(out_ptr.dtype.element_ty))


@triton.jit
def _fused_rope_2d_bwd_kernel(
    grad_ptr,
    cos_ptr,
    sin_ptr,
    dx_ptr,
    stride_gb, stride_gh, stride_gs,
    stride_cb, stride_cs,
    stride_dxb, stride_dxh, stride_dxs,
    H,
    S,
    HALF_DIM: tl.constexpr,
):
    """2D-grid Fused RoPE Backward Kernel."""
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)

    b_idx = bh_idx // H
    h_idx = bh_idx % H

    cols = tl.arange(0, HALF_DIM)
    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols

    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

    grad_offset1 = b_idx * stride_gb + h_idx * stride_gh + s_idx * stride_gs + cols
    grad_offset2 = grad_offset1 + HALF_DIM

    g1 = tl.load(grad_ptr + grad_offset1).to(tl.float32)
    g2 = tl.load(grad_ptr + grad_offset2).to(tl.float32)

    dx1 = g1 * c + g2 * s_val
    dx2 = g2 * c - g1 * s_val

    dx_offset1 = b_idx * stride_dxb + h_idx * stride_dxh + s_idx * stride_dxs + cols
    dx_offset2 = dx_offset1 + HALF_DIM

    tl.store(dx_ptr + dx_offset1, dx1.to(dx_ptr.dtype.element_ty))
    tl.store(dx_ptr + dx_offset2, dx2.to(dx_ptr.dtype.element_ty))


class FastRoPEFunction(torch.autograd.Function):
    """Autograd Function for Fused 2D RoPE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(cos, tuple):
            cos = cos[0]
        if isinstance(sin, tuple):
            sin = sin[0]

        orig_shape = x.shape
        B, H, S, D = orig_shape
        HALF_DIM = D // 2

        out = torch.empty_like(x)

        # Standardize cos/sin to 4D [B, 1, S, D]
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        stride_cb = cos.stride(0) if cos.shape[0] > 1 else 0
        stride_cs = cos.stride(2)

        grid = (B * H, S)
        _fused_rope_2d_fwd_kernel[grid](
            x,
            cos,
            sin,
            out,
            stride_xb=x.stride(0),
            stride_xh=x.stride(1),
            stride_xs=x.stride(2),
            stride_cb=stride_cb,
            stride_cs=stride_cs,
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            stride_os=out.stride(2),
            H=H,
            S=S,
            HALF_DIM=HALF_DIM,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(cos, sin)
        ctx.orig_shape = orig_shape
        ctx.HALF_DIM = HALF_DIM
        ctx.stride_cb = stride_cb
        ctx.stride_cs = stride_cs
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        cos, sin = ctx.saved_tensors
        B, H, S, D = ctx.orig_shape
        HALF_DIM = ctx.HALF_DIM

        dx = torch.empty_like(grad_output)
        grid = (B * H, S)

        _fused_rope_2d_bwd_kernel[grid](
            grad_output,
            cos,
            sin,
            dx,
            stride_gb=grad_output.stride(0),
            stride_gh=grad_output.stride(1),
            stride_gs=grad_output.stride(2),
            stride_cb=ctx.stride_cb,
            stride_cs=ctx.stride_cs,
            stride_dxb=dx.stride(0),
            stride_dxh=dx.stride(1),
            stride_dxs=dx.stride(2),
            H=H,
            S=S,
            HALF_DIM=HALF_DIM,
            num_warps=4,
            num_stages=2,
        )
        return dx, None, None


def apply_fast_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding to input tensor."""
    if isinstance(x, tuple):
        x = x[0]
    out = FastRoPEFunction.apply(x, cos, sin)
    if isinstance(out, tuple):
        out = out[0]
    return out


def patch_qwen2_rope():
    """Patches apply_rotary_pos_emb in transformers.models.qwen2.modeling_qwen2."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:
        return

    original_apply_rotary = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if isinstance(q, tuple):
            q = q[0]
        if isinstance(k, tuple):
            k = k[0]
        if not getattr(q, "is_cuda", False) or q.shape[-1] != 128:
            return original_apply_rotary(q, k, cos, sin, position_ids, unsqueeze_dim)

        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.ndim == 3 and unsqueeze_dim == 1:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_rot = apply_fast_rope(q, cos, sin)
        k_rot = apply_fast_rope(k, cos, sin)

        if isinstance(q_rot, tuple):
            q_rot = q_rot[0]
        if isinstance(k_rot, tuple):
            k_rot = k_rot[0]
        return q_rot, k_rot

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_patched
