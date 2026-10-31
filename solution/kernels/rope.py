"""High-Performance Fused In-Place Rotary Position Embedding (RoPE) for Qwen2.5.
Optimized for NVIDIA Tesla T4 (fp16, SM 7.5).

Key features:
1. Combined Q+K RoPE (FusedRoPEQKFunction):
   Rotates both Query (12 heads) and Key (2 heads) in a single unified Triton kernel launch.
   Eliminates kernel launch overhead and intermediate tensor allocations in PyTorch.
2. In-Register 2D Rotation:
   Performs the half-dim rotation and sine/cosine mixing in GPU registers without
   slicing `x[..., :half]` or `x[..., half:]` in Python.
3. Turing Architecture Tuning:
   num_stages=1 and num_warps=4 for peak register residency on SM 7.5.

Clean-room implementation for Soup Kernel Challenge (Decentrathon 6.0).
"""

from typing import Tuple
import torch
import triton
import triton.language as tl


# =====================================================================
# 1. Combined Q+K Fused RoPE Kernels
# =====================================================================

@triton.jit
def _fused_rope_qk_fwd_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    out_q_ptr, out_k_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_cb, stride_cs,
    stride_oqb, stride_oqh, stride_oqs,
    stride_okb, stride_okh, stride_oks,
    H_q, H_k, S,
    HALF_DIM: tl.constexpr,
):
    """Unified RoPE Forward Kernel for both Q and K in a single launch."""
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)

    total_h = H_q + H_k
    b_idx = bh_idx // total_h
    h_idx = bh_idx % total_h

    cols = tl.arange(0, HALF_DIM)
    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols
    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

    is_q = h_idx < H_q

    if is_q:
        q_h = h_idx
        offset1 = b_idx * stride_qb + q_h * stride_qh + s_idx * stride_qs + cols
        offset2 = offset1 + HALF_DIM
        x1 = tl.load(q_ptr + offset1).to(tl.float32)
        x2 = tl.load(q_ptr + offset2).to(tl.float32)
        y1 = x1 * c - x2 * s_val
        y2 = x1 * s_val + x2 * c
        out_offset1 = b_idx * stride_oqb + q_h * stride_oqh + s_idx * stride_oqs + cols
        out_offset2 = out_offset1 + HALF_DIM
        tl.store(out_q_ptr + out_offset1, y1.to(out_q_ptr.dtype.element_ty))
        tl.store(out_q_ptr + out_offset2, y2.to(out_q_ptr.dtype.element_ty))
    else:
        k_h = h_idx - H_q
        offset1 = b_idx * stride_kb + k_h * stride_kh + s_idx * stride_ks + cols
        offset2 = offset1 + HALF_DIM
        x1 = tl.load(k_ptr + offset1).to(tl.float32)
        x2 = tl.load(k_ptr + offset2).to(tl.float32)
        y1 = x1 * c - x2 * s_val
        y2 = x1 * s_val + x2 * c
        out_offset1 = b_idx * stride_okb + k_h * stride_okh + s_idx * stride_oks + cols
        out_offset2 = out_offset1 + HALF_DIM
        tl.store(out_k_ptr + out_offset1, y1.to(out_k_ptr.dtype.element_ty))
        tl.store(out_k_ptr + out_offset2, y2.to(out_k_ptr.dtype.element_ty))


@triton.jit
def _fused_rope_qk_bwd_kernel(
    grad_q_ptr, grad_k_ptr,
    cos_ptr, sin_ptr,
    dq_ptr, dk_ptr,
    stride_gqb, stride_gqh, stride_gqs,
    stride_gkb, stride_gkh, stride_gks,
    stride_cb, stride_cs,
    stride_dqb, stride_dqh, stride_dqs,
    stride_dkb, stride_dkh, stride_dks,
    H_q, H_k, S,
    HALF_DIM: tl.constexpr,
):
    """Unified RoPE Backward Kernel for both Q and K in a single launch."""
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)

    total_h = H_q + H_k
    b_idx = bh_idx // total_h
    h_idx = bh_idx % total_h

    cols = tl.arange(0, HALF_DIM)
    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols
    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

    is_q = h_idx < H_q

    if is_q:
        q_h = h_idx
        offset1 = b_idx * stride_gqb + q_h * stride_gqh + s_idx * stride_gqs + cols
        offset2 = offset1 + HALF_DIM
        g1 = tl.load(grad_q_ptr + offset1).to(tl.float32)
        g2 = tl.load(grad_q_ptr + offset2).to(tl.float32)
        # Inverse rotation for backward
        dx1 = g1 * c + g2 * s_val
        dx2 = g2 * c - g1 * s_val
        out_offset1 = b_idx * stride_dqb + q_h * stride_dqh + s_idx * stride_dqs + cols
        out_offset2 = out_offset1 + HALF_DIM
        tl.store(dq_ptr + out_offset1, dx1.to(dq_ptr.dtype.element_ty))
        tl.store(dq_ptr + out_offset2, dx2.to(dq_ptr.dtype.element_ty))
    else:
        k_h = h_idx - H_q
        offset1 = b_idx * stride_gkb + k_h * stride_gkh + s_idx * stride_gks + cols
        offset2 = offset1 + HALF_DIM
        g1 = tl.load(grad_k_ptr + offset1).to(tl.float32)
        g2 = tl.load(grad_k_ptr + offset2).to(tl.float32)
        # Inverse rotation for backward
        dx1 = g1 * c + g2 * s_val
        dx2 = g2 * c - g1 * s_val
        out_offset1 = b_idx * stride_dkb + k_h * stride_dkh + s_idx * stride_dks + cols
        out_offset2 = out_offset1 + HALF_DIM
        tl.store(dk_ptr + out_offset1, dx1.to(dk_ptr.dtype.element_ty))
        tl.store(dk_ptr + out_offset2, dx2.to(dk_ptr.dtype.element_ty))


class FusedRoPEQKFunction(torch.autograd.Function):
    """Autograd Function for Combined Q+K RoPE."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, H_q, S, D = q.shape
        _, H_k, _, _ = k.shape
        HALF_DIM = D // 2

        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

        stride_cb = cos.stride(0) if (cos.ndim >= 3 and cos.shape[0] > 1) else 0
        stride_cs = cos.stride(2) if cos.ndim == 4 else cos.stride(1)

        total_h = H_q + H_k
        grid = (B * total_h, S)

        _fused_rope_qk_fwd_kernel[grid](
            q, k,
            cos, sin,
            out_q, out_k,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qs=q.stride(2),
            stride_kb=k.stride(0), stride_kh=k.stride(1), stride_ks=k.stride(2),
            stride_cb=stride_cb, stride_cs=stride_cs,
            stride_oqb=out_q.stride(0), stride_oqh=out_q.stride(1), stride_oqs=out_q.stride(2),
            stride_okb=out_k.stride(0), stride_okh=out_k.stride(1), stride_oks=out_k.stride(2),
            H_q=H_q, H_k=H_k, S=S,
            HALF_DIM=HALF_DIM,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(cos, sin)
        ctx.H_q = H_q
        ctx.H_k = H_k
        ctx.S = S
        ctx.B = B
        ctx.HALF_DIM = HALF_DIM
        ctx.stride_cb = stride_cb
        ctx.stride_cs = stride_cs
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        cos, sin = ctx.saved_tensors
        H_q, H_k, S, B = ctx.H_q, ctx.H_k, ctx.S, ctx.B
        HALF_DIM = ctx.HALF_DIM

        dq = torch.empty_like(grad_q)
        dk = torch.empty_like(grad_k)

        total_h = H_q + H_k
        grid = (B * total_h, S)

        _fused_rope_qk_bwd_kernel[grid](
            grad_q, grad_k,
            cos, sin,
            dq, dk,
            stride_gqb=grad_q.stride(0), stride_gqh=grad_q.stride(1), stride_gqs=grad_q.stride(2),
            stride_gkb=grad_k.stride(0), stride_gkh=grad_k.stride(1), stride_gks=grad_k.stride(2),
            stride_cb=ctx.stride_cb, stride_cs=ctx.stride_cs,
            stride_dqb=dq.stride(0), stride_dqh=dq.stride(1), stride_dqs=dq.stride(2),
            stride_dkb=dk.stride(0), stride_dkh=dk.stride(1), stride_dks=dk.stride(2),
            H_q=H_q, H_k=H_k, S=S,
            HALF_DIM=HALF_DIM,
            num_warps=4,
            num_stages=1,
        )
        return dq, dk, None, None


def apply_fast_rope_qk(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotates both Query and Key in a single Triton kernel launch."""
    return FusedRoPEQKFunction.apply(q, k, cos, sin)


# =====================================================================
# 2. Single-Tensor RoPE (Maintained for Backward Compatibility & Tests)
# =====================================================================

@triton.jit
def _fused_rope_2d_fwd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_xb,
    stride_xh,
    stride_xs,
    stride_cb,
    stride_cs,
    stride_ob,
    stride_oh,
    stride_os,
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

    x_offset1 = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs + cols
    x_offset2 = x_offset1 + HALF_DIM

    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols

    x1 = tl.load(x_ptr + x_offset1).to(tl.float32)
    x2 = tl.load(x_ptr + x_offset2).to(tl.float32)
    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

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
    stride_gb,
    stride_gh,
    stride_gs,
    stride_cb,
    stride_cs,
    stride_dxb,
    stride_dxh,
    stride_dxs,
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

    grad_offset1 = b_idx * stride_gb + h_idx * stride_gh + s_idx * stride_gs + cols
    grad_offset2 = grad_offset1 + HALF_DIM

    cos_offset = b_idx * stride_cb + s_idx * stride_cs + cols

    g1 = tl.load(grad_ptr + grad_offset1).to(tl.float32)
    g2 = tl.load(grad_ptr + grad_offset2).to(tl.float32)
    c = tl.load(cos_ptr + cos_offset).to(tl.float32)
    s_val = tl.load(sin_ptr + cos_offset).to(tl.float32)

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
        orig_shape = x.shape
        B, H, S, D = orig_shape
        HALF_DIM = D // 2

        out = torch.empty_like(x)

        stride_cb = cos.stride(0) if (cos.ndim >= 3 and cos.shape[0] > 1) else 0
        stride_cs = cos.stride(2) if cos.ndim == 4 else cos.stride(1)

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
            num_stages=1,
        )

        ctx.save_for_backward(cos, sin)
        ctx.orig_shape = orig_shape
        ctx.HALF_DIM = HALF_DIM
        ctx.stride_cb = stride_cb
        ctx.stride_cs = stride_cs
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
            num_stages=1,
        )
        return dx, None, None


def apply_fast_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding to input tensor."""
    return FastRoPEFunction.apply(x, cos, sin)


def patch_qwen2_rope():
    """Patches apply_rotary_pos_emb in transformers.models.qwen2.modeling_qwen2."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:
        return

    original_apply_rotary = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not q.is_cuda or q.shape[-1] != 128:
            return original_apply_rotary(q, k, cos, sin, position_ids, unsqueeze_dim)

        if cos.ndim == 3 and unsqueeze_dim == 1:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Fast combined Q+K RoPE
        return apply_fast_rope_qk(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_patched
