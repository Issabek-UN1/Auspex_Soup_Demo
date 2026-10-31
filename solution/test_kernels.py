"""Unit tests for custom Triton kernels against plain PyTorch references.
Validates numerical equivalence for forward and backward passes.

Run on GPU:
    pytest solution/test_kernels.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")

TOLERANCE = {
    torch.float16: 2e-3,
    torch.float32: 1e-4,
}


# =====================================================================
# 1. Custom SwiGLU Tests
# =====================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (2, 64, 1024), (1, 16, 512)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.kernels.swiglu import fast_swiglu

    torch.manual_seed(42)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    # PyTorch reference (in fp32 for numerical baseline)
    ref_gate = gate.float().detach().clone().requires_grad_(True)
    ref_up = up.float().detach().clone().requires_grad_(True)
    ref_out = torch.nn.functional.silu(ref_gate) * ref_up
    ref_out.backward(grad.float())

    ref_dgate = ref_gate.grad.to(dtype)
    ref_dup = ref_up.grad.to(dtype)

    # Custom Triton kernel
    out = fast_swiglu(gate, up)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad, ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad, ref_dup, rtol=tol, atol=tol)


# =====================================================================
# 2. Fused RMSNorm Tests
# =====================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 128, 1536), (1, 16, 1536)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.kernels.rmsnorm import fast_rms_norm

    torch.manual_seed(42)
    D = shape[-1]
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(D, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    # PyTorch reference
    ref_x = x.float().detach().clone().requires_grad_(True)
    ref_w = weight.float().detach().clone().requires_grad_(True)
    var = ref_x.pow(2).mean(-1, keepdim=True)
    ref_out = ref_x * torch.rsqrt(var + eps) * ref_w
    ref_out.backward(grad.float())

    ref_dx = ref_x.grad.to(dtype)
    ref_dw = ref_w.grad.to(dtype)

    # Triton kernel
    out = fast_rms_norm(x, weight, eps)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad, ref_dw, rtol=tol, atol=tol)


# =====================================================================
# 3. Fused Chunked Cross-Entropy Tests
# =====================================================================

@pytest.mark.parametrize("N_rows, V", [(16, 151936), (32, 4096), (4, 1000)])
def test_chunked_ce_matches_pytorch(N_rows, V):
    from solution.kernels.chunked_ce import _chunked_cross_entropy_fused_kernel

    torch.manual_seed(42)
    logits = torch.randn(N_rows, V, device="cuda", dtype=torch.float16)
    targets = torch.randint(0, V, (N_rows,), device="cuda", dtype=torch.long)
    # Mask out some targets as ignore_index (-100)
    targets[0] = -100

    # PyTorch reference
    ref_logits = logits.float().detach().clone().requires_grad_(True)
    mask = (targets != -100)
    denom = mask.sum().clamp(min=1).float()

    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits, targets, ignore_index=-100, reduction="sum"
    ) / denom
    ref_loss.backward()
    ref_grad = ref_logits.grad.half()

    with torch.no_grad():
        log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        ref_entropy = -(torch.exp(log_p) * log_p).sum(-1)
        ref_correct = ((logits.argmax(dim=-1) == targets) & mask).int()

    # Triton kernel
    tri_logits = logits.clone()
    losses = torch.empty(N_rows, device="cuda", dtype=torch.float32)
    entropies = torch.empty(N_rows, device="cuda", dtype=torch.float32)
    correct = torch.empty(N_rows, device="cuda", dtype=torch.int32)

    BLOCK_V = 4096
    _chunked_cross_entropy_fused_kernel[(N_rows,)](
        tri_logits,
        targets,
        losses,
        entropies,
        correct,
        stride_logits_row=tri_logits.stride(0),
        V=V,
        denominator=denom.item(),
        ignore_index=-100,
        BLOCK_V=BLOCK_V,
        num_warps=4,
        num_stages=1,
    )

    tri_loss = losses.sum() / denom

    torch.testing.assert_close(tri_loss, ref_loss, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(entropies, ref_entropy, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(correct, ref_correct)
    torch.testing.assert_close(tri_logits, ref_grad, rtol=2e-3, atol=2e-3)


# =====================================================================
# 4. Fused 2D RoPE Tests
# =====================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 12, 128, 128), (2, 2, 64, 128)])
def test_rope_matches_pytorch(dtype, shape):
    from solution.kernels.rope import apply_fast_rope

    torch.manual_seed(42)
    B, H, S, D = shape
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(S, device="cuda").float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)

    def ref_rotate_half(tensor):
        x1 = tensor[..., : tensor.shape[-1] // 2]
        x2 = tensor[..., tensor.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    ref_x = x.clone().detach().requires_grad_(True)
    ref_out = (ref_x * cos) + (ref_rotate_half(ref_x) * sin)
    ref_out.backward(grad)

    tri_out = apply_fast_rope(x, cos, sin)
    tri_out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(tri_out, ref_out, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rope_qk_matches_pytorch(dtype):
    from solution.kernels.rope import apply_fast_rope_qk

    torch.manual_seed(42)
    B, H_q, H_k, S, D = 2, 12, 2, 64, 128
    q = torch.randn((B, H_q, S, D), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((B, H_k, S, D), device="cuda", dtype=dtype, requires_grad=True)
    grad_q = torch.randn((B, H_q, S, D), device="cuda", dtype=dtype)
    grad_k = torch.randn((B, H_k, S, D), device="cuda", dtype=dtype)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(S, device="cuda").float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)

    def ref_rotate_half(tensor):
        x1 = tensor[..., : tensor.shape[-1] // 2]
        x2 = tensor[..., tensor.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    ref_q = q.clone().detach().requires_grad_(True)
    ref_k = k.clone().detach().requires_grad_(True)
    ref_out_q = (ref_q * cos) + (ref_rotate_half(ref_q) * sin)
    ref_out_k = (ref_k * cos) + (ref_rotate_half(ref_k) * sin)
    (ref_out_q * grad_q).sum().backward()
    (ref_out_k * grad_k).sum().backward()

    tri_out_q, tri_out_k = apply_fast_rope_qk(q, k, cos, sin)
    (tri_out_q * grad_q).sum().backward()
    (tri_out_k * grad_k).sum().backward()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(tri_out_q, ref_out_q, rtol=tol, atol=tol)
    torch.testing.assert_close(tri_out_k, ref_out_k, rtol=tol, atol=tol)
    torch.testing.assert_close(q.grad, ref_q.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(k.grad, ref_k.grad, rtol=tol, atol=tol)


# =====================================================================
# 5. Checkpointed MLP Activation Recomputation Test
# =====================================================================

def test_checkpointed_mlp_matches_reference():
    """Validates that selective activation checkpointing in MLP produces exact gradients."""
    from solution.kernels.swiglu import fast_swiglu

    torch.manual_seed(42)
    B, S, H, intermediate = 2, 64, 256, 512

    scale_in = 1.0 / (H ** 0.5)
    scale_mid = 1.0 / (intermediate ** 0.5)

    raw_x = torch.randn(B, S, H, device="cuda", dtype=torch.float16) * 0.1
    x = raw_x.detach().clone().requires_grad_(True)
    grad = torch.randn(B, S, H, device="cuda", dtype=torch.float16) * 0.1

    gate_w = torch.randn(intermediate, H, device="cuda", dtype=torch.float16) * scale_in
    up_w = torch.randn(intermediate, H, device="cuda", dtype=torch.float16) * scale_in
    down_w = torch.randn(H, intermediate, device="cuda", dtype=torch.float16) * scale_mid

    class DummyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(H, intermediate, bias=False, dtype=torch.float16, device="cuda")
            self.up_proj = torch.nn.Linear(H, intermediate, bias=False, dtype=torch.float16, device="cuda")
            self.down_proj = torch.nn.Linear(intermediate, H, bias=False, dtype=torch.float16, device="cuda")
            self.gate_proj.weight.data.copy_(gate_w)
            self.up_proj.weight.data.copy_(up_w)
            self.down_proj.weight.data.copy_(down_w)
            self.training = True

        def forward(self, hidden_states):
            def _mlp_forward(h):
                return self.down_proj(fast_swiglu(self.gate_proj(h), self.up_proj(h)))
            return torch.utils.checkpoint.checkpoint(_mlp_forward, hidden_states, use_reentrant=False)

    ref_x = raw_x.detach().clone().requires_grad_(True)
    ref_gate = torch.nn.functional.linear(ref_x, gate_w)
    ref_up = torch.nn.functional.linear(ref_x, up_w)
    ref_out = torch.nn.functional.linear(torch.nn.functional.silu(ref_gate) * ref_up, down_w)
    ref_out.backward(grad)

    mlp = DummyMLP().to("cuda")
    out = mlp(x)
    out.backward(grad)

    torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-3, atol=2e-3)


# =====================================================================
# 6. Fast Tensor Core Linear Layer Test
# =====================================================================

def test_fast_linear_matches_pytorch():
    """Validates that fast FP16 linear computes exact projections and gradients."""
    torch.manual_seed(42)
    B, S, in_features, out_features = 2, 64, 256, 512
    x = torch.randn(B, S, in_features, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
    grad = torch.randn(B, S, out_features, device="cuda", dtype=torch.float16)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_out = torch.nn.functional.linear(ref_x, w)
    ref_out.backward(grad)

    test_x = x.detach().clone().requires_grad_(True)
    test_out = torch.nn.functional.linear(test_x, w)
    test_out.backward(grad)

    torch.testing.assert_close(test_out, ref_out)
    torch.testing.assert_close(test_x.grad, ref_x.grad)
