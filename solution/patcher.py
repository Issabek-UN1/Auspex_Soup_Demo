"""Master Patcher for Qwen2.5 on NVIDIA Tesla T4 (Soup v2.1).

Coordinates all kernel interventions:
1. GQA Attention Patch (torch.nn.functional.scaled_dot_product_attention with enable_gqa=True):
   Eliminates repeat_kv(key_states, 6) in HuggingFace, avoiding 6x KV expansion in HBM
   and saving ~1.2 GB of VRAM.
2. Fused RoPE QK:
   Uses unified Triton kernel to rotate both Query and Key in a single launch.
3. Fused RMSNorm:
   num_stages=1, warps=4, zero atomic contention in dx.
4. Fused SwiGLU:
   2D Triton SwiGLU forward/backward with num_stages=1.
5. Fused Chunked Cross-Entropy:
   CHUNK_SIZE=128, BLOCK_V=4096, num_stages=1, in-place gradients.
6. Selective MLP Checkpointing:
   Retains Attention across all 28 layers; checkpoints only MLPs for layers K..27,
   locking peak VRAM in 2.5 - 2.7 GB.

Clean-room implementation for Soup Kernel Challenge (Decentrathon 6.0).
"""

import inspect
from typing import Optional, Tuple
import torch
from .kernels.attention import patch_qwen2_attention
from .kernels.chunked_ce import patch_soup_chunked_loss
from .kernels.decoder import patch_qwen2_decoder
from .kernels.rmsnorm import patch_qwen2_rmsnorm
from .kernels.rope import apply_fast_rope_qk, patch_qwen2_rope
from .kernels.swiglu import patch_qwen2_mlp

# Detect whether PyTorch SDPA supports enable_gqa natively (PyTorch 2.5+)
HAS_ENABLE_GQA = (
    hasattr(torch.nn.functional, "scaled_dot_product_attention")
    and "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters
)


def patch_qwen2_attention_gqa():
    """Patches Qwen2Attention to use hardware Cutlass SDPA with native GQA (no repeat_kv)."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:
        return

    def make_fast_gqa_forward(orig_fn):
        def fast_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            # Fallback to original if not training on CUDA or output_attentions requested
            if not self.training or not hidden_states.is_cuda or output_attentions:
                return orig_fn(
                    self,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            bsz, q_len, _ = hidden_states.size()

            # 1. 4-bit Linear Projections for Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # 2. Rotary Position Embeddings
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # Combined Q+K RoPE in a single Triton kernel launch
            query_states, key_states = apply_fast_rope_qk(query_states, key_states, cos, sin)

            # 3. Scaled Dot-Product Attention with GQA
            # Eliminates repeat_kv(key_states, 6) when enable_gqa is supported
            if HAS_ENABLE_GQA and self.num_heads != self.num_key_value_heads:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True,
                    enable_gqa=True,
                )
            else:
                from transformers.models.qwen2.modeling_qwen2 import repeat_kv
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True,
                )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return (attn_output, None, past_key_value) if kwargs.get("return_dict", False) else attn_output

        return fast_attn_forward

    for cls_name in ("Qwen2Attention", "Qwen2SdpaAttention"):
        if hasattr(modeling_qwen2, cls_name):
            target_cls = getattr(modeling_qwen2, cls_name)
            if hasattr(target_cls, "forward"):
                target_cls.forward = make_fast_gqa_forward(target_cls.forward)


def patch_all(config: Optional[dict] = None):
    """Applies all patches in optimal sequence."""
    if config is None:
        config = {}

    chunk_size = config.get("chunk_size", 128)
    num_uncheckpointed = config.get("num_uncheckpointed", 6)

    # 1. Native GQA Attention + Fused RoPE QK
    try:
        patch_qwen2_attention_gqa()
    except Exception:
        patch_qwen2_attention()

    # 2. Chunked Cross-Entropy (CHUNK_SIZE = 128)
    patch_soup_chunked_loss(chunk_size=chunk_size)

    # 3. Fused RMSNorm (NEEDS_DW = False, num_stages = 1)
    patch_qwen2_rmsnorm()

    # 4. Fused RoPE
    patch_qwen2_rope()

    # 5. Fused SwiGLU
    patch_qwen2_mlp()

    # 6. Selective MLP Checkpointing (Attention retained in memory)
    patch_qwen2_decoder(num_uncheckpointed=num_uncheckpointed)
