"""A tiny Qwen3 checkpoint with random weights, generated on the fly: a real Qwen3 architecture that
runs in milliseconds, for testing tensor and pipeline parallelism without downloading anything.

It keeps the shapes that matter: head_dim decoupled from hidden_size / num_heads (as in every Qwen3),
grouped-query attention, and, by default, an untied lm_head like Qwen3-8B and larger. Weights are scaled
so logits are peaked, which keeps near-ties rare and the greedy comparisons strict.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def make_tiny_qwen3(
    directory: Path,
    tie_word_embeddings: bool = False,
    num_layers: int = 4,
    hidden: int = 64,
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 32,
    intermediate: int = 128,
    vocab: int = 512,
    seed: int = 0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_act": "silu",
        "hidden_size": hidden, "intermediate_size": intermediate, "num_hidden_layers": num_layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "vocab_size": vocab, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "max_position_embeddings": 2048,
        "tie_word_embeddings": tie_word_embeddings, "attention_bias": False, "rope_scaling": None,
        "use_sliding_window": False, "eos_token_id": 0, "bos_token_id": 0, "torch_dtype": "float32",
    }
    g = torch.Generator().manual_seed(seed)

    def w(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * std

    def norm(dim: int) -> torch.Tensor:
        return 1.0 + w(dim, std=0.1)

    tensors = {"model.embed_tokens.weight": w(vocab, hidden, std=1.0), "model.norm.weight": norm(hidden)}
    for i in range(num_layers):
        p = f"model.layers.{i}."
        tensors.update({
            p + "self_attn.q_proj.weight": w(heads * head_dim, hidden, std=hidden**-0.5),
            p + "self_attn.k_proj.weight": w(kv_heads * head_dim, hidden, std=hidden**-0.5),
            p + "self_attn.v_proj.weight": w(kv_heads * head_dim, hidden, std=hidden**-0.5),
            p + "self_attn.o_proj.weight": w(hidden, heads * head_dim, std=(heads * head_dim) ** -0.5),
            p + "self_attn.q_norm.weight": norm(head_dim),
            p + "self_attn.k_norm.weight": norm(head_dim),
            p + "mlp.gate_proj.weight": w(intermediate, hidden, std=hidden**-0.5),
            p + "mlp.up_proj.weight": w(intermediate, hidden, std=hidden**-0.5),
            p + "mlp.down_proj.weight": w(hidden, intermediate, std=intermediate**-0.5),
            p + "input_layernorm.weight": norm(hidden),
            p + "post_attention_layernorm.weight": norm(hidden),
        })
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = w(vocab, hidden, std=0.6)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config))
    return directory
