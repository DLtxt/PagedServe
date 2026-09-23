"""safetensors checkpoint → Qwen3ForCausalLM.

Loading is strict: every tensor in the checkpoint must be consumed and every parameter filled, with
matching shapes. A silently skipped weight (say, a missing q_norm) produces fluent garbage rather
than a crash, so any mismatch is an error here.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from engine.model.qwen3 import Qwen3Config, Qwen3ForCausalLM, RotaryEmbedding

logger = logging.getLogger(__name__)

_ALLOW_PATTERNS = ["*.json", "*.safetensors", "merges.txt"]


def resolve_model_path(model: str, local_files_only: bool = False) -> Path:
    """A local directory as-is, otherwise a HF Hub id resolved through the HF cache."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=_ALLOW_PATTERNS, local_files_only=local_files_only))


def load_config(path: Path) -> Qwen3Config:
    with open(path / "config.json") as f:
        cfg = Qwen3Config.from_hf_dict(json.load(f))
    gen_path = path / "generation_config.json"
    if gen_path.exists():
        with open(gen_path) as f:
            eos = json.load(f).get("eos_token_id")
        if eos is not None:
            extra = tuple(eos) if isinstance(eos, list) else (eos,)
            ids = tuple(dict.fromkeys(cfg.eos_token_ids + extra))
            cfg = Qwen3Config(**{**cfg.__dict__, "eos_token_ids": ids})
    return cfg


def load_model(path: Path, device: torch.device | str, dtype: torch.dtype) -> Qwen3ForCausalLM:
    cfg = load_config(path)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    expected = dict(model.named_parameters())

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(f), framework="pt", device="cpu")) for f in files]
        where = {key: h for h in handles for key in h.keys()}

        def take(key: str) -> torch.Tensor:
            if key not in where:
                raise KeyError(f"checkpoint is missing {key}")
            return where.pop(key).get_tensor(key)

        weights: dict[str, torch.Tensor] = {
            "embed_tokens.weight": take("model.embed_tokens.weight"),
            "norm.weight": take("model.norm.weight"),
        }
        if cfg.tie_word_embeddings:
            where.pop("lm_head.weight", None)  # some tied checkpoints store a redundant copy
        else:
            weights["lm_head.weight"] = take("lm_head.weight")
        for i in range(cfg.num_hidden_layers):
            src, dst = f"model.layers.{i}.", f"layers.{i}."
            weights[dst + "self_attn.qkv_proj.weight"] = torch.cat(
                [take(src + f"self_attn.{p}_proj.weight") for p in ("q", "k", "v")], dim=0
            )
            weights[dst + "mlp.gate_up_proj.weight"] = torch.cat(
                [take(src + f"mlp.{p}_proj.weight") for p in ("gate", "up")], dim=0
            )
            for name in (
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ):
                weights[dst + name] = take(src + name)
        if where:
            raise ValueError(f"checkpoint has {len(where)} unexpected tensors, e.g. {sorted(where)[:5]}")

    missing = expected.keys() - weights.keys()
    unexpected = weights.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"parameter mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, tensor in weights.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model shape {tuple(expected[name].shape)}")
        module_name, _, param_name = name.rpartition(".")
        module = model.get_submodule(module_name)
        module._parameters[param_name] = nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False)

    # Buffers were created on the meta device; rebuild the RoPE table for real, in fp32.
    model.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings, device=device)
    model.eval()
    logger.info("loaded %s (%d tensors) to %s as %s", path, len(weights), device, dtype)
    return model
