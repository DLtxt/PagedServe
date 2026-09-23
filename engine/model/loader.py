"""safetensors checkpoint → Qwen3ForCausalLM.

Loading is strict: every tensor in the checkpoint must be consumed and every parameter filled, with
matching shapes. A silently skipped weight (say, a missing q_norm) produces fluent garbage rather
than a crash, so any mismatch is an error here.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from engine.distributed.parallel import SINGLE, ParallelState, check_parallel
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


_LAYER_TENSORS = {
    f"{part}.weight"
    for part in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm",
        "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm",
    )
}
_LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)")


def _belongs_elsewhere(key: str, first_layer: int, end_layer: int) -> bool:
    """Whether a checkpoint tensor this rank skipped is legitimately someone else's: another pipeline
    stage's layer, the embedding or final norm of a stage that does not hold them, or the redundant
    lm_head copy some tied checkpoints carry. Anything else is an unexpected tensor."""
    if key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        return True
    match = _LAYER_KEY.fullmatch(key)
    return bool(match) and match.group(2) in _LAYER_TENSORS and not first_layer <= int(match.group(1)) < end_layer


def load_model(path: Path, device: torch.device | str, dtype: torch.dtype, parallel: ParallelState = SINGLE) -> Qwen3ForCausalLM:
    """Build this rank's part of the model: its pipeline stage's layers, and its tensor-parallel slice
    of each. Slices are read lazily, so no rank ever loads the full checkpoint."""
    cfg = load_config(path)
    check_parallel(cfg, parallel.tp_size, parallel.pp_size)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg, parallel)
    expected = dict(model.named_parameters())

    tp, r = parallel.tp_size, parallel.tp_rank
    first, end = model.first_layer, model.first_layer + model.num_local_layers

    def rank_slice(total: int) -> tuple[int, int]:
        return r * total // tp, (r + 1) * total // tp

    q_rows = rank_slice(cfg.num_attention_heads * cfg.head_dim)
    kv_rows = rank_slice(cfg.num_key_value_heads * cfg.head_dim)
    mlp_rows = rank_slice(cfg.intermediate_size)
    vocab_rows = rank_slice(cfg.vocab_size)

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(f), framework="pt", device="cpu")) for f in files]
        where = {key: h for h in handles for key in h.keys()}
        used: set[str] = set()

        def take(key: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None) -> torch.Tensor:
            if key not in where:
                raise KeyError(f"checkpoint is missing {key}")
            used.add(key)
            tensor = where[key].get_slice(key)
            if rows is not None:
                return tensor[rows[0] : rows[1]]
            if cols is not None:
                return tensor[:, cols[0] : cols[1]]
            return tensor[:]

        weights: dict[str, torch.Tensor] = {}
        if model.embed_tokens is not None:
            weights["embed_tokens.weight"] = take("model.embed_tokens.weight", rows=vocab_rows)
        if model.norm is not None:
            weights["norm.weight"] = take("model.norm.weight")
        if model.lm_head is not None:
            weights["lm_head.weight"] = take("lm_head.weight", rows=vocab_rows)
        for i in range(first, end):
            src, dst = f"model.layers.{i}.", f"layers.{i - first}."
            weights[dst + "self_attn.qkv_proj.weight"] = torch.cat(
                [take(src + "self_attn.q_proj.weight", rows=q_rows),
                 take(src + "self_attn.k_proj.weight", rows=kv_rows),
                 take(src + "self_attn.v_proj.weight", rows=kv_rows)],
                dim=0,
            )
            weights[dst + "self_attn.o_proj.weight"] = take(src + "self_attn.o_proj.weight", cols=q_rows)
            weights[dst + "mlp.gate_up_proj.weight"] = torch.cat(
                [take(src + "mlp.gate_proj.weight", rows=mlp_rows), take(src + "mlp.up_proj.weight", rows=mlp_rows)], dim=0
            )
            weights[dst + "mlp.down_proj.weight"] = take(src + "mlp.down_proj.weight", cols=mlp_rows)
            for name in ("self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"):
                weights[dst + name] = take(src + name)
        unexpected = sorted(k for k in where.keys() - used if not _belongs_elsewhere(k, first, end))
        if unexpected:
            raise ValueError(f"checkpoint has {len(unexpected)} unexpected tensors, e.g. {unexpected[:5]}")

    missing = expected.keys() - weights.keys()
    extra = weights.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"parameter mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name, tensor in weights.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model shape {tuple(expected[name].shape)}")
        module_name, _, param_name = name.rpartition(".")
        module = model.get_submodule(module_name)
        module._parameters[param_name] = nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False)

    # Buffers were created on the meta device; rebuild the RoPE table for real, in fp32.
    model.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings, device=device)
    model.eval()
    logger.info(
        "loaded %s: layers %d-%d, tp rank %d/%d, %d tensors, to %s as %s",
        path, first, end - 1, r, tp, len(weights), device, dtype,
    )
    return model
