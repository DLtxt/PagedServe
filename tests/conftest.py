"""Shared fixtures. Model-backed tests run in fp32 on PAGEDSERVE_TEST_DEVICE (default cpu) and are
skipped, never auto-downloaded, when the weights are not already in the local HF cache."""

from __future__ import annotations

import os

import pytest
import torch

from engine.model.loader import load_model, resolve_model_path

MODEL = os.environ.get("PAGEDSERVE_MODEL", "Qwen/Qwen3-0.6B-Base")
TEST_DEVICE = os.environ.get("PAGEDSERVE_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session")
def model_path():
    try:
        return resolve_model_path(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(
            f"{MODEL} is not in the local HF cache. Download it once with: "
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\""
        )


@pytest.fixture(scope="session")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="session")
def qwen_fp32(model_path):
    return load_model(model_path, TEST_DEVICE, torch.float32)
