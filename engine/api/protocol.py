"""OpenAI-compatible schemas for /v1/completions, plus the vLLM extensions its benchmark client sends
(ignore_eos, priority). Matching the schema lets vLLM's own benchmark tool drive this server and real
vLLM with identical traffic and identical measurement code."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int | None = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Accepted so standard clients work, but only their neutral values are supported.
    n: int = 1
    logprobs: int | None = None
    echo: bool = False
    seed: int | None = None
    top_k: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    repetition_penalty: float | None = 1.0
    user: str | None = None
    # vLLM extensions
    ignore_eos: bool = False
    priority: float | None = None  # lower is served sooner under --admission-policy priority

    def unsupported(self) -> str | None:
        """Why this request asks for something the engine does not implement, if it does."""
        if self.n != 1:
            return "n > 1 is not supported"
        if self.logprobs is not None:
            return "logprobs are not supported"
        if self.echo:
            return "echo is not supported"
        if self.seed is not None:
            return "per-request seeds are not supported"
        if self.top_k not in (None, 0, -1):
            return "top_k is not supported"
        if self.presence_penalty or self.frequency_penalty:
            return "presence and frequency penalties are not supported"
        if self.repetition_penalty not in (None, 1.0):
            return "repetition_penalty is not supported"
        return None

    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    logprobs: None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "pagedserve"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
