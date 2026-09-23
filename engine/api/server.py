"""OpenAI-compatible HTTP front end: async FastAPI handlers in front of the synchronous engine.

    POST /v1/completions   streaming (server-sent events) or a single JSON response
    GET  /v1/models        GET /health        GET /stats

The engine loop is a task on the same event loop. It never awaits inside a step and yields once
between steps so handlers can run; handlers reach it only through queues. The server binds to
127.0.0.1 by default: this is an unauthenticated endpoint and belongs on localhost.

    python -m engine.api.server --model Qwen/Qwen3-0.6B-Base --port 8000

Split across GPUs, launch every rank with torchrun; rank 0 serves HTTP and the rest become workers:

    torchrun --nproc-per-node 2 -m engine.api.server --tensor-parallel-size 2
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from engine.api.protocol import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorInfo,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)
from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.sequence import RequestOutput, SamplingParams, Sequence

logger = logging.getLogger("engine.api")


class AsyncEngine:
    """The bridge between the async handlers and the synchronous engine."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self.output_queues: dict[int, asyncio.Queue[RequestOutput]] = {}
        self.has_new_work = asyncio.Event()

    async def run_loop(self) -> None:
        while True:
            if not self.engine.has_work():
                await self.has_new_work.wait()  # idle without spinning
                self.has_new_work.clear()
            try:
                outputs = self.engine.step()
            except Exception as exc:  # a failed step must not strand every waiting client
                logger.exception("engine step failed")
                self._fail_all(f"engine error: {exc!r}")
                continue
            for out in outputs:
                queue = self.output_queues.get(out.seq_id)
                if queue is not None:
                    queue.put_nowait(out)
            await asyncio.sleep(0)  # yield so handlers get scheduled

    async def generate(self, seq: Sequence) -> AsyncIterator[RequestOutput]:
        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self.output_queues[seq.seq_id] = queue
        self.engine.add_request(seq)
        self.has_new_work.set()
        try:
            while True:
                out = await queue.get()
                yield out
                if out.finished:
                    return
        finally:
            # Runs on completion, on errors, and on cancellation; abort is a no-op for a finished
            # request. Disconnects do not rely on it alone: see _EngineStream.
            self.engine.abort(seq.seq_id)
            self.output_queues.pop(seq.seq_id, None)

    def abort_request(self, seq_id: int) -> None:
        """Abort a request from outside its own task (a disconnect listener): free its blocks now and
        hand its generator a terminal output so it finishes without waiting to be cancelled."""
        self.engine.abort(seq_id)
        queue = self.output_queues.get(seq_id)
        if queue is not None:
            queue.put_nowait(RequestOutput(seq_id, [], "", True, "abort", 0, 0))

    def _fail_all(self, message: str) -> None:
        for seq_id, queue in list(self.output_queues.items()):
            self.engine.abort(seq_id)
            queue.put_nowait(RequestOutput(seq_id, [], "", True, "error", 0, 0, error=message))


class _EngineStream(StreamingResponse):
    """A StreamingResponse that aborts its request in the engine the moment the client disconnects.

    Starlette reacts to http.disconnect by cancelling the streaming task through an anyio cancel scope,
    and anyio delivers that cancellation only while the task is blocked on a pending future. Our
    generator's queue receives a token every engine iteration, so each delivery attempt finds the task
    just woken and skips it: the cancellation can starve for the whole generation while the engine keeps
    decoding for nobody. Aborting straight from the disconnect signal does not depend on that timing.
    """

    def __init__(self, content, on_disconnect, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self.on_disconnect = on_disconnect

    async def listen_for_disconnect(self, receive) -> None:  # ASGI spec < 2.4 (uvicorn today)
        await super().listen_for_disconnect(receive)
        self.on_disconnect()

    async def stream_response(self, send) -> None:  # ASGI spec >= 2.4: a failed send is the signal
        try:
            await super().stream_response(send)
        except OSError:
            self.on_disconnect()
            raise


def _error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    body = ErrorResponse(error=ErrorInfo(message=message, type=kind, code=status))
    return JSONResponse(status_code=status, content=body.model_dump())


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _prompt_ids(prompt, tokenizer) -> list[int]:
    if isinstance(prompt, str):
        return tokenizer(prompt).input_ids
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
        return _prompt_ids(prompt[0], tokenizer)
    if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
        return list(prompt)
    raise ValueError("batched prompts are not supported: send one prompt per request")


def build_app(engine: LLMEngine, served_model_name: str) -> FastAPI:
    async_engine = AsyncEngine(engine)
    served_names = {served_model_name, engine.config.model}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(async_engine.run_loop())
        task.add_done_callback(lambda t: t.cancelled() or t.exception() is None or logger.error("engine loop died: %r", t.exception()))
        yield
        task.cancel()

    app = FastAPI(title="PagedServe", lifespan=lifespan)
    app.state.async_engine = async_engine

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, str(exc.errors()))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict:
        return engine.stats()

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name, created=int(time.time()))])

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_request: Request):
        if (reason := req.unsupported()) is not None:
            return _error(400, reason)
        if req.model not in served_names:
            return _error(404, f"model {req.model!r} is not served here; this server serves {served_model_name!r}", "not_found_error")
        try:
            prompt_ids = _prompt_ids(req.prompt, engine.tokenizer)
            max_tokens = req.max_tokens if req.max_tokens is not None else engine.config.max_model_len - len(prompt_ids)
            params = SamplingParams(
                temperature=req.temperature, top_p=req.top_p, max_tokens=max_tokens,
                stop=req.stop_strings(), ignore_eos=req.ignore_eos,
            )
            seq = Sequence(engine.new_seq_id(), prompt_ids, params, arrival_time=time.monotonic(), priority=req.priority)
            engine.validate(seq)
        except ValueError as exc:
            return _error(400, str(exc))

        request_id, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return _EngineStream(
                _stream(async_engine, seq, request_id, created, served_model_name, include_usage),
                on_disconnect=lambda: async_engine.abort_request(seq.seq_id),
                media_type="text/event-stream",
            )

        async def collect():
            text, finish_reason, num_output = [], None, 0
            async with aclosing(async_engine.generate(seq)) as outputs:
                async for out in outputs:
                    if out.error is not None:
                        return _error(500, out.error, "server_error")
                    text.append(out.text)
                    finish_reason, num_output = out.finish_reason, out.num_output_tokens
            return CompletionResponse(
                id=request_id,
                created=created,
                model=served_model_name,
                choices=[CompletionChoice(text="".join(text), finish_reason=finish_reason)],
                usage=UsageInfo(prompt_tokens=len(prompt_ids), completion_tokens=num_output, total_tokens=len(prompt_ids) + num_output),
            )

        return await _unless_disconnected(raw_request, collect(), lambda: async_engine.abort_request(seq.seq_id))

    return app


async def _unless_disconnected(request: Request, work, on_disconnect):
    """Run `work`, aborting it if the client disconnects first. Starlette watches for disconnects only
    while streaming, so without this a non-streaming request whose client left would still run to
    completion."""

    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work_task = asyncio.ensure_future(work)
    watch_task = asyncio.ensure_future(disconnected())
    done, _ = await asyncio.wait({work_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
    if work_task in done:
        watch_task.cancel()
        return work_task.result()
    on_disconnect()
    work_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await work_task
    return Response(status_code=499)  # client closed the request; nobody is listening


async def _stream(async_engine: AsyncEngine, seq: Sequence, request_id: str, created: int, model: str, include_usage: bool):
    """One SSE event per generated token, then an optional usage event, then [DONE]."""
    num_output = 0
    async with aclosing(async_engine.generate(seq)) as outputs:
        async for out in outputs:
            if out.error is not None:
                yield _sse({"error": {"message": out.error, "type": "server_error", "code": 500}})
                break
            num_output = out.num_output_tokens
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": out.text, "logprobs": None, "finish_reason": out.finish_reason}],
                }
            )
    if include_usage:
        usage = {"prompt_tokens": seq.num_prompt_tokens, "completion_tokens": num_output, "total_tokens": seq.num_prompt_tokens + num_output}
        yield _sse({"id": request_id, "object": "text_completion", "created": created, "model": model, "choices": [], "usage": usage})
    yield "data: [DONE]\n\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PagedServe OpenAI-compatible server")
    EngineConfig.add_cli_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None, help="model name clients send (default: --model)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = EngineConfig.from_cli_args(args)
    parallel = None
    if config.world_size > 1:
        from engine.distributed.parallel import init_parallel, shutdown_parallel
        from engine.distributed.worker import device_type, run_worker

        parallel = init_parallel(config.tensor_parallel_size, config.pipeline_parallel_size, device_type(config))
        if not parallel.is_driver:
            try:
                run_worker(config, parallel)
            finally:
                shutdown_parallel()
            return
    engine = LLMEngine(config, parallel=parallel)
    app = build_app(engine, args.served_model_name or args.model)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)
    finally:
        engine.shutdown()
        if parallel is not None:
            shutdown_parallel()


if __name__ == "__main__":
    main()
