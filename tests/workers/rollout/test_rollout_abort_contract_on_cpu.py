# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only contracts for vLLM/SGLang abort and migration adapter behavior.

The fake engines exercise verl's real server methods without initializing a model, Ray actor, or GPU.
They do not validate either backend's actual GPU streaming, cache, or logprob implementation.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


class _Config(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _RequestData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SamplingParams(_RequestData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_logprobs = kwargs.get("prompt_logprobs")
        self.logprobs = kwargs.get("logprobs")


class _TraceState:
    @staticmethod
    def trace_state(*args, **kwargs):
        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Context()


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_modules(monkeypatch: pytest.MonkeyPatch, definitions: dict[str, ModuleType]) -> None:
    for name, module in definitions.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, attribute = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                monkeypatch.setattr(parent, attribute, module, raising=False)


def _install_shared_verl_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Initialize verl's package exports before temporarily replacing optional engine-facing helpers.
    importlib.import_module("verl.workers.rollout.replica")
    _install_modules(
        monkeypatch,
        {
            "verl.utils.profiler": _module(
                "verl.utils.profiler",
                build_rollout_dist_profiler=lambda *args, **kwargs: None,
                build_vllm_profiler_args=lambda *args, **kwargs: {},
                build_sglang_profiler_args=lambda *args, **kwargs: {},
                relocate_rollout_traces=lambda *args, **kwargs: None,
                rollout_profiler_global_ranks=lambda *args, **kwargs: [],
            ),
            "verl.utils.tokenizer": _module("verl.utils.tokenizer", normalize_token_ids=list),
            "verl.workers.rollout.utils": _module(
                "verl.workers.rollout.utils",
                get_max_position_embeddings=lambda config: config.max_position_embeddings,
                get_vision_placeholder_token_ids=lambda processor: [],
                qwen2_5_vl_dedup_image_tokens=lambda prompt_ids, processor: prompt_ids,
                run_uvicorn=AsyncMock(),
            ),
        },
    )


@pytest.fixture
def vllm_server_module(monkeypatch: pytest.MonkeyPatch):
    _install_shared_verl_stubs(monkeypatch)
    _install_modules(
        monkeypatch,
        {
            "vllm": _module("vllm", __version__="0.11.0", SamplingParams=_SamplingParams),
            "vllm.entrypoints": _module("vllm.entrypoints"),
            "vllm.entrypoints.cli": _module("vllm.entrypoints.cli"),
            "vllm.entrypoints.cli.serve": _module(
                "vllm.entrypoints.cli.serve", run_headless=lambda *args: None, cmd_init=lambda: []
            ),
            "vllm.engine": _module("vllm.engine"),
            "vllm.engine.arg_utils": _module("vllm.engine.arg_utils", AsyncEngineArgs=_RequestData),
            "vllm.entrypoints.openai": _module("vllm.entrypoints.openai"),
            "vllm.entrypoints.openai.api_server": _module(
                "vllm.entrypoints.openai.api_server", build_app=lambda *args: None, init_app_state=AsyncMock()
            ),
            "vllm.entrypoints.openai.parser": _module("vllm.entrypoints.openai.parser"),
            "vllm.entrypoints.openai.parser.harmony_utils": _module(
                "vllm.entrypoints.openai.parser.harmony_utils", get_encoding=lambda: None
            ),
            "vllm.inputs": _module("vllm.inputs", TokensPrompt=_RequestData),
            "vllm.lora": _module("vllm.lora"),
            "vllm.lora.request": _module("vllm.lora.request", LoRARequest=_RequestData),
            "vllm.outputs": _module("vllm.outputs", RequestOutput=_RequestData),
            "vllm.usage": _module("vllm.usage"),
            "vllm.usage.usage_lib": _module(
                "vllm.usage.usage_lib", UsageContext=SimpleNamespace(OPENAI_API_SERVER="server")
            ),
            "vllm.utils": _module("vllm.utils"),
            "vllm.utils.argparse_utils": _module("vllm.utils.argparse_utils", FlexibleArgumentParser=_RequestData),
            "vllm.v1": _module("vllm.v1"),
            "vllm.v1.engine": _module("vllm.v1.engine", FinishReason=SimpleNamespace(ABORT="abort")),
            "vllm.v1.engine.async_llm": _module("vllm.v1.engine.async_llm", AsyncLLM=_RequestData),
            "verl.utils.vllm.vllm_quant_utils": _module(
                "verl.utils.vllm.vllm_quant_utils", apply_vllm_quant_patches=lambda: None
            ),
            "verl.workers.rollout.vllm_rollout": _module(
                "verl.workers.rollout.vllm_rollout",
                __path__=[str(Path(__file__).parents[3] / "verl" / "workers" / "rollout" / "vllm_rollout")],
            ),
            "verl.workers.rollout.vllm_rollout.utils": _module(
                "verl.workers.rollout.vllm_rollout.utils",
                VLLM_LORA_INT_ID=123,
                VLLM_LORA_NAME="123",
                VLLM_LORA_PATH="unused",
                SuppressSignalInThread=_RequestData,
                build_cli_args_from_config=lambda config: [],
                build_mtp_speculative_config=lambda *args: {},
                extract_prompt_logprobs=lambda **kwargs: None,
                get_vllm_max_lora_rank=lambda rank: rank,
            ),
        },
    )
    name = "verl.workers.rollout.vllm_rollout.vllm_async_server"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "RLInsightLogger", _TraceState)
    return module


@pytest.fixture
def sglang_server_module(monkeypatch: pytest.MonkeyPatch):
    _install_shared_verl_stubs(monkeypatch)
    _install_modules(
        monkeypatch,
        {
            "sglang": _module("sglang", __version__="0.5.12"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.entrypoints": _module("sglang.srt.entrypoints"),
            "sglang.srt.entrypoints.engine": _module("sglang.srt.entrypoints.engine"),
            "sglang.srt.entrypoints.http_server": _module(
                "sglang.srt.entrypoints.http_server",
                ServerArgs=_RequestData,
                _GlobalState=_RequestData,
                app=SimpleNamespace(),
                set_global_state=lambda state: None,
            ),
            "sglang.srt.managers": _module("sglang.srt.managers"),
            "sglang.srt.managers.io_struct": _module(
                "sglang.srt.managers.io_struct",
                ContinueGenerationReqInput=_RequestData,
                GenerateReqInput=_RequestData,
                PauseGenerationReqInput=_RequestData,
                ReleaseMemoryOccupationReqInput=_RequestData,
                ResumeMemoryOccupationReqInput=_RequestData,
            ),
            "sglang.srt.managers.tokenizer_manager": _module(
                "sglang.srt.managers.tokenizer_manager", ServerStatus=SimpleNamespace(Up="up")
            ),
            "verl.workers.rollout.sglang_rollout.sglang_rollout": _module(
                "verl.workers.rollout.sglang_rollout.sglang_rollout", _set_envs_and_config=lambda args: None
            ),
            "verl.workers.rollout.sglang_rollout.utils": _module(
                "verl.workers.rollout.sglang_rollout.utils",
                SGLANG_LORA_NAME="lora",
                lora_rank_of=lambda config: config.lora_rank,
                lora_served_as_adapter=lambda config: False,
                sglang_lora_target_modules=lambda modules: modules,
            ),
        },
    )
    name = "verl.workers.rollout.sglang_rollout.async_sglang_server"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "RLInsightLogger", _TraceState)
    return module


def _server_config() -> _Config:
    return _Config(
        max_model_len=64,
        response_length=16,
        prompt_length=8,
        full_determinism=False,
        enable_rollout_routing_replay=False,
        mtp=None,
        skip_tokenizer_init=True,
    )


def _vllm_frame(token_ids, finish_reason=None, logprob_values=None):
    logprobs = None
    if logprob_values is not None:
        logprobs = [
            {token_id: SimpleNamespace(logprob=logprob)}
            for token_id, logprob in zip(token_ids, logprob_values, strict=False)
        ]
    completion = SimpleNamespace(
        token_ids=token_ids,
        finish_reason=finish_reason,
        logprobs=logprobs,
        routed_experts=None,
        num_preempted=0,
    )
    return SimpleNamespace(outputs=[completion], num_cached_tokens=2, kv_transfer_params=None)


def _make_vllm_server(module, frames=()):
    server = object.__new__(module.vLLMHttpServer)
    server.config = _server_config()
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server.replica_rank = 0
    server.node_rank = 0
    server.global_steps = 3
    server._abort_reasons = {}
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()

    def generate(**kwargs):
        async def stream():
            for frame in frames:
                yield frame

        return stream()

    server.engine = SimpleNamespace(generate=generate)
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_reason", ["weight_sync", "migration"])
@pytest.mark.parametrize("terminal_kind", ["no_outputs", "no_tokens"])
async def test_vllm_empty_abort_keeps_cumulative_prefix_and_reason(vllm_server_module, abort_reason, terminal_kind):
    terminal = (
        SimpleNamespace(outputs=[], num_cached_tokens=None)
        if terminal_kind == "no_outputs"
        else _vllm_frame([], finish_reason="abort")
    )
    frames = [
        _vllm_frame([11], logprob_values=[-0.1]),
        _vllm_frame([11, 12], logprob_values=[-0.1, -0.2]),
        terminal,
    ]
    server = _make_vllm_server(vllm_server_module, frames)
    if abort_reason == "migration":
        server._abort_reasons["req"] = abort_reason

    output = await server.generate([1, 2], {"max_tokens": 4, "logprobs": True}, "req")

    assert output.token_ids == [11, 12]
    assert output.log_probs == [-0.1, -0.2]
    assert output.stop_reason == "aborted"
    assert output.extra_fields["abort_reason"] == abort_reason
    assert "req" not in server._abort_reasons


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "reset_prefix_cache", "expected_cache_resets"),
    [("migration", False, 0), ("weight_sync", True, 1)],
)
async def test_vllm_abort_records_reason_and_honors_cache_policy(
    vllm_server_module, reason, reset_prefix_cache, expected_cache_resets
):
    server = _make_vllm_server(vllm_server_module)
    request_state = SimpleNamespace(
        queue=SimpleNamespace(put=Mock()),
        make_request_output=Mock(return_value="empty-abort"),
    )
    server.engine.output_processor = SimpleNamespace(request_states={"req": request_state}, abort_requests=Mock())
    server.engine.engine_core = SimpleNamespace(abort_requests_async=AsyncMock())
    server.clear_kv_cache = AsyncMock()

    result = await server.abort_request("req", reset_prefix_cache=reset_prefix_cache, reason=reason)

    assert result == {"aborted": True, "request_id": "req", "abort_reason": reason}
    assert server._abort_reasons["req"] == reason
    assert server.clear_kv_cache.await_count == expected_cache_resets


@pytest.mark.asyncio
async def test_vllm_replica_migration_defaults_to_preserving_prefix_cache(vllm_server_module):
    remote_abort = AsyncMock(return_value={"aborted": True, "abort_reason": "migration"})
    replica = object.__new__(vllm_server_module.vLLMReplica)
    replica.servers = [SimpleNamespace(abort_request=SimpleNamespace(remote=remote_abort))]

    result = await replica.abort_request("req")

    assert result["aborted"] is True
    remote_abort.assert_awaited_once_with("req", reset_prefix_cache=False, reason="migration")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logprob_values", "error"),
    [(None, "missing requested response logprobs"), ([-0.1], "2 tokens but 1 logprob entries")],
    ids=["missing", "misaligned"],
)
async def test_vllm_requested_logprobs_fail_explicitly(vllm_server_module, logprob_values, error):
    server = _make_vllm_server(
        vllm_server_module,
        [_vllm_frame([11, 12], finish_reason="length", logprob_values=logprob_values)],
    )

    with pytest.raises(RuntimeError, match=error):
        await server.generate([1], {"max_tokens": 2, "logprobs": True}, "req")


class _SGLangTokenizerManager:
    def __init__(self, output):
        self.output = output
        self.abort_request = AsyncMock()
        self.flush_cache = AsyncMock()

    def generate_request(self, request, _):
        async def stream():
            yield self.output

        return stream()


def _sglang_output(logprobs=((-0.1, 11, None), (-0.2, 12, None)), finish_reason="abort"):
    return {
        "output_ids": [11, 12],
        "meta_info": {
            "finish_reason": {"type": finish_reason},
            "output_token_logprobs": logprobs,
        },
    }


def _make_sglang_server(module, output):
    server = object.__new__(module.SGLangHttpServer)
    server.config = _server_config()
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server.replica_rank = 0
    server.node_rank = 0
    server.global_steps = 4
    server._abort_reasons = {}
    server.tokenizer_manager = _SGLangTokenizerManager(output)
    return server


@pytest.mark.asyncio
async def test_sglang_migration_preserves_partial_output_reason_and_prefix_cache(sglang_server_module):
    server = _make_sglang_server(sglang_server_module, _sglang_output())

    aborted = await server.abort_request("req")
    output = await server.generate([1], {"max_new_tokens": 2, "logprobs": True}, "req")

    assert aborted == {"aborted": True, "request_id": "req", "abort_reason": "migration"}
    server.tokenizer_manager.flush_cache.assert_not_awaited()
    assert output.token_ids == [11, 12]
    assert output.log_probs == [-0.1, -0.2]
    assert output.stop_reason == "abort"
    assert output.extra_fields["abort_reason"] == "migration"


@pytest.mark.asyncio
async def test_sglang_untagged_abort_is_weight_sync(sglang_server_module):
    server = _make_sglang_server(sglang_server_module, _sglang_output())

    output = await server.generate([1], {"max_new_tokens": 2, "logprobs": True}, "req")

    assert output.token_ids == [11, 12]
    assert output.extra_fields["abort_reason"] == "weight_sync"


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [None, [(-0.1, 11, None)]], ids=["missing", "misaligned"])
async def test_sglang_migration_with_invalid_logprobs_fails_explicitly(sglang_server_module, logprobs):
    server = _make_sglang_server(sglang_server_module, _sglang_output(logprobs=logprobs))
    server._abort_reasons["req"] = "migration"

    with pytest.raises(RuntimeError, match="lost token/logprob alignment"):
        await server.generate([1], {"max_new_tokens": 2, "logprobs": True}, "req")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logprobs",
    [None, [(-0.1, 11, None)]],
    ids=["missing", "misaligned"],
)
async def test_sglang_requested_logprobs_fail_explicitly(sglang_server_module, logprobs):
    server = _make_sglang_server(
        sglang_server_module,
        _sglang_output(logprobs=logprobs, finish_reason="length"),
    )

    with pytest.raises(RuntimeError, match="logprob"):
        await server.generate([1], {"max_new_tokens": 2, "logprobs": True}, "req")
