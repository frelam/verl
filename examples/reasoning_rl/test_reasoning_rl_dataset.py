# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Tests for ReasoningRLDataset system-prompt injection (REASONING_RL_SYSTEM_PROMPT).

Run from the repo root:
    python -m pytest examples/reasoning_rl/test_reasoning_rl_dataset.py -q
"""

import datasets
from omegaconf import OmegaConf

from examples.reasoning_rl.reasoning_rl_dataset import ReasoningRLDataset

_ENV_VAR = "REASONING_RL_SYSTEM_PROMPT"
_SYSTEM = "Reason as carefully and thoroughly as possible before answering."


def _user_prompt(text="What is 1+1?"):
    return [{"role": "user", "content": text}]


def _write_parquet(tmp_path, prompts) -> str:
    rows = [
        {
            "data_source": "math_test",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "1"},
            "extra_info": {"split": "train", "index": i},
        }
        for i, prompt in enumerate(prompts)
    ]
    path = tmp_path / "train.parquet"
    datasets.Dataset.from_list(rows).to_parquet(str(path))
    return str(path)


def _make_config(tmp_path, **overrides):
    cfg = {
        "prompt_key": "prompt",
        "max_prompt_length": 1024,
        "truncation": "error",
        "filter_overlong_prompts": False,
        # num_proc=1 keeps the length filter single-process (fast, no pickling).
        "filter_overlong_prompts_workers": 1,
        "cache_dir": str(tmp_path / "cache"),
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


class _CharCountTokenizer:
    """apply_chat_template returns one id per content char, so the rendered
    prompt length is exactly the summed content length — lets the test pin
    max_prompt_length between the with/without-system lengths."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        return [0] * sum(len(m["content"]) for m in messages)


def test_no_injection_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    data_file = _write_parquet(tmp_path, [_user_prompt()])
    ds = ReasoningRLDataset([data_file], None, None, _make_config(tmp_path))
    assert ds.dataframe[0]["prompt"] == _user_prompt()


def test_system_prompt_prepended_to_every_row(tmp_path, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, _SYSTEM)
    data_file = _write_parquet(tmp_path, [_user_prompt(), _user_prompt("What is 2+2?")])
    ds = ReasoningRLDataset([data_file], None, None, _make_config(tmp_path))
    assert len(ds.dataframe) == 2
    for row in ds.dataframe:
        prompt = row["prompt"]
        assert prompt[0] == {"role": "system", "content": _SYSTEM}
        assert prompt[1]["role"] == "user"


def test_existing_system_message_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, _SYSTEM)
    original = [{"role": "system", "content": "original instruction"}, *_user_prompt()]
    data_file = _write_parquet(tmp_path, [original])
    ds = ReasoningRLDataset([data_file], None, None, _make_config(tmp_path))
    assert ds.dataframe[0]["prompt"] == original


def test_injection_counts_against_length_filter(tmp_path, monkeypatch):
    # Pin max_prompt_length so the row fits WITHOUT the system prompt but
    # overflows WITH it: kept when injection is off, filtered when on. This
    # proves injection happens before maybe_filter_out_long_prompts filters.
    user_len = len(_user_prompt()[0]["content"])
    cfg = _make_config(
        tmp_path,
        filter_overlong_prompts=True,
        max_prompt_length=user_len + len(_SYSTEM) - 1,
    )
    data_file = _write_parquet(tmp_path, [_user_prompt()])

    monkeypatch.delenv(_ENV_VAR, raising=False)
    ds = ReasoningRLDataset([data_file], _CharCountTokenizer(), None, cfg)
    assert len(ds.dataframe) == 1

    monkeypatch.setenv(_ENV_VAR, _SYSTEM)
    ds = ReasoningRLDataset([data_file], _CharCountTokenizer(), None, cfg)
    assert len(ds.dataframe) == 0
