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
"""Tests for ``filter_dataset_b.py``.

The suite pins the checks that the stage-B generation script can miss:

* a changed post-``</think>`` tail (the final-answer label) is rejected even when
  the stored ``rewritten_score`` still says 1;
* a changed or newly introduced ``\\boxed{}`` inside the think body is rejected
  (the full-response last-boxed parser cannot see it);
* a logic row whose ``extract_logic_answer`` label changes via the body is
  rejected;
* a stored row that no longer matches its A rollout is rejected;
* the reward is re-derived from the stored response, not trusted from the row;
* rows without ``original_response`` are only kept with the explicit opt-in.

The reward is faked (the real dispatcher needs the training environment), while
the format/think and answer parsers are the real reward helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
DISTILL_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, DISTILL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import filter_dataset_b  # noqa: E402
from examples.reasoning_rl.reward.compute_score import extract_logic_answer, format_ok  # noqa: E402
from examples.reasoning_rl.reward.hallucination_compute_score import extract_final  # noqa: E402

ORIGINAL = "<think>\nwe compute 6*7=\\boxed{42}\n</think>\n\n\\boxed{42}"


def fake_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Answer equality, so the tests do not need math_verify / verl."""
    if data_source.startswith("logic"):
        answer = extract_logic_answer(solution_str)
    else:
        answer = extract_final(solution_str)[1]
    return {"score": 1.0 if answer == str(ground_truth) else 0.0}


def make_row(
    response: str,
    original: str = ORIGINAL,
    *,
    sample_id: str = "task#0",
    data_source: str = "math",
    ground_truth: str = "42",
    score: float = 1.0,
    rewritten_score: float = 1.0,
    **extra,
) -> dict:
    row = {
        "sample_id": sample_id,
        "task_id": sample_id.split("#")[0],
        "prompt": [{"role": "user", "content": "what is 6*7?"}],
        "response": response,
        "original_response": original,
        "data_source": data_source,
        "ground_truth": ground_truth,
        "score": score,
        "rewritten_score": rewritten_score,
        "extra_info": {"task_id": sample_id.split("#")[0]},
    }
    row.update(extra)
    return row


def check(row: dict, **kwargs) -> tuple[str, str, dict]:
    return filter_dataset_b.check_row(
        row,
        reward_fn=fake_reward,
        format_ok=format_ok,
        extract_final=extract_final,
        extract_logic_answer=extract_logic_answer,
        **kwargs,
    )


def test_clean_row_is_kept():
    reason, detail, info = check(make_row(ORIGINAL))
    assert (reason, detail) == ("", "")
    assert info["score"] == 1.0
    assert info["tail_changed"] is False


def test_changed_tail_is_rejected_before_the_reward():
    changed = "<think>\nwe compute 6*7=\\boxed{42}\n</think>\n\n\\boxed{43}"
    reason, detail, info = check(make_row(changed, rewritten_score=1.0))
    assert reason == "tail_changed"
    assert "42" in detail and "43" in detail
    assert info["tail_changed"] is True


def test_changed_tail_requires_the_parsed_label_to_match_when_allowed():
    changed = "<think>\nwe compute 6*7=\\boxed{42}\n</think>\n\n\\boxed{43}"
    reason, _, _ = check(make_row(changed), allow_tail_change=True)
    assert reason == "label_changed"


def test_equivalent_tail_change_can_be_allowed():
    # Extra whitespace changes the tail but not the parsed label, so the opt-in keeps it.
    changed = "<think>\nwe compute 6*7=\\boxed{42}\n</think>\n\n  \\boxed{42}  "
    reason, _, _ = check(make_row(changed), allow_tail_change=True)
    assert reason == ""


def test_body_boxed_change_is_rejected_even_when_the_tail_is_identical():
    response = "<think>\nwe compute 6*7=\\boxed{43}\n</think>\n\n\\boxed{42}"
    reason, detail, _ = check(make_row(response))
    assert reason == "body_answer_changed"
    assert "42" in detail and "43" in detail


def test_earlier_body_boxed_change_is_rejected():
    # Only an earlier intermediate boxed changes; the last boxed (tail) is the
    # same, so the generation-time last-boxed check would miss it.
    original = "<think>\n\\boxed{1} then \\boxed{42}\n</think>\n\n\\boxed{42}"
    response = "<think>\n\\boxed{9} then \\boxed{42}\n</think>\n\n\\boxed{42}"
    reason, _, _ = check(make_row(response, original))
    assert reason == "body_answer_changed"


def test_body_boxed_addition_is_rejected():
    response = "<think>\n6*7 is 42; \\boxed{99}\n</think>\n\n\\boxed{42}"
    reason, _, _ = check(make_row(response, original="<think>\n6*7 is 42\n</think>\n\n\\boxed{42}"))
    assert reason == "body_answer_added"


def test_body_boxed_deletion_is_allowed():
    # Deleting a duplicated conclusion is what the compression prompt asks for.
    response = "<think>\n6*7 is 42\n</think>\n\n\\boxed{42}"
    reason, _, _ = check(make_row(response))
    assert reason == ""


def test_reward_is_recomputed_not_trusted():
    # The stored score says 1 but the response answers 43 against gt 42.
    response = "<think>\n6*7=\\boxed{43}\n</think>\n\n\\boxed{43}"
    reason, detail, info = check(make_row(response, rewritten_score=1.0, original=response))
    assert reason == "reward"
    assert info["score"] == 0.0
    assert "0.0000" in detail


def test_format_failure_is_rejected():
    reason, _, _ = check(make_row("no think block at all"))
    assert reason == "format"


def test_missing_original_is_rejected_by_default_and_opt_in_keeps_it():
    row = make_row(ORIGINAL, original="")
    assert check(row)[0] == "missing_original"
    reason, _, info = check(row, allow_missing_original=True)
    assert reason == ""
    assert info["label_unverifiable"] is True


def test_a_row_mismatch_is_rejected():
    a_row = make_row(ORIGINAL, sample_id="task#0")
    tampered = make_row(ORIGINAL, original="a different original")
    reason, _, _ = check(tampered, a_row=a_row, require_a=True)
    assert reason == "original_response_mismatch"


def test_a_row_does_not_repair_a_missing_b_original():
    # build_dpo_pairs still needs b_row["original_response"], so a missing B
    # copy is a rejection even when A could supply the text.
    reason, _, _ = check(make_row(ORIGINAL, original=""), a_row=make_row(ORIGINAL), require_a=True)
    assert reason == "missing_original"


def test_missing_a_row_is_rejected_when_a_is_configured():
    reason, _, _ = check(make_row(ORIGINAL), a_row=None, require_a=True)
    assert reason == "missing_in_a"


def test_logic_body_label_change_is_rejected_when_the_tail_has_no_boxed():
    original = "<think>\nthe answer is 42\n</think>\n\n42"
    response = "<think>\nthe answer is 7\n</think>\n\n42"
    reason, _, _ = check(make_row(response, original, data_source="logic"))
    assert reason == "label_changed"


def test_main_writes_only_the_passing_rows(tmp_path, monkeypatch):
    rows = [
        make_row(ORIGINAL, sample_id="keep#0"),
        make_row(
            "<think>\nwe compute 6*7=\\boxed{42}\n</think>\n\n\\boxed{43}",
            sample_id="bad_tail#0",
        ),
        make_row(ORIGINAL, sample_id="keep#0"),  # duplicate sample_id
    ]
    b_path = tmp_path / "dataset_b.parquet"
    pq.write_table(pa.Table.from_pylist(rows), b_path)

    written: dict[str, list[dict]] = {}

    def fake_writer(rows_to_write, path):
        written[str(path)] = list(rows_to_write)

    monkeypatch.setattr(filter_dataset_b, "load_reward", lambda: fake_reward)
    monkeypatch.setattr(filter_dataset_b, "load_verifier_helpers", lambda: (format_ok, extract_final))
    monkeypatch.setattr(filter_dataset_b, "write_parquet_rows", fake_writer)

    output_dir = tmp_path / "out"
    rc = filter_dataset_b.main(
        ["--b", str(b_path), "--output_dir", str(output_dir), "--score_workers", "1", "--log_every", "0"]
    )
    assert rc == 0
    kept = [row["sample_id"] for row in written[str(output_dir / "dataset_b_filtered.parquet")]]
    rejected = [row["sample_id"] for row in written[str(output_dir / "dataset_b_rejected.parquet")]]
    assert kept == ["keep#0"]
    assert sorted(rejected) == ["bad_tail#0", "keep#0"]  # duplicate + changed tail

    stats = json.loads((output_dir / "b_filter_stats.json").read_text(encoding="utf-8"))
    assert stats["rows"] == 3
    assert stats["kept"] == 1
    assert stats["drop_reasons"]["tail_changed"] == 1
    assert stats["drop_reasons"]["duplicate_sample_id"] == 1
