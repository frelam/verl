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
"""Tests for the BBH-gap reward (HALLUCINATION_RL_DESIGN.md D32)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbh_verifier import verify_bbh
from compute_score import compute_score


def _solution(body: str, reasoning: str = "the reasoning") -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{body}"


def _gt(task: str, answer: str) -> str:
    return json.dumps({"task": task, "answer": answer})


# ---------------------------------------------------------------------------
# Dyck
# ---------------------------------------------------------------------------


def test_dyck_accepts_the_exact_closing_suffix():
    gt = _gt("dyck_languages", ") ]")
    assert verify_bbh(_solution(") ]"), gt, "bbh_dyck") == 1.0
    assert verify_bbh(_solution("\\boxed{) ]}"), gt, "bbh_dyck") == 1.0


def test_dyck_compares_tokens_not_whitespace():
    gt = _gt("dyck_languages", ") ]")
    assert verify_bbh(_solution(")]"), gt, "bbh_dyck") == 1.0
    assert verify_bbh(_solution("] )"), gt, "bbh_dyck") == 0.0
    assert verify_bbh(_solution("The missing brackets are: ) ]."), gt, "bbh_dyck") == 1.0


def test_dyck_rejects_the_full_balanced_sequence():
    gt = _gt("dyck_languages", ") ]")
    assert verify_bbh(_solution("( [ ] )"), gt, "bbh_dyck") == 0.0


# ---------------------------------------------------------------------------
# multiple choice
# ---------------------------------------------------------------------------


def test_mcq_accepts_boxed_bare_and_answer_line():
    gt = _gt("geometric_shapes", "B")
    assert verify_bbh(_solution("\\boxed{(B)}"), gt, "bbh_geometric_shapes") == 1.0
    assert verify_bbh(_solution("\\boxed{B}"), gt, "bbh_geometric_shapes") == 1.0
    assert verify_bbh(_solution("(B)"), gt, "bbh_geometric_shapes") == 1.0
    assert verify_bbh(_solution("The answer is (B)."), gt, "bbh_geometric_shapes") == 1.0


def test_mcq_rejects_the_wrong_letter():
    gt = _gt("disambiguation_qa", "C")
    assert verify_bbh(_solution("(A)"), gt, "bbh_disambiguation_qa") == 0.0
    assert verify_bbh(_solution("(C)"), gt, "bbh_disambiguation_qa") == 1.0


def test_movie_uses_the_same_letter_contract():
    gt = _gt("movie_recommendation", "D")
    assert verify_bbh(_solution("\\boxed{D}"), gt, "bbh_movie_recommendation") == 1.0
    assert verify_bbh(_solution("\\boxed{C}"), gt, "bbh_movie_recommendation") == 0.0


# ---------------------------------------------------------------------------
# yes / no
# ---------------------------------------------------------------------------


def test_causal_accepts_yes_no_in_any_case():
    gt = _gt("causal_judgement", "Yes")
    assert verify_bbh(_solution("Yes."), gt, "bbh_causal_judgement") == 1.0
    assert verify_bbh(_solution("\\boxed{yes}"), gt, "bbh_causal_judgement") == 1.0
    assert verify_bbh(_solution("No."), gt, "bbh_causal_judgement") == 0.0


# ---------------------------------------------------------------------------
# fail-closed and dispatch
# ---------------------------------------------------------------------------


def test_malformed_ground_truth_scores_zero():
    assert verify_bbh(_solution("Yes"), "not json", "bbh_causal_judgement") == 0.0
    assert verify_bbh(_solution("Yes"), json.dumps({"task": "causal_judgement"}), "bbh_causal_judgement") == 0.0
    assert verify_bbh(_solution("Yes"), json.dumps({"answer": "Yes"}), "unknown_source") == 0.0


def test_dispatcher_routes_bbh_prefix_and_keeps_the_format_gate():
    gt = _gt("dyck_languages", ") ]")
    assert compute_score("bbh_dyck", _solution(") ]"), gt)["score"] == 1.0
    # No think block -> the base format gate zeroes the row before the verifier.
    assert compute_score("bbh_dyck", ") ]", gt)["score"] == 0.0
