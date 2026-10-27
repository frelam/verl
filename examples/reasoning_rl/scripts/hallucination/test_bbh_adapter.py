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
"""Tests for the stage-4 BBH-gap generator (HALLUCINATION_RL_DESIGN.md D32).

The generators are the only source of these rows, so the tests re-derive the
answer independently: Dyck words are re-balanced, SVG paths are re-classified by
the geometry parser, and every MCQ's gold letter is checked against its own
option block.  A mislabelled row must fail here, not in training.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbh_adapter as adapter


# ---------------------------------------------------------------------------
# per-task generators
# ---------------------------------------------------------------------------


def test_dyck_rows_are_balanced_and_unique():
    rng = random.Random(7)
    rows = [adapter.generate_dyck(rng, i) for i in range(300)]
    for generated in rows:
        assert adapter.validate_generated(adapter.TASK_DYCK, generated) is None
        prefix = generated.prompt.split("Input: ", 1)[1].split()
        suffix = generated.answer.split()
        assert suffix and all(token in adapter._DYCK_CLOSERS for token in suffix)
        assert adapter._is_balanced(prefix + suffix)
        assert not adapter._is_balanced(prefix)
        assert generated.prompt.startswith("Complete the rest of the sequence")
    # Short Dyck prefixes legitimately collide; build_task_rows is the layer that
    # turns the raw stream into a deduplicated pool.
    built, report = adapter.build_task_rows(
        adapter.TASK_DYCK, quota=300, seed=7, movielens=None, excluded=set(), seen=set()
    )
    assert len(built) == 300
    assert len({adapter.prompt_key(row["prompt"]) for row in built}) == 300
    assert report["generated"] == 300


def test_geometric_rows_classify_to_their_option():
    rng = random.Random(11)
    rows = [adapter.generate_geometric(rng, i) for i in range(300)]
    for generated in rows:
        assert adapter.validate_generated(adapter.TASK_GEOMETRIC, generated) is None
        match = adapter.re.search(r'<path d="([^"]+)"/>', generated.prompt)
        assert match is not None
        label = next(
            option["text"]
            for option in generated.options
            if option["id"] == generated.correct_option_id
        )
        assert adapter.classify_path(match.group(1)) == label
        assert len(generated.options) in (10, 11)
    labels = {
        next(option["text"] for option in g.options if option["id"] == g.correct_option_id)
        for g in rows
    }
    assert {"circle", "kite", "line", "pentagon", "sector", "triangle"} <= labels


def test_disambiguation_rows_have_three_options_and_both_gold_kinds():
    rng = random.Random(13)
    rows = [adapter.generate_disambiguation(rng, i) for i in range(300)]
    golds = set()
    for generated in rows:
        assert adapter.validate_generated(adapter.TASK_DISAMBIGUATION, generated) is None
        assert [option["id"] for option in generated.options] == ["A", "B", "C"]
        assert generated.options[-1]["text"] == "Ambiguous"
        golds.add(generated.correct_option_id)
    assert golds == {"A", "B", "C"}
    # The ten families must all be represented.
    families = {generated.difficulty for generated in rows}
    assert len(families) >= 10


def test_causal_rows_are_yes_no_and_balanced():
    rng = random.Random(17)
    rows = [adapter.generate_causal(rng, i) for i in range(300)]
    for generated in rows:
        assert adapter.validate_generated(adapter.TASK_CAUSAL, generated) is None
        assert generated.answer in {"Yes", "No"}
        assert generated.prompt.startswith("How would a typical person answer")
        assert "- Yes" in generated.prompt and "- No" in generated.prompt
    yes = sum(1 for generated in rows if generated.answer == "Yes")
    assert 100 <= yes <= 200


def _synthetic_movielens():
    """A tiny fixture with clean genre separation so the generator is testable."""
    titles = {movie_id: f"Movie {movie_id}" for movie_id in range(1, 101)}
    genres = {movie_id: {"Drama"} for movie_id in range(1, 31)}
    genres.update({movie_id: {"Comedy"} for movie_id in range(31, 61)})
    counts = {movie_id: 100 + movie_id for movie_id in range(1, 61)}
    liked = {user: list(range(1, 31)) for user in range(1, 21)}
    disliked = {user: list(range(31, 61)) for user in range(1, 21)}
    return adapter.MovieLens(titles=titles, genres=genres, counts=counts, liked=liked, disliked=disliked)


def test_movie_rows_have_four_distinct_options():
    rng = random.Random(19)
    movielens = _synthetic_movielens()
    rows = [adapter.generate_movie(rng, i, movielens) for i in range(100)]
    for generated in rows:
        assert adapter.validate_generated(adapter.TASK_MOVIE, generated) is None
        assert len(generated.options) == 4
        assert len({option["text"] for option in generated.options}) == 4
        assert generated.correct_option_id in {option["id"] for option in generated.options}
        assert generated.prompt.startswith("Find a movie similar to ")
        assert "Options:" in generated.prompt


# ---------------------------------------------------------------------------
# row shape / build plumbing
# ---------------------------------------------------------------------------


def test_make_row_carries_the_stage1_schema_and_audit_key():
    generated = adapter.generate_disambiguation(random.Random(0), 0)
    row = adapter.make_row(adapter.TASK_DISAMBIGUATION, generated, index=3, seed=44)
    assert row["data_source"] == "bbh_disambiguation_qa"
    assert row["ability"] == "bbh"
    assert row["extra_info"]["domain"] == "bbh"
    assert row["extra_info"]["bbh_task"] == adapter.TASK_DISAMBIGUATION
    assert row["extra_info"]["task_id"] == "bbh:disambiguation_qa:44:000003"
    for key in adapter.schema.EXTRA_INFO_KEYS:
        assert key in row["extra_info"]
    payload = json.loads(row["reward_model"]["ground_truth"])
    assert payload["task"] == adapter.TASK_DISAMBIGUATION
    assert payload["answer"] in {"A", "B", "C"}


def test_validate_generated_rejects_a_broken_dyck_row():
    bad = adapter.Generated(
        prompt="Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: ( ( )",
        answer="]",
    )
    assert adapter.validate_generated(adapter.TASK_DYCK, bad) is not None


def test_build_task_rows_skips_excluded_and_duplicate_prompts():
    first = adapter.generate_dyck(random.Random("bbh:44:dyck_languages"), 1)
    excluded = {adapter.prompt_key([{"role": "user", "content": first.prompt}])}
    rows, report = adapter.build_task_rows(
        adapter.TASK_DYCK,
        quota=20,
        seed=44,
        movielens=None,
        excluded=excluded,
        seen=set(),
    )
    assert rows
    assert report["generated"] == 20
    assert report["already_trained"] >= 1
    assert all(adapter.prompt_key(row["prompt"]) not in excluded for row in rows)


def test_main_writes_a_parquet_without_movie_data(tmp_path):
    out = tmp_path / "bbh.parquet"
    assert adapter.main([
        "--tasks", "dyck_languages,geometric_shapes",
        "--quota", "dyck_languages=5,geometric_shapes=5",
        "--out", str(out),
        "--raw-dir", str(tmp_path / "raw"),
    ]) == 0
    rows = adapter.schema.read_parquet_rows(str(out))
    assert len(rows) == 10
    assert {row["data_source"] for row in rows} == {"bbh_dyck", "bbh_geometric_shapes"}
    report = json.load(open(str(out).replace(".parquet", "_report.json"), encoding="utf-8"))
    assert report["summary"]["rows"] == 10
    assert report["summary"]["by_task"]["dyck_languages"] == 5


def test_load_movielens_and_build_movie_rows_from_csv(tmp_path):
    """Exercise the real CSV loader, not just the hand-built MovieLens fixture."""
    ml_dir = tmp_path / "movielens"
    ml_dir.mkdir()
    movie_lines = ["movieId,title,genres"]
    rating_lines = ["userId,movieId,rating,timestamp"]
    for movie_id in range(1, 61):
        genre = "Drama" if movie_id <= 30 else "Comedy"
        movie_lines.append(f'{movie_id},"Movie {movie_id} (2000)",{genre}')
        for user_id in range(1, 21):
            rating = 5.0 if movie_id <= 30 else 1.0
            rating_lines.append(f"{user_id},{movie_id},{rating},0")
    (ml_dir / "movies.csv").write_text("\n".join(movie_lines) + "\n", encoding="utf-8")
    (ml_dir / "ratings.csv").write_text("\n".join(rating_lines) + "\n", encoding="utf-8")

    movielens = adapter.load_movielens(str(ml_dir), download=False, min_movie_ratings=1)
    assert len(movielens.titles) == 60
    assert len(movielens.liked) >= 1
    out = tmp_path / "bbh.parquet"
    assert adapter.main([
        "--tasks", "movie_recommendation",
        "--quota", "movie_recommendation=20",
        "--out", str(out),
        "--raw-dir", str(ml_dir),
        "--movielens-dir", str(ml_dir),
        "--min-movie-ratings", "1",
        "--no-download",
    ]) == 0
    rows = adapter.schema.read_parquet_rows(str(out))
    assert len(rows) == 20
    for row in rows:
        options = row["extra_info"]["options"]
        assert len(options) == 4
        assert len({option["text"] for option in options}) == 4
        assert row["extra_info"]["correct_option_id"] in {option["id"] for option in options}
