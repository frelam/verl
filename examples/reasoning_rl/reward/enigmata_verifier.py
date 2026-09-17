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
"""Task-aware verifier for the Enigmata puzzle domain (``logic_enigmata``).

Why this exists
---------------
``to_parquet_logic.py`` stores the raw Enigmata ``answer`` field verbatim.  For
many tasks that field is *not* the string the model is asked to produce::

    game24            "The answer is: (12-10)*(12*1) = 24"   (expression, many solutions)
    countdown         "The answer is: 6*11-(15+13)/4 = 59"   (expression, many solutions)
    maze              "The answer is: (1,1)->(1,2)->..."     (prose prefix)
    stack_permutation "The output sequence is a valid stack permutation."  (sentence, not ops)
    eight_puzzle      "[[6, 3, 1], [2, 0, 7], [5, 4, 8]]"    (the *initial* board; the
                                                              answer is a move sequence)
    twiddle           "[[1, 0], [0, 0]]"                     (one of many valid rotations)
    hamiltonian_path  "[24, 21, 23, ...]"                    (one of many valid paths)
    car_painting      "[1, 2, 5, 3, 9, 7, 8, 4, 6, 10]"      (one of many optimal orders)
    full_crosswords   '{"across": [...], "down": [...]}'      (prompt mandates "across: ..., down: ...")

Comparing those strings literally against the model's answer can never succeed
(or rejects every correct answer that is not byte-identical), so a correct
response is scored 0.  Verified against the real pool: the four sliding/shift
puzzles (8/15/nine/sixteen puzzle, 24k rows) store the *initial* board as the
answer, so string comparison could never award a single point; hitori /
kakurasu / light_up / minesweeper compare coordinate *sets*; twiddle,
hamiltonian and car_painting have many valid solutions; campsite stores the
constraint header in front of the board, which the prompt never asks the model
to repeat.

This module mirrors the official per-task verifiers shipped in
``BytedTsinghua-SIA/Enigmata`` (``verifiable_tasks/tasks/<task>/verifier.py``)
adapted to the reasoning_rl reward contract:

* the model answer is already extracted by ``extract_logic_answer`` (``<answer>``
  tags -> ``\\boxed{}`` -> final-answer line -> last fenced block), so the
  per-task logic operates on that candidate;
* expression tasks (game24/countdown) are evaluated arithmetically and, when
  ``meta`` carries the input numbers, checked to use exactly those numbers --
  this is what stops the policy from hacking the reward with ``24``;
* puzzle tasks are *simulated* (sliding puzzles, circular shifts, twiddle
  rotations, hamiltonian paths, car reordering) or checked against their
  prompt-mandated container (crosswords, campsite/star_battle boards,
  zebra tables), exactly as the official verifier does, so any valid solution
  earns the point rather than only the generator's own;
* maze/stack_permutation and every task added above need the original puzzle
  data, which ``to_parquet_logic.py`` stores under ``"meta"`` in
  ``ground_truth``.  Rows built before that change (no ``meta``) still get the
  safe fallbacks documented per function below.

``verify_enigmata`` returns ``True``/``False`` when it owns the task, and
``None`` when the task is not handled here (caller falls back to the generic
matcher).
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import operator
import re

logger = logging.getLogger(__name__)

# Natural-language scaffolding that Enigmata puts in front of the real answer,
# e.g. "The answer is: ...", "The final answer is: ...", "Answer: ...".
_PROSE_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[::]?\s*",
    re.IGNORECASE,
)

_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_NUMBER_RE = re.compile(r"\d+")
# String/grid tasks (campsite, star_battle, ...) ask for the board wrapped in
# <begin_board>...</begin_board> inside the final answer.
_BOARD_RE = re.compile(r"<begin_board>(.*?)<end_board>", re.DOTALL | re.IGNORECASE)

# Tasks whose ground truth needs semantic verification instead of string match.
HANDLED_TASKS = frozenset(
    {
        # arithmetic: the stored answer is one of many valid expressions
        "game24",
        "countdown",
        # path / simulation tasks whose answer is prose or one of many solutions
        "maze",
        "stack_permutation",
        "eight_puzzle",
        "fifteen_puzzle",
        "nine_puzzle",
        "sixteen_puzzle",
        "twiddle",
        "hamiltonian_path",
        "hamiltonian_cycle",
        "car_painting",
        # tasks whose prompt mandates a container the ground truth does not use
        "campsite",
        "star_battle",
        "full_crosswords",
        "tic_tac_toe",
        "zebra_logic",
    }
)

_ARITH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# Guard against a degenerate ``9**9**9``-style rollout hanging the driver.
_MAX_POW_EXPONENT = 64
# ... and against one that keeps the exponents legal but grows the *base*:
# ``((2**64)**64)**64`` adds 64 bits per nesting, so a handful of them asks for
# gigabytes. Real game24/countdown arithmetic never leaves the small-int range,
# so cap every intermediate integer at ~1233 decimal digits.
_MAX_INT_BITS = 4096


def strip_prose_prefix(text: str) -> str:
    """Drop a leading "The answer is:"-style preamble from ``text``."""
    if not text:
        return text
    return _PROSE_PREFIX_RE.sub("", text, count=1).strip()


# ---------------------------------------------------------------------------
# metadata / answer-shape helpers
# ---------------------------------------------------------------------------


def _decode(value):
    """Decode a metadata field that may still be a JSON or Python literal string.

    The Enigmata jsonl wraps ``meta`` in a JSON *string* and several tasks
    double-encode their values (``"question": "[6, 11, 15, 4, 13]"``), so a field
    the verifier needs as a list can arrive as text.  Plain text that is not a
    literal (a maze grid, a graph, a table) is returned unchanged.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (ValueError, TypeError, RecursionError):
        # ValueError covers JSONDecodeError *and* the plain ValueError CPython
        # raises for an integer literal past ``sys.get_int_max_str_digits()``;
        # RecursionError covers a pathologically nested response. Neither is a
        # usable metadata literal, so fall through to literal_eval/the raw text.
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return value


def _as_list(value) -> list | None:
    value = _decode(value)
    if isinstance(value, list | tuple):
        return list(value)
    return None


def _as_int(value) -> int | None:
    value = _decode(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return int(value)
        except (ValueError, OverflowError):
            # ``json``/``ast`` decode nan and 1e999 to a non-finite float, which
            # is not a usable integer field; treat it as missing.
            return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _parse_int_matrix(value) -> list[list[int]] | None:
    """Parse an integer grid from any answer shape Enigmata uses.

    Accepts a nested list, ``"[[1 2,3 4]]"`` (cells space separated, rows comma
    separated -- the SynLogic/Enigmata prompt convention), and a plain
    whitespace/newline separated grid.  Returns ``None`` when the text is not an
    integer grid, so callers never mistake prose for a board.
    """
    decoded = _decode(value)
    if isinstance(decoded, list | tuple) and decoded and all(isinstance(r, list | tuple) for r in decoded):
        try:
            return [[int(c) for c in row] for row in decoded]
        except (TypeError, ValueError, OverflowError):
            # Non-finite floats (1e999 / Infinity) are not grid cells.
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("[[") and text.endswith("]]") and len(text) > 4 and "," in text:
        try:
            return [[int(c) for c in row.split()] for row in text[2:-2].split(",") if row.strip()]
        except ValueError:
            return None
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if not rows:
        return None
    grid: list[list[int]] = []
    for row in rows:
        cells = [c for c in re.split(r"[,\s]+", row) if c]
        try:
            grid.append([int(c) for c in cells])
        except ValueError:
            return None
    return grid or None


def _int_list(value) -> list[int] | None:
    """Parse a flat integer list (``"[1, 2, 3]"``, ``"1 2 3"``, ``"(1,2,3)"``)."""
    decoded = _decode(value)
    if isinstance(decoded, list | tuple) and all(
        isinstance(x, int | float) and not isinstance(x, bool) for x in decoded
    ):
        try:
            return [int(x) for x in decoded]
        except (ValueError, OverflowError):
            # Non-finite floats are not list entries either.
            return None
    if not isinstance(value, str):
        return None
    match = re.search(r"\[([\d\s,]+)\]", value)
    text = match.group(1) if match else value
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if not tokens:
        return None
    try:
        return [int(t) for t in tokens]
    except ValueError:
        return None


_NO_SOLUTION_RE = re.compile(r"no\s+(?:feasible|valid|solution)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# sliding / shift puzzles (8, 15, nine, sixteen puzzle)
# ---------------------------------------------------------------------------

# The official verifier's offsets: the *empty* cell moves by the offset, i.e.
# "L" swaps the blank with its left neighbour.
_SLIDING_MOVES = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}


def _sliding_board_ok(board: list[list[int]]) -> bool:
    """True for a square sliding board holding exactly the tiles 1..n^2-1 and one blank.

    A malformed ``meta.question`` (ragged rows, or no blank at all) would
    otherwise make the parity test / blank lookup below fail on ``next()`` and
    take the rollout with it.
    """
    size = len(board)
    if size < 2 or any(len(row) != size for row in board):
        return False
    return sorted(cell for row in board for cell in row) == list(range(size * size))


def _sliding_solvable(board: list[list[int]]) -> bool:
    """Classic 15-puzzle parity test (goal = 1..n^2-1 followed by the blank)."""
    size = len(board)
    flat = [cell for row in board for cell in row]
    inversions = sum(
        1 for i in range(len(flat)) for j in range(i + 1, len(flat)) if flat[i] and flat[j] and flat[i] > flat[j]
    )
    if size % 2:
        return inversions % 2 == 0
    blank_row = next((i for i, row in enumerate(board) if 0 in row), None)
    if blank_row is None:
        return False
    blank_row_from_bottom = size - blank_row
    return (inversions + blank_row_from_bottom) % 2 == 1


def _sliding_moves(text: str) -> list[str] | None:
    """Extract a move string from a response, or ``None`` when it is not one.

    The official verifier takes the last code block verbatim as the sequence, so
    a response that mixes prose with the moves is rejected.  As a fallback the
    last whitespace/comma separated token is accepted when it consists purely of
    move letters (``"The sequence is LRURDL."``), which is the same answer.
    """
    compact = re.sub(r"[^A-Za-z]", "", text).upper()
    if compact and not set(compact) - set(_SLIDING_MOVES):
        return list(compact)
    for token in re.split(r"[\s,;]+", text.strip()):
        cleaned = re.sub(r"[^A-Za-z]", "", token).upper()
        if cleaned and not set(cleaned) - set(_SLIDING_MOVES):
            return list(cleaned)
    return None


def _apply_sliding(board: list[list[int]], moves: list[str], blank_moves: bool) -> list[list[int]] | None:
    """Replay a move sequence and return the final board (``None`` entry = stuck)."""
    size = len(board)
    grid = [row[:] for row in board]
    empty = next(((i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == 0), None)
    if empty is None:  # no blank: not a sliding board
        return None
    for move in moves:
        d_row, d_col = _SLIDING_MOVES[move]
        if not blank_moves:
            # The prompt says the *tile* moves ("move a tile adjacent to the blank
            # into the blank"); the official verifier replays the inverse reading
            # (the blank moves).  Both are the same puzzle, so both are accepted.
            d_row, d_col = -d_row, -d_col
        row, col = empty[0] + d_row, empty[1] + d_col
        if not (0 <= row < size and 0 <= col < size):
            return None
        grid[empty[0]][empty[1]], grid[row][col] = grid[row][col], grid[empty[0]][empty[1]]
        empty = (row, col)
    return grid


def _verify_sliding_puzzle(prediction: str | None, meta) -> bool:
    """8/15 puzzle: replay the move sequence from the stored initial board."""
    board = _parse_int_matrix((meta or {}).get("question"))
    if not board or not _sliding_board_ok(board):
        return False
    if prediction is None:
        return False
    size = len(board)
    if not _sliding_solvable(board):
        # Provably unsolvable: the only correct response states that.
        return bool(_NO_SOLUTION_RE.search(prediction))
    if _NO_SOLUTION_RE.search(prediction):
        return False
    moves = _sliding_moves(prediction)
    if not moves:
        return False
    goal = [[(i * size + j + 1) % (size * size) for j in range(size)] for i in range(size)]
    return any(_apply_sliding(board, moves, blank_moves) == goal for blank_moves in (True, False))


_SHIFT_MOVE_RE = re.compile(r"([RC])\s*(\d)\s*(\d)")


def _apply_shift(board: list[list[int]], moves: list[tuple[str, int, int]], left: bool) -> list[int]:
    """Replay circular row/column shifts and return the flattened board."""
    size = len(board)
    state = [cell for row in board for cell in row]
    for kind, index, steps in moves:
        steps %= size
        if not (0 <= index < size):
            return []
        if kind == "R":
            start = index * size
            row = state[start : start + size]
            state[start : start + size] = (
                row[steps:] + row[:steps] if left else row[size - steps :] + row[: size - steps]
            )
        else:
            column = state[index::size]
            state[index::size] = (
                column[steps:] + column[:steps] if left else column[size - steps :] + column[: size - steps]
            )
    return state


def _verify_shift_puzzle(prediction: str | None, meta) -> bool:
    """Nine/sixteen puzzle: replay ``["R11", "C23"]`` circular row/column shifts."""
    board = _parse_int_matrix((meta or {}).get("question"))
    if not board or len(board) != len(board[0]):
        return False
    if prediction is None or _NO_SOLUTION_RE.search(prediction):
        # The generator only emits solvable instances, so this response is wrong.
        return False
    matches = _SHIFT_MOVE_RE.findall(prediction.upper())
    if not matches:
        return False
    moves = [(kind, int(index) - 1, int(steps)) for kind, index, steps in matches]
    size = len(board)
    goal = list(range(1, size * size + 1))
    # The prompt does not say which way a row/column rotates, so a sequence that
    # solves the puzzle under either reading is accepted.
    return any(_apply_shift(board, moves, left) == goal for left in (True, False))


# ---------------------------------------------------------------------------
# twiddle
# ---------------------------------------------------------------------------

_TWIDDLE_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_TWIDDLE_GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def _verify_twiddle(prediction: str | None, meta) -> bool:
    """Replay 2x2 counter-clockwise rotations; any solving sequence is accepted."""
    board = _parse_int_matrix((meta or {}).get("question"))
    if not board or len(board) != 3 or any(len(row) != 3 for row in board):
        return False
    if prediction is None:
        return False
    decoded = _decode(prediction)
    rotations: list[tuple[int, int]] = []
    if (
        isinstance(decoded, list | tuple)
        and decoded
        and all(isinstance(pair, list | tuple) and len(pair) == 2 for pair in decoded)
    ):
        try:
            rotations = [(int(pair[0]), int(pair[1])) for pair in decoded]
        except (TypeError, ValueError):
            rotations = []
    if not rotations:
        try:
            rotations = [(int(a), int(b)) for a, b in _TWIDDLE_PAIR_RE.findall(prediction)]
        except ValueError:
            # ``\d+`` can span thousands of digits, which CPython refuses to
            # convert (``sys.get_int_max_str_digits()``); that is not a rotation.
            return False
    if not rotations:
        return False
    grid = [row[:] for row in board]
    for i, j in rotations:
        if i not in (0, 1) or j not in (0, 1):
            return False
        grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j] = (
            grid[i][j + 1],
            grid[i + 1][j + 1],
            grid[i + 1][j],
            grid[i][j],
        )
    return grid == _TWIDDLE_GOAL


# ---------------------------------------------------------------------------
# hamiltonian path / cycle
# ---------------------------------------------------------------------------


def _parse_graph(question) -> tuple[int, set[tuple[int, int]]] | None:
    """Parse the ``"<num_nodes>\\n<u> <v>\\n..."`` graph stored in ``meta``."""
    if not isinstance(question, str):
        return None
    lines = [line.strip() for line in question.strip().splitlines() if line.strip()]
    if not lines:
        return None
    try:
        num_nodes = int(lines[0])
    except ValueError:
        return None
    edges: set[tuple[int, int]] = set()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        edges.add((min(u, v), max(u, v)))
    return num_nodes, edges


def _verify_hamiltonian(prediction: str | None, answer, task: str, meta) -> bool:
    """Validate a path/cycle against the graph: every valid one earns the point."""
    parsed = _parse_graph((meta or {}).get("question"))
    if parsed is None:
        return False
    num_nodes, edges = parsed
    answer_text = str(answer).strip()
    answer_is_no = answer_text.upper().startswith("NO")
    if prediction is None:
        return False
    if re.search(r"\bno\b", prediction, re.IGNORECASE) and not re.search(r"\d", prediction):
        return answer_is_no
    sequence = _int_list(prediction)
    if not sequence:
        return False
    if task == "hamiltonian_path":
        if len(sequence) != num_nodes or sorted(sequence) != list(range(num_nodes)):
            return False
        return all(
            (min(sequence[i], sequence[i + 1]), max(sequence[i], sequence[i + 1])) in edges
            for i in range(num_nodes - 1)
        )
    # A cycle may or may not repeat the starting node; the official verifier
    # strips the repeat, so both spellings are the same answer.
    if len(sequence) == num_nodes + 1 and sequence[0] == sequence[-1]:
        sequence = sequence[:-1]
    if len(sequence) != num_nodes or sorted(sequence) != list(range(num_nodes)):
        return False
    return all(
        (min(sequence[i], sequence[(i + 1) % num_nodes]), max(sequence[i], sequence[(i + 1) % num_nodes])) in edges
        for i in range(num_nodes)
    )


# ---------------------------------------------------------------------------
# car painting
# ---------------------------------------------------------------------------


def _verify_car_painting(prediction: str | None, meta) -> bool:
    """Any permutation within the K-shift budget that hits min_switches is correct."""
    meta = meta or {}
    car_ids = _as_list(meta.get("car_ids"))
    colors = _as_list(meta.get("colors"))
    shift_limit = _as_int(meta.get("K"))
    min_switches = _as_int(meta.get("min_switches"))
    if prediction is None or not car_ids or not colors or shift_limit is None or min_switches is None:
        return False
    try:
        original = [int(c) for c in car_ids]
    except (TypeError, ValueError, OverflowError):
        # OverflowError: a car id decoded from ``1e999``/``Infinity`` is not one.
        return False
    order = _int_list(prediction)
    if not order or sorted(order) != sorted(original):
        return False
    for position, car in enumerate(order, start=1):
        try:
            original_position = original.index(car) + 1
        except ValueError:
            return False
        if abs(position - original_position) > shift_limit:
            return False
    try:
        switches = sum(1 for a, b in zip(order, order[1:], strict=False) if colors[a - 1] != colors[b - 1])
    except (IndexError, TypeError):
        return False
    return switches == min_switches


# ---------------------------------------------------------------------------
# board tasks whose prompt mandates <begin_board> (campsite, star_battle)
# ---------------------------------------------------------------------------


def _board_rows(text: str | None) -> list[str] | None:
    """Return the whitespace-free board rows inside (or around) ``text``."""
    if not text:
        return None
    boards = _BOARD_RE.findall(text)
    body = boards[-1] if boards else text
    rows = []
    for line in body.splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith("```") or re.fullmatch(r"</?[A-Za-z_]+>", line):
            continue  # fence or a stray board/tag marker
        # campsite's ground truth prefixes the board with its constraint header
        # ("total number of tents: ..."), which the prompt never asks the model
        # to repeat.
        if re.match(r"^(?:total number of tents|tents in each (?:row|column))\s*:", line, re.IGNORECASE):
            continue
        rows.append(re.sub(r"\s+", "", line))
    return rows or None


def _verify_board_task(prediction: str | None, answer) -> bool:
    gold = _board_rows(str(answer))
    got = _board_rows(prediction)
    return bool(gold) and bool(got) and gold == got


# ---------------------------------------------------------------------------
# full_crosswords
# ---------------------------------------------------------------------------

_CROSSWORD_LINE_RE = re.compile(r"^(across|down)\s*[::]\s*(.+)$", re.IGNORECASE)


def _crossword_words(values) -> list[str]:
    words = _as_list(values) or []
    return [str(word).replace(" ", "").strip().upper() for word in words]


def _verify_crosswords(prediction: str | None, answer) -> bool:
    """Compare the across/down word lists written in either allowed shape.

    The prompt mandates ``across: W1, W2\\ndown: W1, W2`` while ``ground_truth``
    stores ``{"across": [...], "down": [...]}``; both are accepted.
    """
    if prediction is None:
        return False
    gold = _decode(answer)
    if not isinstance(gold, dict):
        return False
    gold_across, gold_down = _crossword_words(gold.get("across")), _crossword_words(gold.get("down"))
    if not gold_across and not gold_down:
        return False
    decoded = _decode(prediction)
    if isinstance(decoded, dict):
        got_across, got_down = _crossword_words(decoded.get("across")), _crossword_words(decoded.get("down"))
    else:
        parsed: dict[str, list[str]] = {}
        for line in prediction.splitlines():
            match = _CROSSWORD_LINE_RE.match(line.strip().strip("`*# ").strip())
            if not match:
                continue
            words = [w.strip().strip("\"'") for w in re.split(r"[,\s]+", match.group(2).strip())]
            parsed[match.group(1).lower()] = [w for w in words if w and set(w) != {"."}]
        if not parsed:
            return False
        got_across = _crossword_words(parsed.get("across"))
        got_down = _crossword_words(parsed.get("down"))
    return got_across == gold_across and got_down == gold_down


# ---------------------------------------------------------------------------
# zebra logic
# ---------------------------------------------------------------------------


def _table_rows(text) -> list[tuple[str, ...]]:
    """Split a Markdown table into its rows (labels included, layout ignored)."""
    rows: list[tuple[str, ...]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line.count("|") < 2:
            continue
        cells = tuple(re.sub(r"\s+", " ", cell.strip()).lower() for cell in line.split("|")[1:-1])
        if not cells or all(not cell for cell in cells):
            continue
        if all(set(cell) <= set("-: ") for cell in cells):  # |---|---| separator
            continue
        rows.append(cells)
    return rows


def _verify_zebra(prediction: str | None, answer) -> bool:
    """Official rule: every ground-truth row must appear in the model's table."""
    gold = _table_rows(answer)
    got = _table_rows(prediction)
    if not gold or not got:
        return False
    return all(row in got for row in gold)


# ---------------------------------------------------------------------------
# tic tac toe (3x3 optimal move)
# ---------------------------------------------------------------------------

_OPPONENT = {"X": "O", "O": "X"}


def _cell(token) -> str:
    token = str(token).strip().strip("\"'`").upper()
    return token if token in {"X", "O"} else ""


def _parse_board(text, size: int) -> list[list[str]] | None:
    """Parse the board from the prompt's quoted-token rows, a table or a literal."""
    decoded = _decode(text)
    if isinstance(decoded, list | tuple) and decoded and all(isinstance(r, list | tuple) for r in decoded):
        rows = [[_cell(cell) for cell in row] for row in decoded]
    else:
        rows = []
        if not isinstance(text, str):
            return None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("```") or set(line) <= set("-+ "):
                continue
            if "|" in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
            else:
                cells = re.findall(r'"[^"]*"|\'[^\']*\'|[^\s,]+', line)
            cells = [_cell(cell) for cell in cells]
            if len(cells) == size:
                rows.append(cells)
    if len(rows) < size:
        return None
    return rows[-size:]  # the prompt may be echoed before the answer


def _winner(board: list[list[str]]) -> str | None:
    size = len(board)
    lines = [list(row) for row in board]
    lines += [[board[i][j] for i in range(size)] for j in range(size)]
    lines.append([board[i][i] for i in range(size)])
    lines.append([board[i][size - 1 - i] for i in range(size)])
    for line in lines:
        if line[0] and all(cell == line[0] for cell in line):
            return line[0]
    return None


def _board_key(board: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row) for row in board)


@functools.lru_cache(maxsize=4096)
def _minimax(board: tuple[tuple[str, ...]], current: str, maximizing: bool, me: str) -> int:
    grid = [list(row) for row in board]
    winner = _winner(grid)
    if winner == me:
        return 1
    if winner:
        return -1
    if all(cell for row in grid for cell in row):
        return 0
    size = len(grid)
    scores = []
    for i in range(size):
        for j in range(size):
            if grid[i][j]:
                continue
            grid[i][j] = current
            scores.append(_minimax(_board_key(grid), _OPPONENT[current], not maximizing, me))
            grid[i][j] = ""
    if not scores:
        return 0
    return max(scores) if maximizing else min(scores)


def _immediate_moves(board: list[list[str]], player: str) -> list[tuple[int, int]]:
    """Winning moves for ``player``, else the moves that block the opponent."""
    size = len(board)
    grid = [row[:] for row in board]
    for target in (player, _OPPONENT[player]):
        moves = []
        for i in range(size):
            for j in range(size):
                if grid[i][j]:
                    continue
                grid[i][j] = target
                if _winner(grid) == target:
                    moves.append((i, j))
                grid[i][j] = ""
        if moves:
            return moves
    return []


def _best_moves_3x3(board: list[list[str]], player: str) -> list[tuple[int, int]]:
    immediate = _immediate_moves(board, player)
    if immediate:
        return immediate
    size = len(board)
    best_score: int | None = None
    best: list[tuple[int, int]] = []
    for i in range(size):
        for j in range(size):
            if board[i][j]:
                continue
            grid = [row[:] for row in board]
            grid[i][j] = player
            score = _minimax(_board_key(grid), _OPPONENT[player], False, player)
            if best_score is None or score > best_score:
                best_score, best = score, [(i, j)]
            elif score == best_score:
                best.append((i, j))
    return best


def _verify_tic_tac_toe(prediction: str | None, meta) -> bool:
    """Accept any optimal move, not only the generator's own board."""
    meta = meta or {}
    current = meta.get("current_board")
    player = str(meta.get("active_player") or "").strip().upper()
    if not isinstance(current, list) or player not in _OPPONENT:
        return False
    size = len(current)
    if size != 3 or prediction is None:
        return False
    board = [[_cell(cell) for cell in row] for row in current]
    predicted = _parse_board(prediction, size)
    if predicted is None:
        return False
    move = None
    for i in range(size):
        for j in range(size):
            if board[i][j] == predicted[i][j]:
                continue
            if board[i][j] or predicted[i][j] != player or move is not None:
                return False  # must be exactly the active player's single new mark
            move = (i, j)
    if move is None:
        return False
    return move in _best_moves_3x3(board, player)


# ---------------------------------------------------------------------------
# arithmetic-expression tasks (game24, countdown)
# ---------------------------------------------------------------------------


def _safe_arith_eval(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression; ``None`` if it is not one.

    Implemented as an AST walk (no ``eval``) so model output can never reach a
    builtin.  Only numeric literals, unary +/- and + - * / // % ** are allowed.
    """
    try:
        node = ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, int | float):
                raise ValueError("non-numeric literal")
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd | ast.USub):
            value = ev(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp) and type(n.op) in _ARITH_BINOPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow):
                if abs(right) > _MAX_POW_EXPONENT:
                    raise ValueError("exponent too large")
                # Estimate the result size *before* allocating it: bit length of
                # ``left ** right`` is ~``left.bit_length() * right``.
                if isinstance(left, int) and left.bit_length() * abs(right) > _MAX_INT_BITS:
                    raise ValueError("operand too large")
            value = _ARITH_BINOPS[type(n.op)](left, right)
            if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
                raise ValueError("intermediate operand too large")
            return value
        raise ValueError("disallowed expression node")

    try:
        return float(ev(node))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError, MemoryError, RecursionError):
        # MemoryError/RecursionError: the expression itself is too big/deep to be
        # an answer -- refuse it instead of stalling the reward worker.
        return None


def _normalise_expression(text: str) -> str:
    """Canonicalise an expression candidate: strip prose, drop ``= target``,
    map the multiplication/division glyphs the Enigmata prompts allow."""
    s = strip_prose_prefix(text or "")
    s = s.split("=")[0]
    for src, dst in (("×", "*"), ("÷", "/"), ("−", "-"), ("x", "*"), ("X", "*")):
        s = s.replace(src, dst)
    return s.strip()


def _number_key(value) -> str:
    """Canonical string for a number so 4 / 4.0 / "4" compare equal."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        # A metadata integer beyond ~1e308 has no float form (OverflowError);
        # ``str`` still gives it a stable, comparable key.
        return str(value)
    return str(int(f)) if f.is_integer() else repr(f)


def _target_from_answer(answer: str) -> float | None:
    """Recover the target from a countdown/arithmetic ground truth.

    ``"The answer is: 6*11-(15+13)/4 = 59"`` -> ``59.0``.
    """
    s = strip_prose_prefix(str(answer))
    if "=" in s:
        s = s.rsplit("=", 1)[-1]
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else None


def _verify_expression_task(prediction: str | None, answer, target: float, meta) -> bool:
    """Shared game24/countdown check: evaluate the expression and, when the
    input numbers are known, require exactly those numbers to be used."""
    answer_text = strip_prose_prefix(str(answer))
    if "cannot form" in answer_text.lower():
        return prediction is not None and "cannot form" in prediction.lower()
    if prediction is None:
        return False
    expr = _normalise_expression(prediction)
    value = _safe_arith_eval(expr)
    if value is None or abs(value - target) > 1e-4:
        return False
    numbers = _as_list((meta or {}).get("question"))
    numeric_numbers = None
    if numbers:
        try:
            numeric_numbers = sorted(_number_key(n) for n in numbers)
        except (TypeError, ValueError):
            numeric_numbers = None
    used = sorted(_NUMBER_RE.findall(expr))
    if numeric_numbers is not None:
        # The input numbers are known: the expression must use exactly them, so
        # the policy cannot hack the reward by emitting the bare target.
        if used != numeric_numbers:
            return False
    elif not re.search(r"[+\-*/]", expr):
        # Legacy row without metadata: at least require a real expression.
        return False
    return True


# ---------------------------------------------------------------------------
# maze
# ---------------------------------------------------------------------------


def _extract_coords(text: str) -> list[tuple[int, int]]:
    coords = []
    for row, col in _COORD_RE.findall(text or ""):
        try:
            coords.append((int(row), int(col)))
        except ValueError:
            # ``\d+`` can span thousands of digits, which CPython refuses to
            # convert (``sys.get_int_max_str_digits()``); skip such a "coordinate"
            # rather than aborting the reward pass.
            continue
    return coords


def _verify_maze(prediction: str | None, answer, meta) -> bool:
    """Official maze rule when ``meta`` carries the grid; else exact path match.

    Legacy rows (no ``meta``) fall back to comparing the coordinate sequences,
    which is correct whenever the puzzle has a unique path (the Enigmata maze
    generator emits one) and never creates false positives.
    """
    answer_text = strip_prose_prefix(str(answer))
    if "not exist" in answer_text.lower():
        return prediction is not None and "not exist" in prediction.lower()
    if prediction is None:
        return False
    coords = _extract_coords(prediction)
    if not coords:
        return False

    maze = (meta or {}).get("question")
    height = (meta or {}).get("height")
    width = (meta or {}).get("width")
    if isinstance(maze, list) and maze and height and width:
        try:
            height, width = int(height), int(width)
            if coords[0] != (1, 1) or coords[-1] != (height, width):
                return False
            for i, (row, col) in enumerate(coords):
                if not (1 <= row <= height and 1 <= col <= width):
                    return False
                if maze[row - 1][col - 1] == "B":
                    return False
                if i and abs(row - coords[i - 1][0]) + abs(col - coords[i - 1][1]) != 1:
                    return False
            return True
        except (IndexError, TypeError, ValueError, OverflowError):
            return False

    gt_coords = _extract_coords(answer_text)
    return bool(gt_coords) and coords == gt_coords


# ---------------------------------------------------------------------------
# stack_permutation
# ---------------------------------------------------------------------------


def _parse_stack_ops(text: str | None) -> list[str] | None:
    if not text:
        return None
    try:
        obj = ast.literal_eval(strip_prose_prefix(text).strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(obj, list | tuple) and obj and all(isinstance(x, str) for x in obj):
        return list(obj)
    return None


def _simulate_stack(ops: list[str], input_sequence: list, output_sequence: list) -> bool:
    stack: list = []
    input_idx = output_idx = 0
    try:
        for item in ops:
            if item.startswith("Pop"):
                if not stack or stack.pop() != output_sequence[output_idx]:
                    return False
                output_idx += 1
            elif item.startswith("Push"):
                push_element = int(item.strip()[5:-1])
                if push_element != input_sequence[input_idx]:
                    return False
                stack.append(input_sequence[input_idx])
                input_idx += 1
            else:
                return False
    except (IndexError, ValueError):
        return False
    return input_idx == len(input_sequence) and output_idx == len(output_sequence)


_INVALID_MARKERS = ("not a valid", "not valid")


def _verify_stack_permutation(prediction: str | None, answer, meta) -> bool:
    """Official simulation when ``meta`` carries input/output sequences.

    Legacy rows have no sequences in ``ground_truth`` and the prompt is not
    available to the reward function, so a sequence prediction cannot be
    validated without risking false positives -- those rows stay strict until
    the logic parquet is rebuilt with ``meta``.
    """
    answer_text = str(answer)
    answer_invalid = any(marker in answer_text.lower() for marker in _INVALID_MARKERS)
    prediction_invalid = prediction is not None and any(marker in prediction.lower() for marker in _INVALID_MARKERS)
    if answer_invalid:
        return prediction_invalid
    if prediction_invalid:
        return False
    ops = _parse_stack_ops(prediction)
    if ops is None:
        return False
    input_sequence = (meta or {}).get("input_sequence")
    output_sequence = (meta or {}).get("output_sequence")
    if not isinstance(input_sequence, list) or not isinstance(output_sequence, list):
        return False
    return _simulate_stack(ops, input_sequence, output_sequence)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def verify_enigmata(
    prediction: str | None,
    answer,
    task: str | None,
    meta=None,
) -> bool | None:
    """Return the Enigmata verdict, or ``None`` if the task is not handled here.

    Args:
        prediction: the model's extracted final answer (see ``extract_logic_answer``).
        answer: the raw Enigmata ``answer`` field stored in ``ground_truth``.
        task: the Enigmata ``task`` name stored in ``ground_truth``.
        meta: optional task metadata stored in ``ground_truth`` (numbers, maze
            grid, sequences, target).  Absent for rows built before the
            ``to_parquet_logic.py`` meta change.
    """
    if task not in HANDLED_TASKS:
        return None
    meta = meta if isinstance(meta, dict) else None

    if task == "game24":
        target = _as_int((meta or {}).get("target")) or 24
        return _verify_expression_task(prediction, answer, float(target), meta)
    if task == "countdown":
        target = _as_int((meta or {}).get("target"))
        if target is None:
            target = _target_from_answer(answer)
        if target is None:
            return None
        return _verify_expression_task(prediction, answer, float(target), meta)
    if task == "maze":
        return _verify_maze(prediction, answer, meta)
    if task == "stack_permutation":
        return _verify_stack_permutation(prediction, answer, meta)
    if task in {"eight_puzzle", "fifteen_puzzle"}:
        return _verify_sliding_puzzle(prediction, meta)
    if task in {"nine_puzzle", "sixteen_puzzle"}:
        return _verify_shift_puzzle(prediction, meta)
    if task == "twiddle":
        return _verify_twiddle(prediction, meta)
    if task in {"hamiltonian_path", "hamiltonian_cycle"}:
        return _verify_hamiltonian(prediction, answer, task, meta)
    if task == "car_painting":
        return _verify_car_painting(prediction, meta)
    if task in {"campsite", "star_battle"}:
        return _verify_board_task(prediction, answer)
    if task == "full_crosswords":
        return _verify_crosswords(prediction, answer)
    if task == "zebra_logic":
        return _verify_zebra(prediction, answer)
    if task == "tic_tac_toe":
        return _verify_tic_tac_toe(prediction, meta)
    return None
