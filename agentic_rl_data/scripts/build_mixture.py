#!/usr/bin/env python3
"""Deterministic mixture builder for the Agentic RL phase-1 dataset.

Reads ``data/raw_unified/{source}.jsonl`` (unified schema, produced by T1's
``scripts/build_raw.py``), applies the last-mile guards that belong to the mixture
stage, and writes ``data/mixture_v1/{train,heldout}.jsonl`` + ``mixture_stats.json``.

Spec: ``agentic_rl_data_plan.md`` section 3 step 7 (mixture + held-out), section 6
(20/30/25/15/10), section 7 (curriculum), section 8 acceptance #1 (one command,
byte-reproducible). Contracts: ``agentic_rl_data/CONTRACTS.md`` (C6 replay
placeholder, C9 turn accounting, C10 meta keys) and ``schema.py`` v2.

Reproducibility contract
------------------------
* Selection is a seeded *stable hash ranking*: for a given ``seed`` and salt,
  ``rank = sha256(seed \\x1f salt \\x1f task_id)`` as a 256-bit integer; smaller
  rank wins. Python's ``random`` module is deliberately not used, so a CPython
  upgrade cannot change the mixture.
* Ties on the same rank are broken by task id, never by input file order.
* The final files are sorted by ``(source, id)`` and written through
  ``schema.dump_jsonl`` (``sort_keys=True``, non-ASCII preserved).
* Stats contain no timestamps; sha256 of the outputs is included for diffing.

Soft dependency on T1: if a raw file is absent, ``--use-loaders`` tries
``agentic_rl_data.loaders.load_source(name, cfg)`` (README loader contract); if
that is not available the build fails with an explicit error.

Usage
-----
    python -m agentic_rl_data.scripts.build_mixture \
        --config agentic_rl_data/configs/mixture.yaml \
        --raw-dir data/raw_unified \
        --out-dir data/mixture_v1

    # dry run (compute + print stats, write nothing)
    python -m agentic_rl_data.scripts.build_mixture --dry-run

    # smoke-scale rebuild for plan section 7 ("per class ~500 first")
    python -m agentic_rl_data.scripts.build_mixture --total-train 2500

    # after T1 loaders are ready (no raw_unified files needed)
    python -m agentic_rl_data.scripts.build_mixture --use-loaders
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# package bootstrap: support both ``python -m ...`` and direct script execution
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - import guard is exercised by every run
    from agentic_rl_data import schema as schema
except ImportError as _import_exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import agentic_rl_data.schema (cwd={Path.cwd()}): {_import_exc}\n"
        "run from the repository root or add it to PYTHONPATH"
    ) from _import_exc

DEFAULT_CONFIG_PATH = _REPO_ROOT / "agentic_rl_data" / "configs" / "mixture.yaml"

_TRAIN_SALT = "train"
_HELDOUT_SALT = "heldout"
_GROUP_SALT = "group"

#: config values that are still empirical defaults and must be ablated
SUGGEST_KEYS = (
    "seed",
    "total_train",
    "passrate.band",
    "passrate.n_samples",
    "heldout.per_category",
    "filters.dedup.approximate.jaccard",
    "filters.decontam.ngram",
    "curriculum.stages",
    "categories.code_exec",
    "sources.toucan.subsample",
)


class MixtureError(RuntimeError):
    """Base class for mixture build failures (config or data level)."""


class MixtureConfigError(MixtureError):
    """The YAML config is missing keys / internally inconsistent."""


class MixtureDataError(MixtureError):
    """Input data is missing, unreadable or violates the frozen contracts."""


# ===========================================================================
# 1. Minimal YAML loader (PyYAML optional, stdlib-only fallback)
# ===========================================================================
#
# The committed config deliberately uses only block mappings, block sequences,
# scalars and flow sequences, so a ~120-line subset parser is enough. PyYAML is
# used when importable; ``_parse_yaml_subset`` remains the tested fallback.

def _strip_yaml_comment(line: str) -> str:
    """Drop ``#`` comments while respecting single/double quoted scalars."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in line:
        if quote == '"':
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quote = None
            continue
        if quote == "'":
            out.append(ch)
            if ch == "'":
                quote = None
            continue
        if ch == "#":
            break
        if ch in ("'", '"'):
            quote = ch
        out.append(ch)
    return "".join(out).rstrip()


def _find_unquoted(text: str, needle: str) -> int:
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == needle:
            return i
    return -1


def _split_flow_items(inner: str) -> list[str]:
    items: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in inner:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            continue
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current and "".join(current).strip():
        items.append("".join(current).strip())
    return items


def _parse_scalar_token(token: str, *, where: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar_token(part, where=where) for part in _split_flow_items(inner)]
    if (token.startswith("{") and token.endswith("}")) or token.startswith("{"):
        raise MixtureConfigError(
            f"{where}: flow mappings are not supported by the stdlib YAML fallback; "
            "use a block mapping"
        )
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_yaml_block(
    lines: list[tuple[int, str, int]], idx: int, indent: int, where: str
) -> tuple[Any, int]:
    if lines[idx][1].startswith("- ") or lines[idx][1] == "-":
        return _parse_yaml_sequence(lines, idx, indent, where)
    return _parse_yaml_mapping(lines, idx, indent, where)


def _parse_yaml_mapping(
    lines: list[tuple[int, str, int]], idx: int, indent: int, where: str
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while idx < len(lines):
        ind, content, lineno = lines[idx]
        if ind != indent:
            break
        if content == "-" or content.startswith("- "):
            break
        colon = _find_unquoted(content, ":")
        if colon <= 0:
            raise MixtureConfigError(f"{where}:{lineno}: expected 'key: value', got {content!r}")
        key = content[:colon].strip()
        payload = content[colon + 1 :].strip()
        if not key:
            raise MixtureConfigError(f"{where}:{lineno}: empty mapping key")
        if payload == "":
            if idx + 1 < len(lines) and lines[idx + 1][0] > indent:
                value, idx = _parse_yaml_block(lines, idx + 1, lines[idx + 1][0], where)
            else:
                value, idx = None, idx + 1
        else:
            value = _parse_scalar_token(payload, where=f"{where}:{lineno}")
            idx += 1
        if key in result:
            raise MixtureConfigError(f"{where}:{lineno}: duplicate key {key!r}")
        result[key] = value
    return result, idx


def _parse_yaml_sequence(
    lines: list[tuple[int, str, int]], idx: int, indent: int, where: str
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while idx < len(lines):
        ind, content, lineno = lines[idx]
        if ind != indent:
            break
        if not (content == "-" or content.startswith("- ")):
            break
        payload = content[1:].strip()
        if payload == "":
            if idx + 1 < len(lines) and lines[idx + 1][0] > indent:
                value, idx = _parse_yaml_block(lines, idx + 1, lines[idx + 1][0], where)
            else:
                value, idx = None, idx + 1
            items.append(value)
            continue
        colonish = _find_unquoted(payload, ":")
        is_quoted = payload[0] in ("'", '"') and payload[-1] in ("'", '"') and len(payload) >= 2
        if colonish > 0 and not is_quoted:
            # "- key: value" starts an inline mapping item: virtual first line at
            # indent+2, then every following line indented deeper than the dash.
            sub_lines: list[tuple[int, str, int]] = [(indent + 2, payload, lineno)]
            j = idx + 1
            while j < len(lines) and lines[j][0] > indent:
                sub_lines.append(lines[j])
                j += 1
            value, consumed = _parse_yaml_mapping(sub_lines, 0, indent + 2, where)
            if consumed != len(sub_lines):
                raise MixtureConfigError(
                    f"{where}:{lineno}: inconsistent indentation inside sequence item"
                )
            items.append(value)
            idx = j
        else:
            items.append(_parse_scalar_token(payload, where=f"{where}:{lineno}"))
            idx += 1
    return items, idx


def _parse_yaml_subset(text: str, *, source: str = "<yaml>") -> dict[str, Any]:
    """Stdlib-only parser for the YAML subset used by ``mixure.yaml``."""
    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = _strip_yaml_comment(raw)
        if not stripped.strip() or stripped.strip() == "---":
            continue
        if "\t" in stripped:
            raise MixtureConfigError(f"{source}:{lineno}: tabs are not allowed for indentation")
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip(), lineno))
    if not lines:
        return {}
    if lines[0][1].startswith("- "):
        raise MixtureConfigError(f"{source}: top-level must be a mapping")
    value, idx = _parse_yaml_mapping(lines, 0, lines[0][0], source)
    if idx != len(lines):
        raise MixtureConfigError(f"{source}:{lines[idx][2]}: unexpected indentation")
    return value


def load_yaml_document(text: str, *, source: str = "<yaml>") -> dict[str, Any]:
    """PyYAML when available, deterministic stdlib parser otherwise."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _parse_yaml_subset(text, source=source)
    try:
        doc = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - PyYAML raises several classes
        raise MixtureConfigError(f"{source}: invalid YAML: {exc}") from exc
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise MixtureConfigError(f"{source}: top-level must be a mapping")
    return doc


# ===========================================================================
# 2. Config schema (parsed + validated once, then frozen)
# ===========================================================================

@dataclass(frozen=True)
class SourceSpec:
    name: str
    file: str
    task_type: str
    required: bool
    enabled: bool
    subsample: int | None
    license: str | None
    note: str
    #: registry loader name when it differs from the config source name
    #: (e.g. one ``code_exec`` loader serves apps/codecontests/taco via cfg["kind"])
    loader: str | None = None
    loader_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CategorySource:
    name: str
    weight: float
    cap: int | None


@dataclass(frozen=True)
class CategorySpec:
    name: str
    share: float
    heldout: int
    turn_cap: int
    on_unavailable: str
    sources: tuple[CategorySource, ...]


@dataclass(frozen=True)
class StageSpec:
    max_turns: int
    start_fraction: float
    end_fraction: float


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    version: int
    seed: int
    total_train: int
    raw_dir: str
    out_dir: str
    train_name: str
    heldout_name: str
    stats_name: str
    emit_curriculum: bool
    passrate_enabled: bool
    band_min: float
    band_max: float
    n_samples: int
    missing_policy: str
    smoothed: bool
    passrate_require_report: bool
    heldout_per_category: int
    group_aware: bool
    heldout_oversample: int
    max_gt_turns: int
    invalid_schema: str
    exact_dedup: bool
    placeholder_gt_returns: str
    placeholder_keep_share: float
    placeholder_weight: float
    curriculum_enabled: bool
    curriculum_filter_turns: int
    stages: tuple[StageSpec, ...]
    sources: dict[str, SourceSpec]
    categories: tuple[CategorySpec, ...]

    def category(self, name: str) -> CategorySpec:
        for cat in self.categories:
            if cat.name == name:
                return cat
        raise MixtureConfigError(f"unknown category {name!r}")


def _expect_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MixtureConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _expect_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MixtureConfigError(f"{where} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise MixtureConfigError(f"{where} must be >= {minimum}, got {value}")
    return value


def _expect_float(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MixtureConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise MixtureConfigError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _expect_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise MixtureConfigError(f"{where} must be a bool, got {value!r}")
    return value


def _get_block(raw: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    return _expect_mapping(value, f"{where}.{key}")


def parse_config(raw: Any) -> Config:
    """Validate the raw YAML dict and freeze it into a :class:`Config`.

    All structural problems are reported as :class:`MixtureConfigError` with the
    offending config path, because a silently-wrong mixture is unrecoverable.
    """
    raw = _expect_mapping(raw, "config")
    version = _expect_int(raw.get("version", 1), "version", minimum=1)
    seed = _expect_int(raw.get("seed"), "seed")
    total_train = _expect_int(raw.get("total_train"), "total_train", minimum=1)

    paths = _get_block(raw, "paths", "config")
    output = _get_block(raw, "output", "config")
    raw_dir = _expect_str(paths.get("raw_dir", "data/raw_unified"), "paths.raw_dir")
    out_dir = _expect_str(output.get("dir", "data/mixture_v1"), "output.dir")
    train_name = _expect_str(output.get("train", "train.jsonl"), "output.train")
    heldout_name = _expect_str(output.get("heldout", "heldout.jsonl"), "output.heldout")
    stats_name = _expect_str(output.get("stats", "mixture_stats.json"), "output.stats")
    emit_curriculum = _expect_bool(output.get("emit_curriculum", False), "output.emit_curriculum")

    passrate = _get_block(raw, "passrate", "config")
    passrate_enabled = _expect_bool(passrate.get("enabled", True), "passrate.enabled")
    band = passrate.get("band")
    if band is None:
        band_min, band_max = 0.125, 0.875
    elif isinstance(band, dict):
        band_min = _expect_float(band.get("min", 0.125), "passrate.band.min")
        band_max = _expect_float(band.get("max", 0.875), "passrate.band.max")
    elif isinstance(band, list) and len(band) == 2:
        band_min = _expect_float(band[0], "passrate.band[0]")
        band_max = _expect_float(band[1], "passrate.band[1]")
    else:
        raise MixtureConfigError("passrate.band must be a mapping {min,max} or a 2-list")
    if not (0.0 < band_min < band_max < 1.0):
        raise MixtureConfigError(
            f"passrate.band must satisfy 0 < min < max < 1, got [{band_min}, {band_max}]"
        )
    n_samples = _expect_int(passrate.get("n_samples", 8), "passrate.n_samples", minimum=2)
    missing_policy = _expect_str(passrate.get("missing", "keep"), "passrate.missing")
    if missing_policy not in ("keep", "drop"):
        raise MixtureConfigError("passrate.missing must be 'keep' or 'drop'")
    smoothed = _expect_bool(passrate.get("smoothed", False), "passrate.smoothed")
    passrate_require_report = _expect_bool(
        passrate.get("require_report", False), "passrate.require_report"
    )

    heldout = _get_block(raw, "heldout", "config")
    heldout_per_category = _expect_int(
        heldout.get("per_category", 300), "heldout.per_category", minimum=1
    )
    # plan section 3 step 7: "每类别至少 200 条"
    if heldout_per_category < 200:
        raise MixtureConfigError(
            f"heldout.per_category={heldout_per_category} violates plan section 3 "
            "(at least 200 per category)"
        )
    group_aware = _expect_bool(heldout.get("group_aware", True), "heldout.group_aware")
    heldout_oversample = _expect_int(
        heldout.get("oversample", 2), "heldout.oversample", minimum=1
    )

    filters = _get_block(raw, "filters", "config")
    max_gt_turns = _expect_int(filters.get("max_gt_turns", 15), "filters.max_gt_turns", minimum=1)
    invalid_schema = _expect_str(filters.get("invalid_schema", "drop"), "filters.invalid_schema")
    if invalid_schema not in ("drop", "error"):
        raise MixtureConfigError("filters.invalid_schema must be 'drop' or 'error'")
    dedup = _get_block(filters, "dedup", "filters")
    exact_dedup = _expect_bool(dedup.get("exact", True), "filters.dedup.exact")
    mt_fc_filters = filters.get("mt_fc")
    if mt_fc_filters is None:
        placeholder_policy = "drop"
    else:
        mt_fc_filters = _expect_mapping(mt_fc_filters, "filters.mt_fc")
        placeholder_policy = _expect_str(
            mt_fc_filters.get("placeholder_gt_returns", "drop"),
            "filters.mt_fc.placeholder_gt_returns",
        )
    if placeholder_policy not in ("drop", "keep"):
        raise MixtureConfigError(
            "filters.mt_fc.placeholder_gt_returns must be 'drop' or 'keep' "
            "(CONTRACTS.md C6)"
        )
    if mt_fc_filters is not None:
        placeholder_keep_share = _expect_float(
            mt_fc_filters.get("placeholder_keep_share", 0.0),
            "filters.mt_fc.placeholder_keep_share",
        )
        placeholder_weight = _expect_float(
            mt_fc_filters.get("placeholder_weight", 0.5),
            "filters.mt_fc.placeholder_weight",
        )
    else:
        placeholder_keep_share, placeholder_weight = 0.0, 0.5
    if not (0.0 <= placeholder_keep_share <= 1.0):
        raise MixtureConfigError(
            f"filters.mt_fc.placeholder_keep_share must be in [0, 1], got {placeholder_keep_share}"
        )
    if not (0.0 < placeholder_weight <= 1.0):
        raise MixtureConfigError(
            f"filters.mt_fc.placeholder_weight must be in (0, 1], got {placeholder_weight}"
        )
    if placeholder_policy == "drop" and placeholder_keep_share > 0.0:
        raise MixtureConfigError(
            "filters.mt_fc.placeholder_gt_returns='drop' conflicts with "
            "placeholder_keep_share > 0; use 'keep' to preserve part of the quota"
        )

    curriculum = _get_block(raw, "curriculum", "config")
    curriculum_enabled = _expect_bool(curriculum.get("enabled", True), "curriculum.enabled")
    curriculum_filter_turns = _expect_int(
        curriculum.get("filter_gt_turns", max_gt_turns),
        "curriculum.filter_gt_turns",
        minimum=1,
    )
    stages_raw = curriculum.get("stages")
    if stages_raw is None:
        stages: list[StageSpec] = [
            StageSpec(5, 0.0, 0.2),
            StageSpec(10, 0.2, 0.5),
            StageSpec(15, 0.5, 1.0),
        ]
    else:
        if not isinstance(stages_raw, list) or not stages_raw:
            raise MixtureConfigError("curriculum.stages must be a non-empty list")
        stages = []
        for i, stage in enumerate(stages_raw):
            stage = _expect_mapping(stage, f"curriculum.stages[{i}]")
            stages.append(
                StageSpec(
                    max_turns=_expect_int(
                        stage.get("max_turns"), f"curriculum.stages[{i}].max_turns", minimum=1
                    ),
                    start_fraction=_expect_float(
                        stage.get("start_fraction", 0.0), f"curriculum.stages[{i}].start_fraction"
                    ),
                    end_fraction=_expect_float(
                        stage.get("end_fraction", 1.0), f"curriculum.stages[{i}].end_fraction"
                    ),
                )
            )
    if stages:
        if abs(stages[0].start_fraction) > 1e-9 or abs(stages[-1].end_fraction - 1.0) > 1e-9:
            raise MixtureConfigError("curriculum stages must cover fractions 0.0 -> 1.0")
        previous_turns = 0
        for i, stage in enumerate(stages):
            if stage.max_turns <= previous_turns:
                raise MixtureConfigError(
                    f"curriculum.stages[{i}].max_turns must increase "
                    f"(got {stage.max_turns} after {previous_turns})"
                )
            previous_turns = stage.max_turns
            if not (0.0 <= stage.start_fraction < stage.end_fraction <= 1.0):
                raise MixtureConfigError(
                    f"curriculum.stages[{i}] fractions must satisfy 0 <= start < end <= 1"
                )
            if i + 1 < len(stages) and abs(stage.end_fraction - stages[i + 1].start_fraction) > 1e-9:
                raise MixtureConfigError("curriculum stages must be contiguous")

    sources_raw = _expect_mapping(raw.get("sources"), "sources")
    sources: dict[str, SourceSpec] = {}
    for name, spec_raw in sources_raw.items():
        spec = _expect_mapping(spec_raw, f"sources.{name}")
        task_type = _expect_str(spec.get("task_type"), f"sources.{name}.task_type")
        if task_type not in schema.TASK_TYPES:
            raise MixtureConfigError(
                f"sources.{name}.task_type {task_type!r} not in {sorted(schema.TASK_TYPES)}"
            )
        subsample = spec.get("subsample")
        if subsample is not None:
            subsample = _expect_int(subsample, f"sources.{name}.subsample", minimum=1)
        license_name = spec.get("license")
        if license_name is not None:
            license_name = _expect_str(license_name, f"sources.{name}.license")
        sources[name] = SourceSpec(
            name=name,
            file=_expect_str(spec.get("file"), f"sources.{name}.file"),
            task_type=task_type,
            required=_expect_bool(spec.get("required", False), f"sources.{name}.required"),
            enabled=_expect_bool(spec.get("enabled", True), f"sources.{name}.enabled"),
            subsample=subsample,
            license=license_name,
            note=str(spec.get("note") or ""),
            loader=(
                _expect_str(spec.get("loader"), f"sources.{name}.loader")
                if spec.get("loader") is not None
                else None
            ),
            loader_cfg=(
                _expect_mapping(spec.get("loader_cfg"), f"sources.{name}.loader_cfg")
                if spec.get("loader_cfg") is not None
                else {}
            ),
        )

    categories_raw = _expect_mapping(raw.get("categories"), "categories")
    categories: list[CategorySpec] = []
    share_sum = 0.0
    for cat_name, cat_raw in categories_raw.items():
        cat = _expect_mapping(cat_raw, f"categories.{cat_name}")
        share = _expect_float(cat.get("share"), f"categories.{cat_name}.share")
        if not (0.0 < share < 1.0):
            raise MixtureConfigError(f"categories.{cat_name}.share must be in (0, 1)")
        share_sum += share
        cat_heldout = _expect_int(
            cat.get("heldout", heldout_per_category), f"categories.{cat_name}.heldout", minimum=1
        )
        if cat_heldout < 200:
            raise MixtureConfigError(
                f"categories.{cat_name}.heldout={cat_heldout} violates plan section 3 (>= 200)"
            )
        turn_cap = _expect_int(cat.get("turn_cap", max_gt_turns), f"categories.{cat_name}.turn_cap", minimum=1)
        if turn_cap > max_gt_turns:
            raise MixtureConfigError(
                f"categories.{cat_name}.turn_cap={turn_cap} exceeds filters.max_gt_turns={max_gt_turns}"
            )
        on_unavailable = _expect_str(
            cat.get("on_unavailable", "fail"), f"categories.{cat_name}.on_unavailable"
        )
        if on_unavailable not in ("fail", "skip", "redistribute"):
            raise MixtureConfigError(
                f"categories.{cat_name}.on_unavailable must be fail|skip|redistribute"
            )
        sources_list = cat.get("sources")
        if not isinstance(sources_list, list) or not sources_list:
            raise MixtureConfigError(f"categories.{cat_name}.sources must be a non-empty list")
        members: list[CategorySource] = []
        weight_sum = 0.0
        for i, member_raw in enumerate(sources_list):
            member = _expect_mapping(member_raw, f"categories.{cat_name}.sources[{i}]")
            source_name = _expect_str(member.get("name"), f"categories.{cat_name}.sources[{i}].name")
            if source_name not in sources:
                raise MixtureConfigError(
                    f"categories.{cat_name} references unknown source {source_name!r}"
                )
            if not sources[source_name].enabled:
                raise MixtureConfigError(
                    f"categories.{cat_name} references disabled source {source_name!r}"
                )
            weight = _expect_float(member.get("weight"), f"categories.{cat_name}.{source_name}.weight")
            if not (0.0 < weight <= 1.0):
                raise MixtureConfigError(
                    f"categories.{cat_name}.{source_name}.weight must be in (0, 1]"
                )
            weight_sum += weight
            cap = member.get("cap")
            if cap is not None:
                cap = _expect_int(cap, f"categories.{cat_name}.{source_name}.cap", minimum=1)
            members.append(CategorySource(source_name, weight, cap))
        if abs(weight_sum - 1.0) > 1e-6:
            raise MixtureConfigError(
                f"categories.{cat_name} source weights must sum to 1.0, got {weight_sum}"
            )
        categories.append(
            CategorySpec(
                name=cat_name,
                share=share,
                heldout=cat_heldout,
                turn_cap=turn_cap,
                on_unavailable=on_unavailable,
                sources=tuple(members),
            )
        )
    if abs(share_sum - 1.0) > 1e-6:
        raise MixtureConfigError(f"category shares must sum to 1.0, got {share_sum}")
    if not categories:
        raise MixtureConfigError("categories must not be empty")

    used = {member.name for cat in categories for member in cat.sources}
    unused = {name for name, spec in sources.items() if spec.enabled and name not in used}
    if unused:
        raise MixtureConfigError(
            f"enabled sources not referenced by any category: {sorted(unused)}"
        )

    return Config(
        raw=raw,
        version=version,
        seed=seed,
        total_train=total_train,
        raw_dir=raw_dir,
        out_dir=out_dir,
        train_name=train_name,
        heldout_name=heldout_name,
        stats_name=stats_name,
        emit_curriculum=emit_curriculum,
        passrate_enabled=passrate_enabled,
        band_min=band_min,
        band_max=band_max,
        n_samples=n_samples,
        missing_policy=missing_policy,
        smoothed=smoothed,
        passrate_require_report=passrate_require_report,
        heldout_per_category=heldout_per_category,
        group_aware=group_aware,
        heldout_oversample=heldout_oversample,
        max_gt_turns=max_gt_turns,
        invalid_schema=invalid_schema,
        exact_dedup=exact_dedup,
        placeholder_gt_returns=placeholder_policy,
        placeholder_keep_share=placeholder_keep_share,
        placeholder_weight=placeholder_weight,
        curriculum_enabled=curriculum_enabled,
        curriculum_filter_turns=curriculum_filter_turns,
        stages=tuple(stages),
        sources=sources,
        categories=tuple(categories),
    )


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise MixtureConfigError(f"config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    return parse_config(load_yaml_document(text, source=str(config_path)))


# ===========================================================================
# 3. deterministic helpers (ranking, turns, pass-rate guard)
# ===========================================================================

def rank_of(seed: int, salt: str, value: str) -> int:
    """Stable 256-bit rank; smaller wins. Input-order independent by design."""
    digest = hashlib.sha256(f"{seed}\x1f{salt}\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _ranked_tuple(seed: int, salt: str, task: Mapping[str, Any]) -> tuple[int, str, str]:
    """Deterministic heap key: (rank, id, canonical payload).

    The canonical payload is the final tie-break so duplicate ids in a raw file
    still order deterministically (and never fall through to comparing dicts,
    which would raise TypeError inside ``heapq``).
    """
    tid = str(task.get("id"))
    return rank_of(seed, salt, tid), tid, schema.canonical_json(task)


def effective_turns(task: Mapping[str, Any]) -> int:
    """CONTRACTS.md C9 turn accounting used by S2 filtering, curriculum, reports."""
    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    n_calls = len(reference.get("gt_calls") or []) if isinstance(reference, Mapping) else 0
    gt_turns = reference.get("gt_turns") if isinstance(reference, Mapping) else None
    n_gt_turns = len(gt_turns) if isinstance(gt_turns, list) else 0
    expected = meta.get("expected_turns")
    n_expected = int(expected) if isinstance(expected, int) and not isinstance(expected, bool) else 0
    return max(n_calls, n_gt_turns, n_expected, 1)


#: Schema v2.2 constant; no silent fallback (critic T2-08).
try:
    GT_RETURNS_SOURCES = schema.GT_RETURNS_SOURCES
except AttributeError as _exc:  # pragma: no cover - schema contract guard
    raise RuntimeError(
        "agentic_rl_data/schema.py must provide GT_RETURNS_SOURCES (schema v2.2)"
    ) from _exc


def declared_gt_returns_source(task: Mapping[str, Any]) -> str | None:
    """Return the loader's *static* declaration (CONTRACTS.md C6/D11), else None.

    ``meta["gt_returns_source"]`` is stamped by T1 loaders and is machine-checked
    by schema (v2.2 requires it for every ``mt_fc`` task). Nothing in the env is
    executed; this is a pure data-side declaration.
    """
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    value = meta.get("gt_returns_source")
    if isinstance(value, str) and value in GT_RETURNS_SOURCES:
        return value
    return None


def infer_gt_returns_source(task: Mapping[str, Any]) -> str:
    """Fallback inference for un-stamped rows (C6 priority order)."""
    env = task.get("env") if isinstance(task.get("env"), Mapping) else {}
    env_config = env.get("config") if isinstance(env.get("config"), Mapping) else {}
    if env_config.get("gt_returns"):
        return "config"
    reference = task.get("reference") if isinstance(task.get("reference"), Mapping) else {}
    if reference.get("tool_returns"):
        return "reference"
    gt_turns = reference.get("gt_turns")
    if isinstance(gt_turns, list):
        for turn in gt_turns:
            if not isinstance(turn, Mapping):
                continue
            for key in ("tool_results", "results", "return"):
                value = turn.get(key)
                if value:
                    return "gt_turns"
    return "placeholder"


def gt_returns_source(task: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(source_kind, origin)``; origin is ``static`` or ``inferred``.

    Static declaration always wins (Lead D11); inference only fills the gap for
    legacy / un-stamped rows, and the two counts are reported separately.
    """
    declared = declared_gt_returns_source(task)
    if declared is not None:
        return declared, "static"
    return infer_gt_returns_source(task), "inferred"


def pass_rate_rejection(task: Mapping[str, Any], cfg: Config) -> str | None:
    """Return a drop reason for the pass-rate guard (plan section 3 step 6).

    Mandatory rule: pass_rate must be strictly inside (0, 1); all-correct /
    all-wrong tasks are always dropped. The configured band is applied on top;
    it is recommended by the report and configurable (ablation item).
    """
    if not cfg.passrate_enabled:
        return None
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    rate = meta.get("pass_rate")
    if rate is None:
        return "missing_pass_rate" if cfg.missing_policy == "drop" else None
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        return "invalid_pass_rate"
    rate = float(rate)
    if rate <= 0.0:
        return "all_wrong"
    if rate >= 1.0:
        return "all_correct"
    if rate < cfg.band_min or rate > cfg.band_max:
        return "outside_passrate_band"
    return None


def _task_group(task: Mapping[str, Any]) -> str:
    meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
    group = meta.get("group_id")
    if isinstance(group, str) and group.strip():
        return group
    return str(task.get("id"))


def _task_dedup_key(task: Mapping[str, Any]) -> str:
    return schema.task_dedup_key(task)


# ===========================================================================
# 4. input: per-source pool selection (bounded memory, deterministic)
# ===========================================================================

@dataclass
class SourcePool:
    """Top-ranked candidates of one source for one category."""

    name: str
    train: list[Mapping[str, Any]]
    held: list[Mapping[str, Any]]
    stats: dict[str, Any]


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise MixtureDataError(f"cannot open {path}: {exc}") from exc
    with handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MixtureDataError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise MixtureDataError(f"{path}:{lineno}: row must be a JSON object")
            yield row


def _loader_tasks(
    spec: SourceSpec, cfg: Config, raw_dir: Path
) -> Iterable[dict[str, Any]] | None:
    """Soft dependency on T1's loader registry (README loader contract)."""
    try:
        from agentic_rl_data import loaders as loader_pkg  # type: ignore[attr-defined]
    except ImportError:
        return None
    load_source = getattr(loader_pkg, "load_source", None)
    if not callable(load_source):
        return None
    loader_name = spec.loader or spec.name
    loader_cfg: dict[str, Any] = {
        "path": str(raw_dir / spec.file),
        "raw_dir": str(raw_dir),
        "source": spec.name,
        "license": spec.license,
    }
    loader_cfg.update(spec.loader_cfg)
    try:
        try:
            return load_source(loader_name, loader_cfg)
        except TypeError:
            return load_source(loader_name)
    except KeyError as exc:
        # loaders registry has no such source (e.g. user-side reasoning_math pool)
        return None
    except Exception as exc:  # noqa: BLE001 - surface loader errors with context
        raise MixtureDataError(f"loader for source {spec.name!r} failed: {exc}") from exc


def _source_iter(
    cfg: Config, spec: SourceSpec, *, raw_dir: Path, use_loaders: bool
) -> tuple[Iterator[dict[str, Any]], str, dict[str, Any]]:
    """Return (iterator, input kind, stats) for one source."""
    path = raw_dir / spec.file
    info: dict[str, Any] = {"file": str(path), "input": "jsonl"}
    if path.exists():
        if path.stat().st_size == 0:
            info["input"] = "empty"
            return iter(()), "empty", info
        return _iter_jsonl(path), "jsonl", info
    if use_loaders:
        tasks = _loader_tasks(spec, cfg, raw_dir)
        if tasks is not None:
            info["input"] = "loaders"
            return iter(tasks), "loaders", info
        raise MixtureDataError(
            f"source {spec.name!r}: raw file {path} is missing and "
            "agentic_rl_data.loaders.load_source is unavailable; "
            "run T1's scripts/build_raw.py first or wait for the loader registry"
        )
    if spec.required:
        raise MixtureDataError(
            f"required source {spec.name!r} is missing: {path}\n"
            "  -> run T1 scripts/build_raw.py (plan section 3 step 1), or pass "
            "--use-loaders once the loader registry is ready"
        )
    info["input"] = "missing_optional"
    return iter(()), "missing_optional", info


def _keep_best(
    heap: list[tuple[tuple[int, str, str], Mapping[str, Any]]],
    limit: int,
    seed: int,
    salt: str,
    task: Mapping[str, Any],
) -> None:
    """Bounded top-``limit`` heap keyed by (rank, id, payload); order-independent."""
    if limit <= 0:
        return
    key = _ranked_tuple(seed, salt, task)
    if len(heap) < limit:
        heapq.heappush(heap, (key, task))
        return
    # heap[0] is the *worst* kept candidate: largest rank then largest id
    if key < heap[0][0]:
        heapq.heapreplace(heap, (key, task))


def _heap_tasks(
    heap: list[tuple[tuple[int, str, str], Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Sorted best-first list from the bounded heap."""
    ordered = sorted(heap, key=lambda item: item[0])
    return [task for _, task in ordered]


def load_source_pool(
    cfg: Config,
    category: CategorySpec,
    member: CategorySource,
    *,
    category_quota: int,
    raw_dir: Path,
    use_loaders: bool,
) -> SourcePool:
    """Stream one source, apply local guards and keep the top candidates only."""
    spec = cfg.sources[member.name]
    scaled_cap = _scale_cap(cfg, member.cap)
    train_want = (
        scaled_cap
        if scaled_cap is not None
        else category_quota + category.heldout
    )
    held_want = min(
        category.heldout,
        int(math.ceil(category.heldout * member.weight)) + cfg.heldout_oversample,
    )
    train_limit = train_want + held_want + 8
    held_limit = held_want + 8
    if spec.subsample is not None:
        train_limit = min(train_limit, spec.subsample)
        held_limit = min(held_limit, spec.subsample)

    stats: dict[str, Any] = {
        "file": spec.file,
        "task_type": spec.task_type,
        "required": spec.required,
        "license": spec.license,
        "cap_declared": member.cap,
        "cap_scaled": scaled_cap,
        "rows": 0,
        "valid": 0,
        "invalid_schema": 0,
        "invalid_examples": [],
        "dropped": {},
        "placeholder_static": 0,
        "placeholder_inferred": 0,
        "placeholder_kept": 0,
        "gt_returns_sources": {},
        "train_candidates": 0,
        "held_candidates": 0,
    }

    def _drop(reason: str, count: int = 1) -> None:
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + count

    train_heap: list[tuple[tuple[int, str, str], Mapping[str, Any]]] = []
    held_heap: list[tuple[tuple[int, str, str], Mapping[str, Any]]] = []

    iterator, input_kind, _input_info = _source_iter(
        cfg, spec, raw_dir=raw_dir, use_loaders=use_loaders
    )
    stats["file"] = str(raw_dir / spec.file)
    stats["input"] = input_kind
    if input_kind in {"missing", "missing_optional"}:
        return SourcePool(member.name, [], [], stats)
    if input_kind == "empty":
        if spec.required:
            raise MixtureDataError(
                f"required source {spec.name!r} is empty: {raw_dir / spec.file}"
            )
        stats["dropped"]["empty_file"] = 1
        return SourcePool(member.name, [], [], stats)

    max_turns = min(cfg.max_gt_turns, category.turn_cap)
    for task in iterator:
        stats["rows"] += 1
        if not isinstance(task, dict):
            raise MixtureDataError(
                f"source {spec.name!r}: loader yielded {type(task).__name__}, expected dict"
            )
        issues = schema.validate_task(task)
        if issues:
            if cfg.invalid_schema == "error":
                raise MixtureDataError(
                    f"source {spec.name!r}: schema-invalid task {task.get('id')!r}: {issues}"
                )
            stats["invalid_schema"] += 1
            if len(stats["invalid_examples"]) < 5:
                stats["invalid_examples"].append(
                    {"id": task.get("id"), "errors": issues[:3]}
                )
            continue
        stats["valid"] += 1

        if schema_expected_turns_exceeded(task, max_turns):
            _drop("over_max_turns")
            continue
        reason = pass_rate_rejection(task, cfg)
        if reason:
            _drop(reason)
            continue
        if spec.task_type == "mt_fc":
            source_kind, origin = gt_returns_source(task)
            stats["gt_returns_sources"][source_kind] = (
                stats["gt_returns_sources"].get(source_kind, 0) + 1
            )
            if source_kind == "placeholder":
                # Static loader declaration (C6/D11) must not be conflated with
                # fallback inference; both counts are reported separately.
                if origin == "static":
                    stats["placeholder_static"] += 1
                else:
                    stats["placeholder_inferred"] += 1
                if (
                    cfg.placeholder_gt_returns == "drop"
                    or cfg.placeholder_keep_share <= 0.0
                ):
                    _drop("degraded_replay_placeholder")
                    continue
                stats["placeholder_kept"] += 1
                # Kept only as budgeted training mass; never into held-out
                # (degraded replay would make evaluation meaningless).
                _keep_best(train_heap, train_limit, cfg.seed, _TRAIN_SALT, task)
                continue

        _keep_best(train_heap, train_limit, cfg.seed, _TRAIN_SALT, task)
        _keep_best(held_heap, held_limit, cfg.seed, _HELDOUT_SALT, task)

    pool = SourcePool(
        member.name,
        _heap_tasks(train_heap),
        _heap_tasks(held_heap),
        stats,
    )
    stats["train_candidates"] = len(pool.train)
    stats["held_candidates"] = len(pool.held)
    return pool


def schema_expected_turns_exceeded(task: Mapping[str, Any], max_turns: int) -> bool:
    return effective_turns(task) > max_turns


# ===========================================================================
# 5. quota allocation (largest-remainder with caps; deterministic)
# ===========================================================================

def largest_remainder(total: int, shares: Sequence[tuple[str, float]]) -> dict[str, int]:
    """Split ``total`` over named shares exactly, deterministic remainder order."""
    if total < 0:
        raise ValueError("total must be >= 0")
    weight_sum = sum(weight for _, weight in shares)
    if weight_sum <= 0:
        raise ValueError("shares must be positive")
    counts: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    allocated = 0
    for name, weight in shares:
        ideal = total * weight / weight_sum
        base = int(math.floor(ideal))
        counts[name] = base
        allocated += base
        fractions.append((ideal - base, name))
    leftover = total - allocated
    # largest fractional part first, name as deterministic tie-break
    for _, name in sorted(fractions, key=lambda item: (-item[0], item[1]))[:leftover]:
        counts[name] += 1
    if leftover > len(fractions):
        # only possible with floating-point oddities; assign to the largest share
        name = max(shares, key=lambda item: item[1])[0]
        counts[name] += leftover - len(fractions)
    return counts


def allocate_train_counts(
    target: int,
    entries: Sequence[tuple[str, float, int | None, int]],
) -> dict[str, int]:
    """Allocate ``target`` rows over sources proportional to weight, bounded by cap/avail.

    ``entries`` = (name, weight, cap_or_none, available). Returns actual counts;
    the sum can be below target when the capacities are exhausted (caller reports
    the deficit instead of silently violating the cap).
    """
    caps = {
        name: (available if cap is None else min(cap, available))
        for name, _weight, cap, available in entries
    }
    alloc = {name: 0 for name, _weight, _cap, _available in entries}
    remaining = target
    weights = {name: weight for name, weight, _cap, _available in entries}
    while remaining > 0:
        active = [name for name in caps if caps[name] > 0 and alloc[name] < caps[name]]
        if not active:
            break
        weight_sum = sum(weights[name] for name in active)
        if weight_sum <= 0:
            break
        progressed = 0
        fractions: list[tuple[float, str]] = []
        for name in active:
            capacity_left = caps[name] - alloc[name]
            ideal = remaining * weights[name] / weight_sum
            give = min(int(math.floor(ideal)), capacity_left)
            if give > 0:
                alloc[name] += give
                remaining -= give
                progressed += give
            fractions.append((ideal - int(math.floor(ideal)), name))
        if remaining > 0 and progressed == 0:
            candidates = [name for name in active if caps[name] > alloc[name]]
            if not candidates:
                break
            pick = max(candidates, key=lambda name: (fractions_dict(fractions)[name], name))
            alloc[pick] += 1
            remaining -= 1
    return alloc


def fractions_dict(fractions: Sequence[tuple[float, str]]) -> dict[str, float]:
    return {name: fraction for fraction, name in fractions}


# ===========================================================================
# 6. held-out selection (group aware when meta.group_id exists)
# ===========================================================================

def _group_weighted_key(seed: int, source_weight: float, group: str) -> float:
    """Exponential-race key: proportional to ``source_weight``, deterministic.

    P(group selected first) is proportional to its source weight, which keeps
    group-aware held-out roughly on the configured source mix.
    """
    u = rank_of(seed, _GROUP_SALT, group) / float(1 << 256)
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return -math.log(1.0 - u) / max(source_weight, 1e-9)


def select_heldout(
    cfg: Config,
    category: CategorySpec,
    pools: Sequence[SourcePool],
    target: int,
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    """Choose held-out rows; never split a group across train/held-out."""
    info: dict[str, Any] = {"groups": 0, "held": 0}
    held: dict[str, list[Mapping[str, Any]]] = {pool.name: [] for pool in pools}
    if target <= 0 or not pools:
        return held, info

    if not cfg.group_aware:
        entries = [
            (pool.name, weight, None, len(pool.held))
            for pool, weight in ((pool, _weight_of(category, pool.name)) for pool in pools)
        ]
        counts = allocate_train_counts(target, entries)
        for pool in pools:
            held[pool.name] = list(pool.held[: counts.get(pool.name, 0)])
        info["held"] = sum(len(v) for v in held.values())
        return held, info

    # Build groups from the *union* of train/held candidates, so that a group whose
    # best-ranked member sits in the held pool but whose other members sit in the
    # train pool is still selected as a whole (no session leakage across the split).
    candidates: dict[str, Mapping[str, Any]] = {}
    for pool in pools:
        for task in list(pool.train) + list(pool.held):
            candidates.setdefault(str(task.get("id")), task)
    group_tasks: dict[str, list[Mapping[str, Any]]] = {}
    group_weight: dict[str, float] = {}
    source_for_group: dict[str, str] = {}
    pool_names = {pool.name for pool in pools}
    for gid, task in ((_task_group(task), task) for task in candidates.values()):
        source_name = str(task.get("source"))
        if source_name not in pool_names:
            raise MixtureDataError(
                f"task {task.get('id')!r} has source={source_name!r} but category "
                f"{category.name!r} only contains {sorted(pool_names)}; "
                "loaders must set task['source'] to the configured source name"
            )
        source_for_group.setdefault(gid, source_name)
        group_tasks.setdefault(gid, []).append(task)
    partial_groups = 0
    complete_groups: list[str] = []
    for gid, tasks in group_tasks.items():
        source_name = source_for_group.get(gid)
        if source_name not in group_weight:
            if source_name is None:
                continue
            group_weight[gid] = _weight_of(category, source_name)
        declared = _declared_group_size(tasks)
        if declared is not None and len(tasks) < declared:
            partial_groups += 1
            continue
        complete_groups.append(gid)
    groups_sorted = sorted(
        complete_groups,
        key=lambda gid: (
            _group_weighted_key(cfg.seed, group_weight[gid], gid),
            source_for_group[gid],
            gid,
        ),
    )
    chosen: list[str] = []
    total = 0
    for gid in groups_sorted:
        size = len(group_tasks[gid])
        if total + size > target:
            continue
        chosen.append(gid)
        total += size
    if not chosen and groups_sorted:
        # A single group larger than the target: still take it whole (session
        # leakage is worse than overshooting held-out by a few rows).
        chosen.append(groups_sorted[0])
        total = len(group_tasks[groups_sorted[0]])
        partial_groups += 1
    for gid in chosen:
        source = str(group_tasks[gid][0].get("source"))
        held.setdefault(source, []).extend(group_tasks[gid])
    info["groups"] = len(chosen)
    info["held"] = total
    info["partial_groups"] = partial_groups
    info["overshoot"] = 1 if total > target else 0
    return held, info


def _declared_group_size(tasks: Sequence[Mapping[str, Any]]) -> int | None:
    """Group size declared by the loader (meta.group_size), else unknown."""
    for task in tasks:
        meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
        size = meta.get("group_size")
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            return size
    return None


def _weight_of(category: CategorySpec, source_name: str) -> float:
    for member in category.sources:
        if member.name == source_name:
            return member.weight
    raise MixtureConfigError(f"source {source_name!r} not in category {category.name!r}")


# ===========================================================================
# 7. build
# ===========================================================================

def _config_digest(cfg: Config, overrides: Mapping[str, Any]) -> str:
    payload = schema.canonical_json({"config": cfg.raw, "overrides": overrides})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _materialise(
    pools: Sequence[SourcePool],
    held_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    train_counts: Mapping[str, int],
    *,
    placeholder_budget: int | None = None,
) -> tuple[
    dict[str, list[Mapping[str, Any]]], dict[str, int], dict[str, int]
]:
    """Take train rows per source after excluding held rows, respecting dedup.

    ``placeholder_budget`` caps how many ``mt_fc`` rows with degraded (placeholder)
    gt returns may enter this category (CONTRACTS.md C6). ``None`` disables the cap
    (non-mt_fc categories / explicit keep-all is never used).
    """
    train: dict[str, list[Mapping[str, Any]]] = {}
    duplicate_counts: dict[str, int] = {}
    placeholder_counts: dict[str, int] = {}
    remaining_budget = placeholder_budget
    for pool in pools:
        held_ids = {str(task.get("id")) for task in held_by_source.get(pool.name, [])}
        kept: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        duplicates = 0
        placeholders = 0
        for task in pool.train:
            tid = str(task.get("id"))
            if tid in held_ids:
                continue
            key = _task_dedup_key(task)
            if key in seen:
                duplicates += 1
                continue
            is_placeholder = (
                task.get("task_type") == "mt_fc"
                and gt_returns_source(task)[0] == "placeholder"
            )
            if is_placeholder:
                if remaining_budget is not None:
                    if remaining_budget <= 0:
                        continue
                    remaining_budget -= 1
                placeholders += 1
            seen.add(key)
            kept.append(task)
            if len(kept) >= train_counts.get(pool.name, 0):
                break
        train[pool.name] = kept
        duplicate_counts[pool.name] = duplicates
        placeholder_counts[pool.name] = placeholders
    return train, duplicate_counts, placeholder_counts


def _placeholder_budget(cfg: Config, headline_target: int) -> int:
    if cfg.placeholder_gt_returns != "keep" or cfg.placeholder_keep_share <= 0.0:
        return 0
    return int(math.floor(headline_target * cfg.placeholder_keep_share))


def _scale_cap(cfg: Config, cap: int | None) -> int | None:
    """Scale an absolute per-source cap with ``total_train`` (critic T2-01).

    Caps in ``mixture.yaml`` are absolute row counts at the config's baseline
    ``total_train`` (8000 for the phase-1 pilot). When the build overrides
    ``--total-train`` (e.g. 2500 smoke scale / 32k full scale), every cap scales
    linearly so the configured share of each source is preserved instead of the
    small sources being starved at larger scales.
    """
    if cap is None:
        return None
    base = cfg.raw.get("total_train")
    try:
        base_total = int(base) if base is not None else cfg.total_train
    except (TypeError, ValueError):
        base_total = cfg.total_train
    if base_total <= 0:
        base_total = cfg.total_train
    ratio = cfg.total_train / float(base_total)
    return max(1, int(round(cap * ratio)))


def spec_task_type(cfg: Config, source_name: str) -> str:
    spec = cfg.sources.get(source_name)
    if spec is None:
        raise MixtureConfigError(f"unknown source {source_name!r}")
    return spec.task_type


def _category_slots(
    cfg: Config,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Total train quota per category via largest remainder over category shares."""
    quotas = largest_remainder(
        cfg.total_train, [(cat.name, cat.share) for cat in cfg.categories]
    )
    return quotas, []


def build_mixture(
    cfg: Config,
    *,
    raw_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    total_train: int | None = None,
    use_loaders: bool = False,
    strict: bool = False,
    emit_curriculum: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build train/held-out files and return the stats payload.

    The returned dict is exactly what ``mixture_stats.json`` contains, so tests
    can assert on it without reading files.
    """
    effective_seed = int(seed) if seed is not None else cfg.seed
    effective_total = int(total_train) if total_train is not None else cfg.total_train
    if effective_total < 1:
        raise MixtureConfigError("total_train must be positive")
    raw_root = Path(raw_dir) if raw_dir is not None else Path(cfg.raw_dir)
    out_root = Path(out_dir) if out_dir is not None else Path(cfg.out_dir)
    curriculum_flag = cfg.emit_curriculum if emit_curriculum is None else bool(emit_curriculum)

    if strict and cfg.invalid_schema == "drop":
        cfg = replace(cfg, invalid_schema="error")

    overrides = {
        "seed": effective_seed,
        "total_train": effective_total,
        "raw_dir": str(raw_root),
        "out_dir": str(out_root),
        "use_loaders": use_loaders,
        "strict": strict,
        "emit_curriculum": curriculum_flag,
    }
    # Re-seed config copy; frozen dataclass so replace() is safe.
    cfg = replace(cfg, seed=effective_seed, total_train=effective_total)

    quotas, _ = _category_slots(cfg)
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "config_sha256": _config_digest(cfg, overrides),
        "seed": effective_seed,
        "total_train_target": effective_total,
        "total_train_actual": 0,
        "heldout_target": 0,
        "heldout_actual": 0,
        "config_version": cfg.version,
        "switches": {
            "passrate": {
                "enabled": cfg.passrate_enabled,
                "band": [cfg.band_min, cfg.band_max],
                "n_samples": cfg.n_samples,
                "missing": cfg.missing_policy,
                "smoothed": cfg.smoothed,
            },
            "dedup_exact": cfg.exact_dedup,
            "decontam": _declared_switch(cfg.raw, "filters", "decontam"),
            "dedup_approximate": _declared_switch(cfg.raw, "filters", "dedup"),
            "curriculum": [stage.max_turns for stage in cfg.stages] if cfg.curriculum_enabled else [],
            "placeholders": {
                "policy": cfg.placeholder_gt_returns,
                "keep_share": cfg.placeholder_keep_share,
                "downweight": cfg.placeholder_weight,
                "rule": "CONTRACTS.md C6/D11: static meta.gt_returns_source wins, "
                "inference only as fallback",
            },
            "schemas": {
                "reference_keys": list(schema.REQUIRED_REFERENCE_KEYS),
                "meta_keys": list(schema.REQUIRED_META_KEYS),
            },
        },
        # Explicit "was this step actually executed on this data?" flags, so a
        # "final" mixture cannot imply the upstream filters ran (critic T2-04).
        "filters_applied": {
            "passrate": False,
            "passrate_note": "build_mixture only guards meta.pass_rate; run "
            "scripts/filter_by_passrate.py first and mark data_report.md [RUN]",
            "placeholder_static": True,
            "dedup_exact_in_build": cfg.exact_dedup,
            "decontam": False,
            "decontam_note": "executed by T1 scripts/decontam.py; build only records "
            "the declared switch (0-hit acceptance is T1's report)",
        },
        "categories": {},
        "sources": {},
        "warnings": warnings,
    }

    pools_by_category: dict[str, list[SourcePool]] = {}
    held_by_category: dict[str, dict[str, list[Mapping[str, Any]]]] = {}

    # --- phase 1: pools ----------------------------------------------------
    for cat in cfg.categories:
        pools: list[SourcePool] = []
        for member in cat.sources:
            pool = load_source_pool(
                cfg,
                cat,
                member,
                category_quota=quotas[cat.name],
                raw_dir=raw_root,
                use_loaders=use_loaders,
            )
            stats["sources"][f"{cat.name}:{pool.name}"] = pool.stats
            pools.append(pool)
        usable = sum(len(pool.train) + len(pool.held) for pool in pools)
        stats["categories"][cat.name] = {
            "share_target": cat.share,
            "train_target": quotas[cat.name],
            "heldout_target": cat.heldout,
            "usable_candidates": usable,
            "on_unavailable": cat.on_unavailable,
        }
        if usable == 0:
            if cat.on_unavailable == "fail":
                raise MixtureDataError(
                    f"category {cat.name!r} has no usable rows; sources: "
                    f"{[member.name for member in cat.sources]}. "
                    "Build the raw data first or change categories.<name>.on_unavailable."
                )
            stats["categories"][cat.name]["usable_candidates"] = 0
            continue
        pools_by_category[cat.name] = pools

    # --- phase 2: held-out -------------------------------------------------
    for cat in cfg.categories:
        pools = pools_by_category.get(cat.name, [])
        if not pools:
            held_by_category[cat.name] = {}
            stats["categories"][cat.name]["heldout_actual"] = 0
            continue
        held, held_info = select_heldout(cfg, cat, pools, cat.heldout)
        held_by_category[cat.name] = held
        held_count = sum(len(v) for v in held.values())
        stats["categories"][cat.name]["heldout_actual"] = held_count
        stats["categories"][cat.name]["heldout_groups"] = held_info.get("groups", 0)
        if held_count < cat.heldout:
            warnings.append(
                f"category {cat.name}: held-out {held_count} < target {cat.heldout} "
                "(group-aware packing or pool shortage)"
            )

    # --- phase 3: train quotas --------------------------------------------
    train_by_category: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    deficit = 0
    for cat in cfg.categories:
        pools = pools_by_category.get(cat.name)
        if not pools:
            deficit += quotas[cat.name]
            stats["categories"][cat.name]["train_actual"] = 0
            continue
        entries = []
        for pool in pools:
            member = next(m for m in cat.sources if m.name == pool.name)
            held_ids = {str(t.get("id")) for t in held_by_category[cat.name].get(pool.name, [])}
            available = sum(1 for t in pool.train if str(t.get("id")) not in held_ids)
            # When placeholders are dropped/budgeted, availability for allocation
            # must not count rows that cannot be taken.
            if (
                spec_task_type(cfg, pool.name) == "mt_fc"
                and _placeholder_budget(cfg, quotas[cat.name]) == 0
            ):
                available = sum(
                    1
                    for t in pool.train
                    if str(t.get("id")) not in held_ids
                    and gt_returns_source(t)[0] != "placeholder"
                )
            entries.append((pool.name, member.weight, _scale_cap(cfg, member.cap), available))
        counts = allocate_train_counts(quotas[cat.name], entries)
        ph_budget = _placeholder_budget(cfg, quotas[cat.name])
        train, duplicates, placeholders = _materialise(
            pools,
            held_by_category[cat.name],
            counts,
            placeholder_budget=ph_budget,
        )
        actual = sum(len(v) for v in train.values())
        train_by_category[cat.name] = train
        deficit += quotas[cat.name] - actual
        stats["categories"][cat.name].update(
            {
                "train_target": quotas[cat.name],
                "train_actual": actual,
                "train_source_counts": {name: len(rows) for name, rows in train.items()},
                "train_duplicates_exact": duplicates,
                "placeholder_budget": ph_budget,
                "placeholder_train": sum(placeholders.values()),
            }
        )

    # --- phase 4: absorb deficit from categories that still have capacity ---
    if deficit > 0:
        stats["deficit_initial"] = deficit
        for cat in cfg.categories:
            pools = pools_by_category.get(cat.name)
            if not pools or cat.on_unavailable == "skip":
                continue
            if deficit <= 0:
                break
            entries = []
            for pool in pools:
                member = next(m for m in cat.sources if m.name == pool.name)
                held_ids = {str(t.get("id")) for t in held_by_category[cat.name].get(pool.name, [])}
                current = len(train_by_category[cat.name].get(pool.name, []))
                available = sum(
                    1 for t in pool.train if str(t.get("id")) not in held_ids
                )
                entries.append((pool.name, member.weight, _scale_cap(cfg, member.cap), available - current))
            extra = allocate_train_counts(deficit, entries)
            if not any(extra.values()):
                continue
            base_counts = {
                name: len(rows) for name, rows in train_by_category[cat.name].items()
            }
            merged = {name: base_counts.get(name, 0) + extra.get(name, 0) for name in base_counts}
            used_budget = int(stats["categories"][cat.name].get("placeholder_train", 0))
            remaining_budget = max(
                0, _placeholder_budget(cfg, quotas[cat.name]) - used_budget
            )
            train, _dups, placeholders = _materialise(
                pools,
                held_by_category[cat.name],
                merged,
                placeholder_budget=remaining_budget,
            )
            new_actual = sum(len(v) for v in train.values())
            gain = new_actual - stats["categories"][cat.name]["train_actual"]
            stats["categories"][cat.name]["train_actual"] = new_actual
            stats["categories"][cat.name]["train_source_counts"] = {
                name: len(rows) for name, rows in train.items()
            }
            stats["categories"][cat.name]["placeholder_train"] = used_budget + sum(
                placeholders.values()
            )
            stats["categories"][cat.name]["absorbed_deficit"] = gain
            train_by_category[cat.name] = train
            deficit -= gain
        stats["deficit_remaining"] = deficit
        if deficit > 0:
            warnings.append(
                f"total_train shortfall {deficit}: not enough eligible rows under caps; "
                "effective shares deviate from section 6 (see mixture_stats.json)"
            )
    else:
        stats["deficit_initial"] = 0
        stats["deficit_remaining"] = 0

    # --- phase 5: final dedup + materialise all ----------------------------
    train_all: list[Mapping[str, Any]] = []
    for cat in cfg.categories:
        for name in sorted(train_by_category.get(cat.name, {})):
            train_all.extend(train_by_category[cat.name][name])
    held_all: list[Mapping[str, Any]] = []
    for cat in cfg.categories:
        for name in sorted(held_by_category.get(cat.name, {})):
            held_all.extend(held_by_category[cat.name][name])

    train_all, held_all, dedup_stats = _final_dedup(cfg, train_all, held_all)
    if dedup_stats["train_removed"] or dedup_stats["held_removed"]:
        warnings.append(
            "final exact dedup removed rows: "
            f"train={dedup_stats['train_removed']} held={dedup_stats['held_removed']}"
        )

    trimmed_placeholders = _trim_placeholders_to_share(cfg, train_all, train_by_category, stats)
    if trimmed_placeholders:
        warnings.append(
            f"placeholder keep-share cap trimmed {trimmed_placeholders} mt_fc row(s) "
            "(actual mt_fc rows fell below the headline target)"
        )

    # --- phase 6: write ----------------------------------------------------
    train_path = out_root / cfg.train_name
    held_path = out_root / cfg.heldout_name
    stats_path = out_root / cfg.stats_name
    stats["outputs"] = {}
    if not dry_run:
        train_count = schema.dump_jsonl(train_all, train_path)
        held_count = schema.dump_jsonl(held_all, held_path)
        stats["outputs"]["train"] = {
            "path": str(train_path),
            "count": train_count,
            "sha256": _sha256_file(train_path),
        }
        stats["outputs"]["heldout"] = {
            "path": str(held_path),
            "count": held_count,
            "sha256": _sha256_file(held_path),
        }
        if curriculum_flag and cfg.curriculum_enabled:
            stats["outputs"]["curriculum"] = _emit_curriculum(cfg, train_all, out_root)
    else:
        stats["outputs"]["train"] = {"path": str(train_path), "count": len(train_all), "sha256": ""}
        stats["outputs"]["heldout"] = {
            "path": str(held_path), "count": len(held_all), "sha256": ""
        }

    stats["total_train_actual"] = len(train_all)
    stats["heldout_actual"] = len(held_all)
    stats["heldout_target"] = sum(cat.heldout for cat in cfg.categories)
    stats["dedup"] = dedup_stats
    _finalise_stats_shares(stats, cfg, train_all, held_all)

    if not dry_run:
        _write_json(stats_path, stats)
    return stats


def _trim_placeholders_to_share(
    cfg: Config,
    train_all: list[Mapping[str, Any]],
    train_by_category: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    stats: dict[str, Any],
) -> int:
    """Trim placeholder mt_fc rows so their share of actual mt_fc rows <= keep_share.

    The per-category placeholder budget is computed from the *headline target*;
    when a category under-fills (deficit), that budget can exceed ``keep_share``
    of the **actual** mt_fc rows.  The excess placeholder rows are dropped and
    the category stats updated so ``mixture_stats.json`` stays self-consistent.
    Returns the number of trimmed rows (0 when the cap is not triggered).
    """
    keep = float(cfg.placeholder_keep_share)
    if cfg.placeholder_gt_returns != "keep" or keep <= 0.0 or keep >= 1.0:
        return 0

    def is_placeholder(task: Mapping[str, Any]) -> bool:
        return task.get("task_type") == "mt_fc" and gt_returns_source(task)[0] == "placeholder"

    placeholders = [task for task in train_all if is_placeholder(task)]
    mt_total = sum(1 for task in train_all if task.get("task_type") == "mt_fc")
    real_mt = mt_total - len(placeholders)
    if not placeholders or real_mt <= 0:
        return 0
    limit = int(math.floor(real_mt * keep / (1.0 - keep)))
    excess = len(placeholders) - limit
    if excess <= 0:
        return 0
    removed = {id(task) for task in placeholders[:excess]}
    train_all[:] = [task for task in train_all if id(task) not in removed]
    for cat in cfg.categories:
        for name, rows in list(train_by_category.get(cat.name, {}).items()):
            kept_rows = [task for task in rows if id(task) not in removed]
            dropped = len(rows) - len(kept_rows)
            if not dropped:
                continue
            train_by_category[cat.name][name] = kept_rows  # type: ignore[index]
            cat_stats = stats.get("categories", {}).get(cat.name)
            if isinstance(cat_stats, dict):
                cat_stats["train_actual"] = max(0, int(cat_stats.get("train_actual", 0)) - dropped)
                cat_stats["placeholder_train"] = max(
                    0, int(cat_stats.get("placeholder_train", 0)) - dropped
                )
                cat_stats["train_source_counts"] = {
                    key: len(value) for key, value in train_by_category[cat.name].items()
                }
    return excess


def _declared_switch(raw: Mapping[str, Any], block: str, key: str) -> Any:
    value = (raw.get(block) or {}).get(key)
    return value


def _final_dedup(
    cfg: Config,
    train_all: Sequence[Mapping[str, Any]],
    held_all: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Exact dedup of the final split; held-out wins over train on a collision."""
    held_sorted = sorted(held_all, key=lambda task: (str(task.get("source", "")), str(task.get("id", ""))))
    train_sorted = sorted(train_all, key=lambda task: (str(task.get("source", "")), str(task.get("id", ""))))
    held_removed = 0
    held_keys: set[str] = set()
    held_kept: list[Mapping[str, Any]] = []
    for task in held_sorted:
        key = _task_dedup_key(task)
        if key in held_keys:
            held_removed += 1
            continue
        held_keys.add(key)
        held_kept.append(task)
    train_removed = 0
    train_keys: set[str] = set()
    train_kept: list[Mapping[str, Any]] = []
    for task in train_sorted:
        key = _task_dedup_key(task)
        if key in held_keys:
            train_removed += 1
            continue
        if key in train_keys:
            train_removed += 1
            continue
        train_keys.add(key)
        train_kept.append(task)
    return train_kept, held_kept, {"train_removed": train_removed, "held_removed": held_removed}


def _emit_curriculum(
    cfg: Config, train_all: Sequence[Mapping[str, Any]], out_root: Path
) -> dict[str, Any]:
    """Write per-stage train subsets (approximate turn estimate, plan section 7)."""
    curriculum_dir = out_root / "curriculum"
    emitted: dict[str, Any] = {}
    for stage in cfg.stages:
        limit = min(stage.max_turns, cfg.max_gt_turns)
        subset = [task for task in train_all if effective_turns(task) <= limit]
        path = curriculum_dir / f"train_max_turns{stage.max_turns}.jsonl"
        count = schema.dump_jsonl(subset, path)
        emitted[f"stage_{stage.max_turns}"] = {
            "path": str(path),
            "count": count,
            "max_turns": limit,
            "fraction": [stage.start_fraction, stage.end_fraction],
            "sha256": _sha256_file(path),
        }
    return emitted


def _finalise_stats_shares(
    stats: dict[str, Any],
    cfg: Config,
    train_all: Sequence[Mapping[str, Any]],
    held_all: Sequence[Mapping[str, Any]],
) -> None:
    actual_total = len(train_all) or 1
    for cat in cfg.categories:
        cat_stats = stats["categories"].setdefault(cat.name, {})
        actual = int(cat_stats.get("train_actual", 0))
        cat_stats["share_target"] = cat.share
        cat_stats["share_effective"] = actual / actual_total
        cat_stats["share_deviation"] = cat_stats["share_effective"] - cat.share
    # task_type distribution of the final split (plan section 3 step 8)
    def _task_type_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in rows:
            key = str(task.get("task_type"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    stats["task_type_counts"] = {
        "train": _task_type_counts(train_all),
        "heldout": _task_type_counts(held_all),
    }
    stats["turn_distribution"] = _turn_distribution(train_all)
    stats["pass_rate_distribution"] = _pass_rate_distribution(train_all)
    stats["placeholder"] = {
        "policy": cfg.placeholder_gt_returns,
        "keep_share": cfg.placeholder_keep_share,
        "downweight": cfg.placeholder_weight,
        "static_declared": sum(
            int(row.get("placeholder_static", 0)) for row in stats["sources"].values()
        ),
        "inferred_fallback": sum(
            int(row.get("placeholder_inferred", 0)) for row in stats["sources"].values()
        ),
        "train_rows": sum(
            int(cat.get("placeholder_train", 0)) for cat in stats["categories"].values()
        ),
        "mt_train_rows": stats["task_type_counts"]["train"].get("mt_fc", 0),
    }
    mt_rows = stats["placeholder"]["mt_train_rows"]
    stats["placeholder"]["share_of_mt_train"] = (
        stats["placeholder"]["train_rows"] / mt_rows if mt_rows else 0.0
    )


def _turn_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11-15": 0, "over_15": 0}
    values: list[int] = []
    for task in rows:
        turns = effective_turns(task)
        values.append(turns)
        if turns <= 1:
            buckets["1"] += 1
        elif turns <= 3:
            buckets["2-3"] += 1
        elif turns <= 5:
            buckets["4-5"] += 1
        elif turns <= 10:
            buckets["6-10"] += 1
        elif turns <= 15:
            buckets["11-15"] += 1
        else:
            buckets["over_15"] += 1
    if not values:
        return {"buckets": buckets, "mean": None, "max": None, "min": None}
    return {
        "buckets": buckets,
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
    }


def _pass_rate_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates: list[float] = []
    missing = 0
    for task in rows:
        meta = task.get("meta") if isinstance(task.get("meta"), Mapping) else {}
        rate = meta.get("pass_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            rates.append(float(rate))
        else:
            missing += 1
    if not rates:
        return {"count": 0, "missing": missing}
    buckets = {"(0,0.25]": 0, "0.25-0.5": 0, "0.5-0.625": 0, "0.625-0.75": 0, "0.75-1)": 0}
    for rate in rates:
        if rate <= 0.25:
            buckets["(0,0.25]"] += 1
        elif rate <= 0.5:
            buckets["0.25-0.5"] += 1
        elif rate <= 0.625:
            buckets["0.5-0.625"] += 1
        elif rate <= 0.75:
            buckets["0.625-0.75"] += 1
        else:
            buckets["0.75-1)"] += 1
    ordered = sorted(rates)
    median = ordered[len(ordered) // 2]
    return {
        "count": len(rates),
        "missing": missing,
        "mean": sum(rates) / len(rates),
        "median": median,
        "buckets": buckets,
    }


# ===========================================================================
# 8. CLI
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_mixture",
        description=(
            "Build data/mixture_v1/{train,heldout}.jsonl from data/raw_unified/*.jsonl "
            "deterministically (plan section 3 step 7, section 6/7)."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"mixture YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--raw-dir", default=None, help="override paths.raw_dir")
    parser.add_argument("--out-dir", default=None, help="override output.dir")
    parser.add_argument("--seed", type=int, default=None, help="override seed (ablation only)")
    parser.add_argument(
        "--total-train",
        type=int,
        default=None,
        help="override total_train (e.g. 2500 for the plan section 7 smoke scale)",
    )
    parser.add_argument(
        "--use-loaders",
        action="store_true",
        help="when a raw file is missing, fall back to agentic_rl_data.loaders (T1)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="schema-invalid rows are fatal instead of dropped-and-counted",
    )
    parser.add_argument(
        "--emit-curriculum",
        action="store_true",
        default=None,
        help="also write curriculum/train_max_turns*.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    return parser


def _summary_lines(stats: Mapping[str, Any]) -> list[str]:
    lines = [
        "[build_mixture] seed={seed} total_train={total_train_actual}/{total_train_target} "
        "heldout={heldout_actual}/{heldout_target}".format(**stats),
    ]
    for cat_name, cat_stats in stats["categories"].items():
        lines.append(
            "  {name:<12} target={share_target:>6.2%} actual={actual:>6.2%} "
            "train={train:>4} held={held:>4}".format(
                name=cat_name,
                share_target=cat_stats.get("share_target", 0.0),
                actual=cat_stats.get("share_effective", 0.0),
                train=cat_stats.get("train_actual", 0),
                held=cat_stats.get("heldout_actual", 0),
            )
        )
    if stats.get("warnings"):
        for warning in stats["warnings"]:
            lines.append(f"  WARNING: {warning}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        stats = build_mixture(
            cfg,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            total_train=args.total_train,
            use_loaders=args.use_loaders,
            strict=args.strict,
            emit_curriculum=args.emit_curriculum,
            dry_run=args.dry_run,
        )
    except MixtureError as exc:
        print(f"[build_mixture] ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[build_mixture] ERROR (I/O): {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        for line in _summary_lines(stats):
            print(line)
    else:
        print(
            f"[build_mixture] train={stats['total_train_actual']} "
            f"heldout={stats['heldout_actual']} warnings={len(stats['warnings'])}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
