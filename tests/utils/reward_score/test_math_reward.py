# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
from pathlib import Path


def _load_math_reward():
    path = Path(__file__).resolve().parents[3] / "verl/utils/reward_score/math_reward.py"
    spec = importlib.util.spec_from_file_location("math_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math_reward = _load_math_reward()


def test_tuple_fraction_answer():
    sol = r"Therefore \boxed{\left( \frac{7}{9}, \frac{2}{9} \right)}"
    gt = r"\left( \frac{7}{9}, \frac{2}{9} \right)"
    assert math_reward.compute_score(sol, gt) == 1.0


def test_multiple_choice_letter():
    assert math_reward.compute_score(r"answer \boxed{E}", "(E)") == 1.0
    assert math_reward.compute_score(r"answer \boxed{\text{E}}", "(E)") == 1.0


def test_extract_boxed_answer_rejects_garbage():
    bad = r"Thus \boxed{1} and also \boxed{\right] }"
    assert math_reward.extract_boxed_answer(bad) == "1"


def test_extract_boxed_answer_skips_boxed_space_trap():
    sol = r"Therefore $\boxed \left[\begin{array}{c}1\end{array}\right] }$"
    boxed = math_reward.last_boxed_only_string(sol)
    assert boxed is not None
    content = math_reward.remove_boxed(boxed)
    assert math_reward._is_valid_boxed_content(content)


GT_WORDS = (
    "buxton callus cameron contribute extensible marque methanol olympic precise "
    "procrustean seepage shelf sideboard tty typescript unitary verify"
)


def test_word_list_aligned_boxed():
    sol = r"""\boxed{
\begin{aligned}
&buxton, \\
&callus, \\
&cameron, \\
&contribute, \\
&precise, \\
&extensible, \\
&marque, \\
&methanol, \\
&olympic, \\
&procrustean, \\
&seepage, \\
&shelf, \\
&sideboard, \\
&tty, \\
&typescript, \\
&unitary, \\
&verify
\end{aligned}
}"""
    assert math_reward.compute_score(sol, GT_WORDS) == 1.0


def test_word_list_text_and_comma_boxed():
    sol_text = (
        r"\boxed{\text{buxton, callus, cameron, contribute, extensible, marque, "
        r"methanol, olympic, precise, procrustean, seepage, shelf, sideboard, tty, "
        r"typescript, unitary, verify}}"
    )
    sol_comma = (
        r"\boxed{buxton, callus, cameron, contribute, extensible, marque, methanol, "
        r"olympic, precise, procrustean, seepage, shelf, sideboard, tty, typescript, "
        r"unitary, verify}"
    )
    assert math_reward.compute_score(sol_text, GT_WORDS) == 1.0
    assert math_reward.compute_score(sol_comma, GT_WORDS) == 1.0


def test_extract_math_answer_rejects_garbage_ground_truth():
    assert math_reward.extract_math_answer(r"\boxed{)}") is None
    assert math_reward.extract_math_answer(r"\boxed{> > )}") is None
    assert math_reward.extract_math_answer(r"\boxed{( { ( ) } )}") is None


def test_reject_garbage_boxed_and_ground_truth():
    assert math_reward.last_boxed_only_string(r"answer \boxed{)}") is None
    assert math_reward.last_boxed_only_string(r"answer \boxed{ ] )}") is None
    assert not math_reward.is_valid_ground_truth("} )")
    assert not math_reward.is_valid_ground_truth("] )")
    assert math_reward.compute_score(r"answer \boxed{)}", "} )") == 0.0
    assert math_reward.compute_score(r"answer \boxed{42}", "42") == 1.0


def test_dfrac_matches_malformed_frac_ground_truth():
    sol = r"Therefore \boxed{\dfrac{40}{7}}"
    gt = r"\frac{40}7"
    assert math_reward.compute_score(sol, gt) == 1.0


def test_fallback_extracts_numeric_answer_without_boxed():
    sol = "thinking about it response The final value is 28."
    assert math_reward.compute_score(sol, "28") == 1.0


def test_invalid_ground_truth_returns_zero_even_if_model_boxed():
    assert math_reward.compute_score(r"answer \boxed{-6}", "") == 0.0
    assert math_reward.compute_score(r"answer \boxed{42}", "} )") == 0.0


def test_pmatrix_extraction_and_scoring():
    sol = r"The answer is \boxed{\begin{pmatrix} 7 \\ -13 \end{pmatrix}}"
    gt = r"\begin{pmatrix} 7 \\ -13 \end{pmatrix}"
    extracted = math_reward.extract_math_answer(sol)
    assert extracted is not None
    assert "pmatrix" not in extracted.split() or "7" in extracted
    assert extracted != "pmatrix pmatrix"
    assert math_reward.compute_score(sol, gt) == 1.0
