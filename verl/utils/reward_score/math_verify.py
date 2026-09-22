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

import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()

# Raw model outputs here can be 16k-token chain-of-thought.  Scoring only
# needs the final answer, which the prompt format pins to the end of the
# response, and math_verify tries extraction matches rightmost-first, so
# parsing only a tail window keeps semantics while bounding regex/sympy
# work per sample.
_PRED_PARSE_MAX_CHARS = 8192


def _silence_math_verify_logs():
    """Process-pool initializer: keep math_verify's loggers quiet in workers.

    On parse timeout math_verify logs the *entire* prediction at WARNING
    ("Timeout during parsing: {pred}") -- with 16k-token CoT outputs this
    floods fully-async training logs, since pool-worker stderr lands there.
    """
    logging.getLogger("math_verify").setLevel(logging.CRITICAL)


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_silence_math_verify_logs,
                )
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    # Tail window only (see _PRED_PARSE_MAX_CHARS).  Keep the default finite
    # parsing_timeout on purpose rather than parsing_timeout=None: the inner
    # signal.alarm is what actually frees a worker stuck on a pathological
    # input -- future.result(timeout=...) in compute_score() only unblocks
    # the caller, it cannot reclaim the worker process.  The timeout's
    # WARNING log is silenced by _silence_math_verify_logs above.
    extracted_pred = parse(model_output[-_PRED_PARSE_MAX_CHARS:], pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
