"""Pipeline entry points for ``agentic_rl_data`` (plan §3 steps 1-8).

Each script is runnable both ways::

    python -m agentic_rl_data.scripts.build_raw --help
    python agentic_rl_data/scripts/build_raw.py --help

Scripts add the repo root to ``sys.path`` themselves, so direct execution works
without installing the package.  ``build_mixture`` is the single entry point
required by plan §8 acceptance #1.
"""

__all__ = [
    "build_raw",
    "build_mixture",
    "dedup",
    "decontam",
    "filter_by_passrate",
    "filter_rounds",
    "replay_check",
    "report",
    "run_pipeline",
    "export_verl_parquet",
    "verl_reward_bridge",
]
