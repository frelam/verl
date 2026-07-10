# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""This package is deprecated — kept for reference.

The current Online DPO architecture uses Uni-Agent Gateway:

    Agent (hermes_entrypoint.py) → Gateway → vLLM (Qwen3-4B)
    Runner (custom_hermes_runner.py) manages sessions & reward

Tools are executed via subprocess in isolated workspaces by the agent
entrypoint — no BaseTool registration needed.

See custom_hermes_runner.py and hermes_entrypoint.py for the current
implementation.
"""
