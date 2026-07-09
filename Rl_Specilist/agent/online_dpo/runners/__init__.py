# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""This package is deprecated.  Tools live in ``../tools/``.

The correct architecture for Online DPO is:

    verl model (ToolAgentLoop) → sandbox tools → observations → model → ...

Tools are registered via ``tool_config.yaml``; no custom agent loops needed.
"""
