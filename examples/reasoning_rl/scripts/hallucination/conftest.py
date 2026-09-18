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
"""Put this directory on ``sys.path`` so the adapters can ``import schema``.

The adapters are run both as scripts (``python kk_adapter.py``, where the script's
own directory is already on the path) and by pytest from the repo root (where it
is not).  This hook covers the second case; the reward module uses the equivalent
inline ``sys.path.insert`` fallback.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
