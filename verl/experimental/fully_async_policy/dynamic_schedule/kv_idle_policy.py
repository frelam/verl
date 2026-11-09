# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""KV-idleness adapter for the legacy DynamicResourceController."""

from .base import DynamicScheduleContext, register_policy
from .default_policy import DefaultDynamicSchedulePolicy


@register_policy("kv_idle")
class KVIdleDynamicSchedulePolicy(DefaultDynamicSchedulePolicy):
    """Use structured KV idleness when available, otherwise keep legacy counters.

    This policy only controls hybrid activate/deactivate. It never drives repack
    migration from request-count fallbacks.
    """

    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext) -> bool:
        if ctx.replica_metrics:
            return is_hybrid_active and bool(ctx.idle_replicas)
        return super().should_deactivate(global_steps, is_hybrid_active, ctx)

    def should_activate_after_step(
        self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext
    ) -> bool:
        if ctx.replica_metrics and ctx.idle_replicas:
            return False
        return super().should_activate_after_step(global_steps, is_hybrid_active, ctx)
