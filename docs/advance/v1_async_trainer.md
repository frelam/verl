# V1 Async Trainer

Last updated: 11/06/2026.

The V1 trainer provides two asynchronous PPO training modes under the standard `verl.trainer.main_ppo` entry point:

- `colocate_async` runs generation and training on the same GPU pool.
- `separate_async` runs generation continuously on standalone rollout GPUs and trains on a hybrid GPU pool. It can optionally lend idle hybrid trainer GPUs to generation.

Both modes use the V1 `TransferQueue`, asynchronous replay buffer, and partial rollout client. This guide explains their execution model, configuration, and tuning.

## Choose a Trainer Mode

Set the mode with `trainer.v1.trainer_mode`.


| Mode                         | Rollout resources                                                  | Typical use                                                                                   |
| ---------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `sync`                       | Hybrid rollout replicas colocated with the trainer                 | Baseline PPO and workloads where strict synchronization is preferred                          |
| `colocate_async`             | Hybrid rollout replicas colocated with the trainer                 | Use warmup batches + partial rollout to accelerate.                                           |
| `separate_async`             | Dedicated standalone rollout replicas plus hybrid trainer replicas | Separate resources without switch and offload costs. More compact and efficient rollout pool. |
| `separate_async` with switch | Same as `separate_async`                                           | Reduce trainer idle time when it's hard to set a perfect rollouter-trainer ratio.             |


`colocate_async` and `separate_async` both enable partial-rollout through `FullyAsyncLLMServerClient`. If generation is aborted during a mode transition, completed tokens are retained and the remaining generation is retried. A resumed trajectory can therefore span multiple model versions.

![v1_trainer_timeline](
https://github.com/Begunner/verl-link/blob/main/v1_trainer.svg?raw=true)

## The V1 Training Loop

All trainer modes process the same number of PPO mini-batches per global step:

```text
mini-batches per PPO epoch
  = data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size
```

The modes differ in where this split happens. `sync` and `colocate_async` sample the full training batch in one controller round, then the actor worker divides it into PPO mini-batches. `separate_async` streams one PPO mini-batch through the controller at a time, overlapping rollout with mini-batch training.

The following invariant makes that value equal to the number of PPO mini-batches in `separate_async`:

```text
data.train_batch_size
  = trainer.v1.separate_async.parameter_sync_step
  × actor_rollout_ref.actor.ppo_mini_batch_size
```

For example, a train batch of 64 with a PPO mini-batch of 16 requires `parameter_sync_step=4`.

## Partial Rollout and Staleness

Partial rollout is part of the V1 async client behavior. With the default `trainer.v1.async_trajectory.single_version_per_trajectory=false`, when a request is aborted:

1. Tokens and log probabilities produced before the abort are retained.
2. The client retries the unfinished trajectory through the load balancer.
3. The resumed request can run with a newer model version.
4. The KV cache is reconstructed for the retained prefix before decoding continues.

This avoids dropping long-running trajectories during rollout/trainer transitions, at the cost of extra prefill work and within-trajectory policy-version changes.

### Single-version trajectories

Set `trainer.v1.async_trajectory.single_version_per_trajectory=true` to reject cross-version resume. A non-migration abort then raises `TrajectoryAbortedError` before partial output is merged or a length stop can hide the abort. After all sibling sessions settle, the prompt group is marked `failure` with reason `abort_discard`; the normal async failure path evicts it and dispatches only the required replacement shortfall. `rollout/aborted_discarded` counts these discarded groups.

Migration aborts remain resumable because repack preserves the same weight version. In contrast, global weight push aborts in both colocated and standalone rollout pools, so enabling this option with `weight_sync.mode=global_push` can discard long trajectories and increase resampling cost. The default remains `false` for backward compatibility. This guarantee is per trajectory `(uid, session_id)`, not per GRPO prompt group.

### Off-policy control

The async replay buffer applies staleness policy to prompt-group age:


| Parameter                                     | Default | Meaning                                                                                                                                                                      |
| --------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trainer.v1.sampler.max_off_policy_threshold` | `8`     | Prompt-age threshold, where age is `current_global_step - prompt_global_steps + 1`                                                                                         |
| `trainer.v1.sampler.max_off_policy_strategy`  | `drop`  | `drop` evicts terminal groups whose age exceeds the threshold; `wait` blocks when an in-flight group reaches it; `observe` only reports threshold violations               |

`observe` performs no staleness eviction or blocking and therefore imposes no staleness upper bound. DAPO filtering, rollout-failure handling, capacity control, and required refill still apply. `training/off_policy/over_threshold_groups` is a latest-state gauge counting outstanding train groups above the observation threshold. In `sync` mode, `observe` has no effect and startup emits a warning.

Keep dispatch age separate from generation-version metadata: `prompt_global_steps` drives the age-based sampler and capacity rules, while `min_global_steps` and `max_global_steps` record the actual replica weight versions used to generate a trajectory.

Monitor both forms of off-policy behavior:

- `training/off_policy/trajectory_spans/*`: number of model versions used within one trajectory. `1` means that the trajectory was generated entirely with one version.
- `training/off_policy/trajectory_staleness/*`: gap between the newest version used by a trajectory and the current training version.
- `training/off_policy/trajectory_staleness_worst/*`: gap between the oldest version used by a trajectory and the current training version.
- `training/off_policy/over_threshold_groups`: current number of outstanding groups beyond the configured prompt-age threshold, including in `observe` mode.

### Bounded asynchronous dispatch

`trainer.v1.async_trajectory.max_outstanding_groups` limits scheduling depth rather than engine memory. It counts whole train-partition prompt groups in all four states:

```text
outstanding = pending + running + finished + failure
```

The default `null` preserves legacy warmup/single-batch prepare and 1:1 eviction refill without creating a dispatch ledger. An explicit value must be at least `data.train_batch_size`; setting it in `sync` mode fails at startup.


| Parameter                                    | Default  | Behavior when the cap is enabled                                      |
| -------------------------------------------- | -------- | --------------------------------------------------------------------- |
| `capacity_policy`                            | `hybrid` | `backpressure`, `drop_oldest`, or `hybrid`                            |
| `high_watermark_ratio`                       | `0.8`    | Optional prefetch pauses at `ceil(cap × ratio)` under hysteresis      |
| `low_watermark_ratio`                        | `0.5`    | Paused optional prefetch resumes at `floor(cap × ratio)` or below     |

The policies differ only when the cap is set:

- `backpressure` applies high/low-watermark hysteresis to optional prefetch and never performs capacity eviction.
- `drop_oldest` ignores watermark pausing and permits oldest-terminal-group eviction when synchronized outstanding state exceeds the cap.
- `hybrid` combines both behaviors.

Optional prefetch can dispatch multiple batches up to the remaining hard-cap slots, rounded down to `data.gen_batch_size` (or `train_batch_size` when unset). It does not wrap into the next dataloader epoch. Watermarks are checked before optional dispatch, so a cap equal to one train batch can still dispatch its initial required batch. Required refill for the current batch is based on the shortfall in `sampleable + pending + running`; it bypasses watermark pauses but never bypasses the hard cap. Required refill may wrap epochs and records the borrowed prompts so later optional prefetch is reduced accordingly.

Capacity eviction removes only complete terminal groups, oldest `(prompt_global_steps, uid)` first, and never splits `rollout.n`. It does not trigger an immediate 1:1 replacement. Dispatch-budget checks add no new blocking wait, although existing replay sampling and separate-async switch waits remain. Selecting a group clears its prompt metadata and releases scheduling capacity, but trajectory fields live until step cleanup; consequently this cap is not a TransferQueue physical-memory quota.

With bounded dispatch enabled, checkpoints also save `dispatch_ledger.pt`. The ledger tracks each dataloader epoch's dispatched, trained, prefetched, borrowed, and optional-skipped prompt counts plus prompt-to-epoch ownership. See [Checkpoint and Validation Behavior](#checkpoint-and-validation-behavior) for restore semantics.

### Per-replica admission control

`actor_rollout_ref.rollout.max_inflight_requests_per_replica` sets a load-balancer admission cap. The default `null` is unlimited. The unit is an active `generate` request on each replica, not a prompt group: `rollout.n` creates multiple sessions and a multi-turn trajectory issues one request per turn. This is independent of the engine's decode-time `max_num_seqs` limit.

For an unleased request, a full sticky replica is reselected within the same-version, serving, non-full replica set. A trajectory holding a per-replica-pull lease stays on its bound replica and waits instead of being rebalanced. Only when no eligible replica has capacity does the client retry with bounded exponential sleep and jitter; the load-balancer actor never sleeps. `full_determinism=true` cannot be combined with this cap.

### Built-in group samplers

The default sampler remains FIFO. Built-ins are selection-only mixins composed with the correct sync or async replay-buffer base, so staleness, DAPO, failure, capacity, and admission behavior remain in the common path. Select one with:

```yaml
trainer:
  v1:
    sampler:
      custom_sampler:
        path: verl.trainer.ppo.v1.samplers
        name: Mixed  # FIFO | RewardVariance | Mixed | Length
      sampler_kwargs:
        fifo_ratio: 0.5
        age_bonus: 0.1
        short_first: true
        reward_metric: null
```


| Name             | Selection behavior                                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FIFO`           | Sort by `(prompt_global_steps, uid)`; equivalent to the default order                                                                                     |
| `RewardVariance` | Rank by `population_reward_stddev + age_bonus × age`, then FIFO/uid ties                                                                                  |
| `Mixed`          | Take `floor(fifo_ratio × batch_size)` FIFO groups, then fill without duplicates by reward-variance priority                                               |
| `Length`         | Sort each group by the sum of its trajectory-tag `response_len` values; `short_first=true` selects shortest first, otherwise longest first, with FIFO ties |

`RewardVariance` and `Mixed` use canonical pre-KL rewards by default, or the reward field named by `sampler_kwargs.reward_metric`, and cache immutable finished-group statistics. They require rule-based reward or an independent reward resource pool; a colocated reward model computes rewards only after sampling and is rejected at startup. Missing reward values or `response_len` metadata raise errors rather than silently changing order. External custom sampler loading and its constructor contract are unchanged.

### Prompt refilling in replay buffer and checkpoint recovery

Both async modes use `ReplayBufferAsync`, which samples prompt groups from the TransferQueue. During sampling, it repeatedly synchronizes TransferQueue metadata and classifies terminal groups. A training group is evicted when any of the following applies:

- Its prompt age exceeds `max_off_policy_threshold` while `max_off_policy_strategy=drop`.
- It is removed by the configured DAPO group filter.
- Its rollout finishes with `failure`.

With legacy defaults, eviction reasons are unioned so a group matching multiple conditions is removed once, and evicting `k` groups fetches exactly `k` replacements. Enabling single-version trajectories or bounded dispatch switches to gap-based refill: replacements are dispatched only for the current batch shortfall after counting sampleable and in-flight groups. This prevents speculative over-refill; capacity eviction itself never requests a replacement. Refill can repeat if required replacement groups also fail or are filtered.

With `max_off_policy_strategy=wait`, stale groups are not evicted or refilled. Instead, sampling blocks when an in-flight prompt reaches the threshold, allowing it to finish and remain trainable. With `observe`, only DAPO-filtered and failed groups are evicted and replaced. Refill applies to the training partition; validation does not perform staleness, filtering, or failure refill.

When async trainer saves a checkpoint, pending and running groups have already consumed dataloader entries, so they are reissued from their saved prompts during recovery when TransferQueue checkpointing is available. Finished groups and their completed trajectories are restored as-is and remain available for sampling; they are not regenerated.

### Separate Async training Granularity and Decoupled PPO

`separate_async` uses a mini-batch training granularity, trying to overlap rollout and training in the same step. When a mini-batch is samplable, it is trained at once. This advances the timing of `update_weights` and reduces staleness.

When `separate_async` recomputes old log probabilities, actor weights may change between controller-level mini-batches. To keep the same `pi_old` for the entire `parameter_sync_step` cycle, the first mini-batch copies `pi_old` to CPU and computes with the weights already on GPU. Before each later mini-batch, the trainer:

1. Copies the current updated weights to CPU.
2. Restores `pi_old` to GPU and computes old log probabilities.
3. Restores the current weights to GPU and clears their temporary CPU copy.

For `N` mini-batches, this results in `N` `save_model_to_cpu` calls and `2 * (N - 1)` `restore_model_from_cpu` calls. The old-policy weights themselves are saved once and restored `N - 1` times; the remaining calls preserve the current weights around old-log-probability computation. For example, four mini-batches make four save calls and six restore calls, of which one save and three restores transfer `pi_old`.

These transfers and old log prob computings are skipped when `algorithm.rollout_correction.bypass_mode=True`.


## Separate Async Step Switching (experimental)

Enable step switching with:

```bash
trainer.v1.separate_async.hybrid_rollout.enable_switch=True
```

The switch addresses a specific idle window: after a PPO step finishes, the trainer may have to wait for standalone rollout to produce enough sampleable groups for the next step. During that window, the trainer's hybrid replicas can join the standalone load balancer and help generate samples.

![separate_async_switch_timeline](
https://github.com/Begunner/verl-link/blob/main/sepa_switch.svg?raw=true)

The upper timeline shows `separate_async` without switching; the lower timeline shows hybrid GPUs joining rollout during idle windows when switching is enabled.

### Switch to trainer threshold

The trainer converts `switch_threshold_ratio` into a number of sampleable groups:

```text
target = round(switch_threshold_ratio × train_batch_size)
threshold = clamp(target, one_mini_batch, train_batch_size)
```

`switch_threshold_ratio` defines the target number of prompt groups ready for sampling before switching to trainer. At the end of a step, if the next step's buffer already meets the target or is expected to reach it soon without hybrid assistance (that is, the estimated benefit of lending does not exceed the measured switch cost), the hybrid replicas remain in trainer mode. If the buffer is below the target and switch is enabled, the hybrid replicas enter rollout mode and switch back to trainer mode once the target is reached. The one-mini-batch floor guarantees that at least one mini-batch is ready to train immediately.

### Adaptive threshold

Since the sample-length distribution can evolve throughout RL training, the resource balance required by rollout and training can shift over time. The switch threshold therefore adapts to observed trainer idle time instead of assuming a fixed optimal resource split.

With `adaptive_switch_threshold=True`, the threshold ratio reacts to observed sample wait:

- After `switch_threshold_release_steps` consecutive idle steps, increase the ratio by `switch_threshold_step_up`.
- After `switch_threshold_release_steps` consecutive non-idle steps, decrease the ratio by `switch_threshold_step_down`.

The release interval applies in both directions and prevents noisy steps from changing the threshold.

### Switch configuration

All settings under `trainer.v1.separate_async.hybrid_rollout` are ignored unless `enable_switch=True`.


| Parameter                        | Default | Description                                                                          |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `enable_switch`                  | `false` | Allow hybrid trainer replicas to help rollout between steps                          |
| `switch_threshold_ratio`         | `0.4`   | Target sampleable fraction before hybrid replicas are reclaimed; must be in `(0, 1]` |
| `adaptive_switch_threshold`      | `true`  | Adapt the reclaim threshold from observed trainer idle time                          |
| `switch_threshold_step_up`       | `0.05`  | Ratio increase after sustained idle                                                  |
| `switch_threshold_step_down`     | `0.03`  | Ratio decrease after sustained calm                                                  |
| `switch_threshold_release_steps` | `2`     | Consecutive idle or calm steps required before adjustment                            |
| `switch_cost_window_size`        | `3`     | Number of recent transition costs used by the decision                               |


Temporarily, step switching cannot be combined with rollout PD disaggregation.

## Standalone Per-Replica Weight Pull

`separate_async` defaults to `weight_sync.mode=global_push`, which retains the synchronous abort/push/resume path. Standalone rollout can instead publish immutable snapshots and update each replica independently:

```yaml
trainer:
  v1:
    trainer_mode: separate_async
    separate_async:
      weight_sync:
        mode: per_replica_pull
        cache_versions: 2
        retry_interval_s: 5
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: nixl  # or kimi_ckpt_engine
```

This mode is restricted to the standalone pool in `separate_async`. It requires NIXL or KIMI and a validated vLLM/SGLang adapter; startup rejects Mooncake/other backends, TRT-LLM, hybrid rollout switching, non-`separate_async` modes, and `full_determinism=true`. Colocated and hybrid-switch weight updates retain their existing global-push behavior.

Publication retains the existing trainer `on_step_end` and `parameter_sync_step` cadence. The actor exports a version at most once per checkpoint manager. `WeightSnapshotStore` privately clones and checksums every tensor, then atomically makes the complete version visible; a failed export is not published. Readers pin an immutable version and receive defensive tensor copies. The newest `cache_versions` snapshots are retained, but any actively referenced older version remains pinned until release. `cache_versions` must be at least two.

The standalone load balancer tracks `latest`, per-replica `local` versions, and `serving → draining → updating → serving` state:

1. A new trajectory acquires a `(uid, session_id)` lease on a latest-available serving replica before its agent loop starts. The lease pins replica and weight version across every turn and is released in `finally`; different sessions in one prompt group are not a group-wide lease.
2. After a newer snapshot is published, a stale serving replica enters `draining` and accepts no new trajectory leases. Existing leased trajectories continue on their original replica/version.
3. When both generate calls and trajectory leases reach zero, the replica enters `updating` and installs one committed snapshot. NIXL sends the stored snapshot to that replica without exporting it again; KIMI uses a direct CPU snapshot RPC without a collective broadcast. Success updates the actual local version and resumes serving when caught up. Failure retains the old version in `draining` and retries after `retry_interval_s`.

This is drain-then-install, not an engine-internal dual-version hot swap. It avoids the standalone pool's global abort, but a long-lived trajectory can keep its replica draining. Weight-affinity leases are not failure-detection leases and do not provide the running-group recovery described under D2 below.

## Repack and Real Replica Metrics

Repack is disabled by default. Observation and Best-Fit planning require structured engine metrics; migration is a separate, also default-off switch:

```yaml
trainer:
  v1:
    async_trajectory:
      repack:
        enable: false
        interval_s: 5
        kv_cache_threshold: 0.99
        roofline_batch_size: null
        migration:
          enable: false       # false means collect and plan only (dry-run)
          timeout_s: 30
          fallback_to_lb: false
actor_rollout_ref:
  rollout:
    enable_stats: false
    stats_interval_s: 5
    disable_log_stats: true  # must be false when enable_stats=true
```

To run repack, set `repack.enable=true`, `rollout.enable_stats=true`, and `rollout.disable_log_stats=false`. V1 repack runs on the standalone manager in `separate_async`; `sync` rejects it and `colocate_async` warns that it is a no-op. `migration.enable=true` additionally requires `repack.enable=true`. With stats disabled there is no repack polling RPC.

### Metrics and idleness

The current vLLM and SGLang adapters fetch each replica's Prometheus `/metrics` endpoint and construct snapshots containing KV-cache utilization, running/waiting request counts, weight version, PD role, timestamp, and serving state. Missing, timed-out, incomplete, out-of-order, non-serving, or stale samples are not treated as idle; failed collection clears that replica's history instead of reusing an old sample. Snapshot freshness is bounded by `max(2 × stats_interval_s, 2 × repack.interval_s)`.

A replica is idle only when two complete, fresh, serving snapshots have the same weight version and role, there is no waiting request, and:

```text
current_kv < min(kv_cache_threshold, previous_kv)
current_running_requests < request_budget
request_budget = roofline_batch_size or max_inflight_requests_per_replica or max_num_seqs
```

The `kv_idle` policy also exposes these structured signals to the experimental dynamic resource controller. Its request-count fallback controls only legacy hybrid activate/deactivate and never authorizes repack migration.

### Best-Fit planning and migration limits

Sources are considered from lowest to highest KV load. The planner chooses the serving destination with the fullest projected KV load that still fits, subject to cumulative request limits and deterministic server-ID ties. Sources and destinations must have the same weight version and PD role; self-moves, released destinations, and move chains are excluded. Planning is an estimate over recent snapshots, not a reservation by itself.

The planner can additionally enforce exact `num_tokens`/`token_budget` fields when an adapter supplies them. The current vLLM/SGLang Prometheus adapters do not populate those token fields, so their KV-fit calculation uses utilization estimates. Before a directed migration, the load balancer atomically rechecks destination serving state, identical weight version, and real admission-request capacity, then reserves capacity and transfers sticky/attempt state and any trajectory lease. It does not infer exact token occupancy from request counts.

With `migration.enable=false`, ticks stop after producing a dry-run plan. With migration enabled:

- The source is drained, target capacity is reserved before abort, and the source request is aborted with reason `migration` and `reset_prefix_cache=false`. The client re-prefills the retained prompt plus response prefix on the same-version destination; KV cache itself is not transferred.
- A request that completes before abort is left untouched. If reservation fails, `fallback_to_lb=false` skips that request; opt-in `fallback_to_lb=true` allows the normal same-version load-balancing resume path.
- vLLM and SGLang preserve partial output and reject unsafe response-token/logprob misalignment. Requests without registered engine identity cannot be prepared for directed migration. TRT-LLM and other adapters fail fast when migration is requested.
- A persistent coordinator provides periodic collect → idle → plan → optional execute ticks, single-flight overlap skipping, exponential failure backoff, weight-sync exclusion, and stop/cancellation cleanup. Missing metrics, stale state, or version conflicts produce no migration rather than a cross-version fallback.

Migration adds re-prefill work and can invalidate prefix reuse. It is not a zero-cost operation, and the configuration does not claim the Laminar paper's migration latency or throughput results.

## Configuration

### Colocate async

Add the following overrides to an existing V1 PPO launch:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=colocate_async \
trainer.v1.colocate_async.num_warmup_batches=1
```

The warmup batch starts generation before the first training step, reducing the initial empty-buffer wait.

### Separate async

The following example uses two trainer nodes and two standalone rollout nodes:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=separate_async \
trainer.nnodes=2 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.nnodes=2 \
actor_rollout_ref.rollout.n_gpus_per_node=8 \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
trainer.v1.separate_async.parameter_sync_step=4 \
trainer.v1.separate_async.num_warmup_batches=1
```

`separate_async` requires a non-`naive` checkpoint-engine backend such as `nccl`, `nixl`, or `mooncake` for standalone rollout weight synchronization.

To enable step switching, add:

```bash
trainer.v1.separate_async.hybrid_rollout.enable_switch=True \
trainer.v1.separate_async.hybrid_rollout.adaptive_switch_threshold=True
```

## Feature Support and Fail-Fast Validation

The matrix describes implemented code paths and configuration checks, not completed hardware certification:


| Feature                       | vLLM                       | SGLang                     | TRT-LLM                    | Mode/backend constraints                                                        |
| ----------------------------- | -------------------------- | -------------------------- | -------------------------- | ------------------------------------------------------------------------------- |
| Single-version abort/discard  | Client path                | Client path                | Client path                | Async modes; global push warns about discard cost                               |
| `observe`, bounded dispatch   | Trainer/TransferQueue path | Trainer/TransferQueue path | Trainer/TransferQueue path | Async modes; explicit outstanding cap in `sync` fails                           |
| Per-replica admission         | Built-in LB path           | Built-in LB path           | Built-in LB path           | Positive integer or `null`; incompatible with full determinism                  |
| Immutable per-replica pull    | NIXL/KIMI code path        | NIXL/KIMI code path        | Startup error              | Standalone `separate_async`; no hybrid switch or full determinism               |
| Structured stats/idle/repack  | Prometheus snapshot path   | Prometheus snapshot path   | No snapshot adapter        | Standalone `separate_async`; colocate warns/no-ops, sync fails                  |
| Directed trajectory migration | Implemented engine path    | Implemented engine path    | Startup error              | Requires repack + stats; incompatible with full determinism                     |

Startup validation rejects invalid ranges (`0 <= low_watermark_ratio < high_watermark_ratio <= 1`, cap below one train batch, non-positive admission cap, `fifo_ratio` outside `[0,1]`, negative `age_bonus`, or `cache_versions < 2`). It also rejects reward-variance sampling with a colocated reward model, per-replica pull outside `separate_async` or without NIXL/KIMI, pull combined with hybrid switching, migration without repack/stats, migration on TRT-LLM or an unknown adapter, and `enable_stats=true` with `disable_log_stats=true`.

A YAML custom router must declare every enabled capability in its `capabilities` list: `admission_control` for the admission cap, `version_registry` and `trajectory_lease` for per-replica pull, and `migration` for directed migration. Missing declarations fail at initialization; declarations alone do not validate the implementation.

### Defaults, rollback, and verification boundary

All behavior-changing Laminar controls are off or legacy-compatible by default. Roll back with:

```text
trainer.v1.async_trajectory.single_version_per_trajectory=false
trainer.v1.async_trajectory.max_outstanding_groups=null
actor_rollout_ref.rollout.max_inflight_requests_per_replica=null
trainer.v1.sampler.custom_sampler.path=null
trainer.v1.sampler.custom_sampler.name=null
trainer.v1.separate_async.weight_sync.mode=global_push
trainer.v1.async_trajectory.repack.enable=false
trainer.v1.async_trajectory.repack.migration.enable=false
trainer.v1.async_trajectory.repack.migration.fallback_to_lb=false
actor_rollout_ref.rollout.enable_stats=false
```

The default capacity policy/watermarks and sampler kwargs are inert while their corresponding cap or built-in sampler is unset. When returning from per-replica pull to global push, restart/reconfigure through the normal global weight-sync procedure and account for in-flight leased trajectories; do not treat a configuration edit as an atomic live migration. Disabling migration retains harmless stats/dry-run observation if repack and stats remain enabled.

Windows/CPU unit tests and mocked engine tests can validate pure ordering, watermark, ledger, routing-state, metrics-parsing, and coordinator logic. They cannot prove GPU/RDMA weight integrity or liveness, real NIXL/KIMI installation, continuous generation during multi-replica drain/update, vLLM/SGLang abort and partial token/logprob/routed-expert contracts, migration correctness under engine races, or end-to-end throughput. Those claims require the corresponding GPU, RDMA, engine integration, and benchmark environments; no such new performance result is asserted here.

## Observability and Tuning

Start with the following timing metrics:

- `timing_s/gen`: trainer time spent waiting for the next trainable train-batch. (Also means idle time for separate_async's hybrid gpus)
- `timing_s/update_actor`: actor update time.
- `timing_s/update_weights`: standalone weight synchronization time in `separate_async`.
- `timing_s/switch_wait`: time during which lent hybrid replicas help fill the switch-to-trainer threshold. (It is not idle.)
- `timing_s/switch_to_rollout`: trainer-to-rollout transition time, including load-balancer registration, sticky-cache clearing, hybrid weight update, and generation resume.
- `timing_s/switch_to_trainer`: rollout-to-trainer transition time, including load-balancer removal, request abort, and hybrid replica sleep.

When switching is enabled, inspect:

- `separate_async/switch/threshold_ratio`
- `separate_async/switch/sample_wait_seconds`
- `separate_async/switch/idle`
- `separate_async/decision/sampleable_count`
- `separate_async/decision/remaining`
- `separate_async/decision/benefit_seconds`
- `separate_async/decision/effective_switch_cost_seconds`
- `separate_async/decision/should_switch_to_rollout`

### Laminar metrics

Metrics are aggregated by kind: counters are reported as per-step interval deltas, state values as latest gauges, and durations as delta `sum`/`count` plus a `last` value. Missing engine statistics omit the corresponding engine-metric keys rather than fabricating zero load.

- Scheduling and staleness: `rollout/outstanding_groups`, `rollout/{pending,running,finished,failure}_groups`, `rollout/oldest_running_age`, `rollout/capacity_dropped_groups`, `rollout/dispatch_skipped_steps`, `rollout/watermark/{high,low}`, `rollout/aborted_discarded`, and `training/off_policy/over_threshold_groups`.
- Admission and weight versions: `rollout/inflight_requests/{server_id}`, `rollout/admission_wait_seconds`, `rollout/admission_wait_count`, `rollout/version_lag/{mean,max}`, `rollout/draining_servers`, `rollout/updating_servers`, `rollout/active_trajectory_leases`, `rollout/drain_wait_seconds`, `rollout/drain_wait_count`, `rollout/update_duration_seconds`, `rollout/update_duration_count`, and `rollout/update_failures`.
- Engine and idle snapshots: `rollout/kv_cache_util/{mean,max}`, `rollout/num_running_reqs`, `rollout/num_waiting_reqs`, `rollout/stats_age_seconds`, `rollout/idle_replicas`, `rollout/idle_kv_cache_util`, and `rollout/idle_req_count`.
- Coordinator and plans: `rollout/repack/{trigger_count,skipped_overlap,plan_empty,backoff_active,planned_sources,planned_destinations,planned_moves,estimated_released_replicas,migration_attempts,migration_successes,migration_failures,last_run_age_seconds}` and `rollout/repack/plan_duration_ms`. Stage timings use `rollout/repack/{collect,plan,migration,tick}_duration_seconds/{sum,count,last}`.
- Request migration: `rollout/migration/{requests_total,success,failed,reprefill_tokens,duration_seconds,fallback_to_lb}`.

`rollout/running_groups` and `rollout/oldest_running_age` are especially important while running-group TTL recovery remains unimplemented. Per-replica `inflight_requests` are gauges; admission, migration, drain, and update counters are interval deltas, not lifetime totals.

Practical tuning order:

1. Balance trainer and standalone rollout resources before enabling switching.
2. Set `parameter_sync_step` from the required batch-size invariant.
3. Choose `max_off_policy_threshold` and `drop`, `wait`, or `observe` from the workload's policy-lag tolerance; use bounded dispatch when `observe` needs an independent scheduling-depth safety net.
4. Add admission and outstanding caps conservatively, then validate sampler behavior with FIFO before trying Length or reward-priority ordering.
5. Enable switching when `timing_s/gen` shows sustained trainer idle time. Roll out repack in stats → dry-run → migration order rather than enabling migration directly.

## Checkpoint and Validation Behavior

When the installed TransferQueue supports checkpointing, V1 async checkpoints persist its state alongside model and dataloader state. Finished samples are restored directly. Old partial trajectories for pending/running groups are cleared, and their persisted prompts are reissued after resume so already-fetched dataloader entries are not lost.

With bounded dispatch, `dispatch_ledger.pt` is saved beside the dataloader and TransferQueue state and is mandatory when that checkpoint is loaded. Restore loads dataloader state, then epoch/prompt ownership in the ledger, then TransferQueue outstanding state. Reissuing pending/running prompts starts new generation attempts but does not fetch the dataloader again or record new ledger dispatches, preventing duplicate epoch consumption. Required-refill borrowing and subsequent optional-prefetch deductions therefore survive restart. A bounded-dispatch checkpoint without its ledger fails instead of silently risking duplicate or skipped prompts.

**D2 remains manual-recovery-only.** Running prompt groups have no TTL, heartbeat, or crash-detection lease. An AgentLoop worker crash can leave a group `running` forever; `wait` can then block indefinitely, `drop` does not reclaim it, and the outstanding cap cannot evict a nonterminal group. Per-replica-pull trajectory leases provide weight affinity, not failure detection. Monitor `rollout/running_groups` and `rollout/oldest_running_age`; if a group stops progressing, manually restart/recover from a valid checkpoint. There is no automatic timeout, requeue, or live partial-token recovery in this version.

Validation shares the same AgentLoop and rollout server pool with unfinished training trajectories. Those partial trajectories continue running alongside validation requests, so `timing_s/testing` includes the contention and rollout capacity they consume rather than measuring validation generation in isolation.

In `separate_async`, validation makes hybrid replicas available for rollout if they are currently in trainer mode.

## Benchmark

The measurements below are the existing baseline V1 mode/switch results. They do not evaluate the Laminar-inspired controls added in this guide and must not be read as evidence for paper-level repack, weight-pull, or migration throughput.

### V1 trainer all modes

The three modes were compared over their first 150 steps with the same four-node budget. `sync` and `colocate_async` used all four nodes as hybrid resources, while `separate_async` used two hybrid trainer nodes and two standalone rollout nodes.

- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP4) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768
- Decoupled PPO enabled


| Mode             | Resource split          | 150-step training time | Aggregate tokens/s | Mean response length |
| ---------------- | ----------------------- | ---------------------- | ------------------ | -------------------- |
| `sync`           | 4 hybrid                | 22.79 h                | 12,053             | 12,720               |
| `colocate_async` | 4 hybrid                | 14.72 h (-35.4%)       | 18,852 (+56.4%)    | 12,854               |
| `separate_async` | 2 hybrid + 2 standalone | 14.10 h (-38.1%)       | 18,829 (+56.2%)    | 12,288               |


![v1_modes_quality_step](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_quality_step.png?raw=true)

![v1_modes_timing_components](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_timing_components.png?raw=true)

![v1_modes_offpolicy](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_offpolicy.png?raw=true)

![v1_modes_policy_alignment](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_policy_alignment.png?raw=true)


### Separate_async switching

The step switch was evaluated for the first 100 steps of two otherwise identical runs:

- 3 × 8 H100 GPUs: two trainer/hybrid nodes and one standalone rollout node.
- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP8) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768
- Decoupled PPO disabled


| Mode        | Resource split          | 100-step training time | Aggregate tokens/s | Mean response length | Mean reward |
| ----------- | ----------------------- | ---------------------- | ------------------ | -------------------- | ----------- |
| `no-switch` | 2 hybrid + 1 standalone | 13.15 h                | 14,604             | 13,343               | 0.7755      |
| `switch`    | 2 hybrid + 1 standalone | 11.79 h (-10.3%)       | 16,445 (+12.6%)    | 13,478               | 0.7762      |

![switch_quality_step](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_quality_step.png?raw=true)

![switch_timing_components](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_timing_components.png?raw=true)

![switch_offpolicy](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_offpolicy.png?raw=true)
