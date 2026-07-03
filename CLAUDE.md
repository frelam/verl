# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) v0.8.0.dev — ByteDance Seed team's open-source RL training framework for LLMs. Implements the HybridFlow programming model (EuroSys 2025) to decouple computation from data dependencies in RLHF/GRPO/PPO/DAPO training. Scales to 671B models with FSDP/FSDP2/Megatron-LM training + vLLM/SGLang inference, on NVIDIA/AMD/Ascend NPU.

## Development Commands

```bash
# Install (pick one inference engine)
pip install -e .[test,vllm]        # with vLLM
pip install -e .[test,sglang]      # with SGLang

# Lint & format (via pre-commit)
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg

# Run specific hook
pre-commit run ruff --all-files

# Tests
pytest tests/ -k "_on_cpu"                        # CPU-only tests
pytest tests/ --ignore-glob='*_on_cpu.py'          # GPU tests
pytest tests/utils/reward_score/test_math_reward.py  # single test file
pytest tests/utils/reward_score/test_math_reward.py -k "test_something"  # single test case

# Build docs
cd docs && pip install -r requirements-docs.txt && make clean && make html

# Generate resolved trainer config
python scripts/print_cfg.py
```

- **Ruff**: line length 120, rules E/F/UP/B/I/G. Ignores F405, F403, E731, B007, UP032, G004.
- **Mypy**: blanket `ignore_errors = true`; strict checking only for `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager.*`.
- **Tests**: mirror `verl/` namespace layout. Files named `*_on_cpu.py` = CPU only. `tests/special_*` directories are for multi-GPU, end-to-end, NPU, sanity, and standalone tests.

## Architecture

### Hybrid Controller Model

The framework uses a **single-controller / multi-worker** pattern built on Ray:

```
Driver (CPU process)
  └─ TaskRunner (Ray actor)
       └─ RayPPOTrainer.fit()
            ├─ RayWorkerGroup[Rollout]    → generate sequences (vLLM/SGLang)
            ├─ RayWorkerGroup[Actor]      → compute log_probs, update policy (FSDP/Megatron)
            ├─ RayWorkerGroup[Critic]     → compute values, update value function
            ├─ RayWorkerGroup[RefPolicy]  → compute reference log_probs (KL penalty)
            └─ RayWorkerGroup[RewardModel]→ reward model scoring
```

### Training Loop (`verl/trainer/ppo/ray_trainer.py` → `RayPPOTrainer.fit()`)

Each step:
1. **Generate** — rollout workers produce responses via vLLM/SGLang
2. **Repeat & Merge** — replicate prompts × `rollout.n`, union with generated output
3. **Compute Rewards** — reward manager scores responses (rule-based or reward model)
4. **Compute Old Log Probs** — actor computes log probs of its own generations
5. **Compute Ref Log Probs** — reference model (frozen) for KL penalty
6. **Compute Values** — critic if using GAE advantage estimator
7. **Advantage Estimation** — GAE, GRPO, REINFORCE++, RLOO, etc. (`core_algos.py`)
8. **Update Critic** — MSE on value function
9. **Update Actor** — clipped PPO surrogate objective (or GRPO/DAPO variant)
10. **Weight Sync** — `checkpoint_manager.update_weights()` pushes updated weights back to rollout replicas

### Data Protocol (`verl/protocol.py`)

`DataProto` is the universal data transfer object — a PyTorch `TensorDict` (`batch`) + `dict[str, np.ndarray]` (`non_tensor_batch`) + `dict` (`meta_info`). All inter-worker communication uses this protocol.

### Worker Architecture (Two Paths)

Controlled by `trainer.use_legacy_worker_impl`:
- **Legacy** (`"enable"`/`"auto"`): Separate worker classes per role — `AsyncActorRolloutRefWorker`, `CriticWorker` (from `fsdp_workers.py` or `megatron_workers.py`)
- **New Engine** (`"disable"`, default): Unified `ActorRolloutRefWorker` + `TrainingWorker` (from `engine_workers.py`), using pluggable `BaseEngine` abstraction

### Role Types (`verl/trainer/ppo/utils.py`)

`Actor`, `Rollout`, `ActorRollout` (hybrid), `Critic`, `RefPolicy`, `RewardModel`, `ActorRolloutRef`, `TeacherModel`

### Reward System

Two layers:
1. **Reward Manager** (`verl/workers/reward_manager/`) — orchestrates how rewards are computed (per-sample naive, batched, DAPO, PRIME). Registered via `@register("name")` decorator.
2. **Reward Functions** (`verl/utils/reward_score/`) — the actual scoring logic, dispatched by `default_compute_score()` in `__init__.py` based on `data_source` string (e.g., `"openai/gsm8k"` → `gsm8k.compute_score()`, `"lighteval/MATH"` → `math_reward.compute_score()`).

Custom reward functions can be loaded externally via `custom_reward_function.path` in config.

### Configuration System

Hydra + OmegaConf with structured dataclass configs in `verl/trainer/config/`:
- **Entry point**: `ppo_trainer.yaml` (defaults chain loads actor, critic, rollout, reward, algorithm, model, data sub-configs)
- **Override**: any Hydra-style CLI override, e.g. `algorithm.adv_estimator=grpo data.train_batch_size=256`
- **Algorithm config**: `verl/trainer/config/algorithm.py` — `AlgoConfig`, `KLControlConfig`, `FilterGroupsConfig`
- **Generated configs**: `_generated_ppo_trainer.yaml` — the fully resolved config (auto-generated, don't edit manually)

### Dataset Pipeline

Raw Parquet files → `RLHFDataset` (`verl/utils/dataset/rl_dataset.py`) → tokenize + apply chat template → `StatefulDataLoader` → `DataProto`

Parquet schema expected:
```
data_source, prompt (chat format), ability, reward_model ({style, ground_truth}), extra_info
```

Preprocessing scripts: `examples/data_preprocess/`

## Training Invocation

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

See `examples/ppo_trainer/` for reference scripts. Training recipes for specific algorithms (GRPO, DAPO, GSPO, RLOO, CRPO, etc.) live under `examples/<algo>_trainer/`.

## Key Directories

| Directory | Purpose |
|---|---|
| `verl/trainer/ppo/` | Core training loop, advantage computation, reward loading |
| `verl/workers/` | Distributed worker implementations (FSDP, Megatron, engine) |
| `verl/single_controller/ray/` | Ray-based distributed orchestration layer |
| `verl/utils/reward_score/` | Rule-based and model-based reward functions |
| `verl/utils/dataset/` | Dataset loading and preprocessing |
| `verl/trainer/config/` | Hydra YAML configs and Python dataclass configs |
| `verl/models/` | Model registry and weight loading for HF/Megatron models |
| `verl/experimental/` | Experimental features (agent loop, one-step off-policy, VLA) |
| `examples/data_preprocess/` | Data preprocessing scripts for various datasets |
| `scripts/` | Utility scripts (config gen, checkpoint conversion, diagnostics) |
| `Rl_Specilist/` | Training recipes and docs (math, muon, agent, offpolicy) |

## FSDP Worker Details

`verl/workers/fsdp_workers.py` is the main worker file for FSDP-based training. It defines `ActorRolloutRefWorker` and `CriticWorker` classes that run as Ray remote actors. Each worker manages its own model shard under FSDP, optimizer states, and learning rate scheduler. The `update_actor()` and `update_critic()` methods handle the actual forward/backward/optimizer step.

## Reward Score Conventions

When adding a new reward function in `verl/utils/reward_score/`:
1. Implement `compute_score(data_source, solution_str, ground_truth, extra_info) → float`
2. Register it in `__init__.py`'s `default_compute_score()` dispatch table
3. For batch processing, add to `math_batch.py` or create a `BatchRewardManager`-compatible function

## NPU/Ascend Support

`verl/__init__.py` patches Torch for Ascend NPU at import time. NPU-specific tests are in `tests/special_npu/`. Use `requirements-npu.txt` for NPU dependencies. Training on NPU uses `--npu` flags in configs under `verl/trainer/config/npu_profile/`.
