# Online DPO — Sandbox Agent + Gateway Training

## Architecture

```
┌─── AKernel Sandbox (per-data-point Docker) ──────────┐
│  Dataset specifies image per sample                   │
│  Agent sidecar mounted at /opt/hermes-agent           │
│  Agent runs inside sandbox, calls Gateway via tunnel  │
│  Tools (bash, file-edit) execute in sandbox           │
└──────────────────────────────────────────────────────┘
           │ HTTP (tunnel: 127.0.0.1:38197)
           ▼
┌─── Uni-Agent Gateway (FastAPI) ───────────────────────┐
│  /v1/chat/completions                                 │
│  Tool-call parsing + token-level trajectory capture   │
│  Per-session base_url + reward_info_url               │
└──────────────────────────────────────────────────────┘
           │ chat/completions
           ▼
┌─── vLLM (training model) ────────────────────────────┐
│  Inference engine, weights updated each DPO step      │
└──────────────────────────────────────────────────────┘
           │
           ▼
  Runner posts raw data (task, agent_output, sandbox_results)
           │
           ▼
  custom_reward_function (uni_agent.reward.llm_judge.compute_score)
    ├─ Sandbox test results → accuracy_reward
    ├─ LLM judge (if configured) → process_reward
    └─ Weighted combination → final reward_score
           │
           ▼
  best vs worst pairing → DPO loss → model update
           │
           ▼
  Next rollout — model weights updated ✨
```

**Key components:**

| Component | Location | Role |
|-----------|----------|------|
| Sandbox runners | `uni-agent/examples/blackbox_recipes/` | Create AKernel sandbox, run agent, collect raw output |
| Gateway | `uni-agent/uni_agent/gateway/` | Token-level trajectory capture |
| Reward plugin | `uni_agent/uni_agent/reward/llm_judge.py` | Pluggable `compute_score` — sandbox eval + LLM judge |
| Online DPO config | `config/agent_hermes_gateway.yaml` | Hydra config wiring everything together |
| Judge prompts | `prompts/` + `uni_agent/reward/prompts/` | Dataset-specific scoring rubrics |

---

## Quick Start

### 1. Prerequisites

```bash
# AKernel sandbox (required)
export AKERNEL_SERVER_ADDRESS="x.x.x.x:8888"
export AKERNEL_TOKEN="<your-token>"

# LLM judge API key (optional — only needed for tasks with llm_judge scoring)
export JUDGE_API_KEY=sk-xxx  # or DEEPSEEK_API_KEY

# Model and data
export MODEL_PATH=~/models/Qwen3-4B
export TRAIN_DATA=~/data/online_dpo/prompts/train.parquet
export VAL_DATA=~/data/online_dpo/prompts/val.parquet
```

### 2. Build Tool Images (one-time)

```bash
# Hermes agent sidecar
cd ~/workspace/uni-agent
# See: examples/blackbox_recipes/hermes_agent/Dockerfile.hermes-agent-tool

# Claude Code sidecar
# See: examples/blackbox_recipes/claude_code/Dockerfile.claude-code-tool
```

### 3. Start Training

```bash
bash run_hermes_gateway_dpo.sh <dataset> 8 ~/ckpt/hermes-gateway-dpo
```

---

## Per-Sample Scoring Config

Each dataset row can specify its scoring strategy via `extra_info.tools_kwargs.scoring`:

```python
# SWE-bench: sandbox test results only
{"sandbox_eval": True}

# Subjective tasks: LLM judge only
{"sandbox_eval": False, "llm_judge": True}

# Both: weighted combination
{"sandbox_eval": True, "llm_judge": True, "sandbox_weight": 0.5}
```

The scoring is executed by `uni_agent.reward.llm_judge.compute_score` (verl `custom_reward_function` pattern). This function:
1. Reads sandbox test results from the runner (if `sandbox_eval`)
2. Calls the LLM judge API (if `llm_judge`)
3. Aggregates into a final `reward_score`

---

## Configuration

`config/agent_hermes_gateway.yaml`:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    custom:
      agent_framework:
        agent_runners:
          hermes_agent:
            runner_fqn: examples.blackbox_recipes.hermes_agent.hermes_agent_runner.hermes_agent_runner
            dispatch_mode: ray_task
          claude_code:
            runner_fqn: examples.blackbox_recipes.claude_code.claude_code_runner.claude_code_runner
            dispatch_mode: ray_task
  actor:
    policy_loss:
      loss_mode: dpo
      dpo_beta: 0.1

algorithm:
  use_dpo: true

reward:
  custom_reward_function:
    path: uni_agent.reward.llm_judge    # Pluggable scoring plugin
    name: compute_score
```

---

## Core Modules

### Sandbox Runners (`uni-agent/examples/blackbox_recipes/`)

Scoring-agnostic: they execute the agent, optionally run sandbox tests, and post raw data to the Gateway. Scoring decisions are deferred to `custom_reward_function`.

```
Runner (hermes_agent_runner / claude_code_runner)
  ├─ Create AKernel sandbox (per-sample Docker image)
  ├─ Mount agent sidecar (/opt/hermes-agent or /opt/claude-code)
  ├─ Run agent against Gateway
  │     └─ Agent → Gateway → vLLM
  │     └─ Agent ← Gateway ← assistant reply
  │     └─ Agent → execute tools in sandbox
  │     └─ ... loop until complete or max_turns ...
  ├─ If scoring.sandbox_eval: run tests in sandbox
  ├─ POST raw data → Gateway (task, agent_output, accuracy_reward, scoring)
  └─ Cleanup sandbox
```

### Reward Plugin (`uni_agent/uni_agent/reward/llm_judge.py`)

Verl-compatible `custom_reward_function`.  Called by `RewardLoopWorker` after the runner finishes.

| Function | Purpose | Called by |
|----------|---------|-----------|
| `compute_score()` | Main entry — reads extra_info, applies scoring pipeline | RewardLoopWorker |
| `judge_single()` | Single-trajectory LLM judge | `compute_score` |
| `load_judge_prompt()` | Load dataset-specific rubric | `compute_score` |

Environment:
```bash
export JUDGE_MODEL=deepseek-chat        # Judge model
export JUDGE_BASE_URL=https://api.deepseek.com
export JUDGE_API_KEY=sk-xxx             # API key
export JUDGE_PROMPTS_DIR=/path/to/prompts  # Custom judge prompts dir (optional)
```

### Judge Prompts

Dataset-specific scoring rubrics are loaded from:
1. `JUDGE_PROMPTS_DIR` env var
2. `uni_agent/reward/prompts/` (default)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | — | Training model path (required) |
| `TRAIN_DATA` | — | Training data parquet (required) |
| `VAL_DATA` | — | Validation data parquet (required) |
| `AKERNEL_SERVER_ADDRESS` | — | AKernel sandbox server (required) |
| `AKERNEL_TOKEN` | — | AKernel auth token (required) |
| `N_SAMPLES` | `8` | Rollouts per prompt |
| `AGENT_MAX_TURNS` | `100` | Max agent conversation turns |
| `DPO_BETA` | `0.1` | DPO temperature |
| `ACTOR_LR` | `1e-6` | Actor learning rate |
| `GATEWAY_COUNT` | `1` | Gateway instances |
| `NUM_AGENT_WORKERS` | `8` | Agent worker count |
| `JUDGE_MODEL` | `deepseek-chat` | LLM judge model |
| `JUDGE_BASE_URL` | `https://api.deepseek.com` | Judge API URL |
| `JUDGE_API_KEY` | — | Judge API key (for llm_judge scoring) |
| `HERMES_TOOL_IMAGE` | `...hermes-agent-tool:latest` | Hermes sidecar image |
| `CLAUDE_TOOL_IMAGE` | `...claude-code-tool:latest` | Claude Code sidecar image |
| `HERMES_RUN_TIMEOUT` | `3600` | Hermes agent timeout (seconds) |
| `CLAUDE_RUN_TIMEOUT` | `3600` | Claude Code timeout (seconds) |

---

## Training Metrics

```
actor/dpo_loss              — DPO loss ↓
actor/dpo_accuracy          — Preference accuracy ↑
actor/dpo_chosen_reward     — Chosen sample reward ↑
actor/dpo_rejected_reward   — Rejected sample reward ↓
reward/mean_score           — Average reward ↑
```

---

## Directory Structure

```
Rl_Specilist/agent/online_dpo/
├── config/
│   └── agent_hermes_gateway.yaml      # DPO training config
├── hermes_entrypoint.py               # Hermes agent entrypoint (sandbox)
├── claude_code_entrypoint.py          # Claude Code agent entrypoint (sandbox)
├── custom_hermes_runner.py            # Legacy local-workspace runner
├── custom_claude_runner.py            # Legacy local-workspace runner
├── framework.py                       # Thin re-export of OpenAICompatibleAgentFramework
├── reward/
│   ├── __init__.py
│   └── llm_judge.py                   # Thin re-export → uni_agent.reward.llm_judge
├── tests/
│   ├── __init__.py
│   └── test_hermes_entrypoint.py
├── setup/
│   ├── install.sh
│   └── download_data.sh
├── prompts/                           # Judge prompt templates
│   ├── coding_judge.txt
│   ├── math_judge.txt
│   └── terminal_judge.txt
├── run_hermes_gateway_dpo.sh          # Training launch script
└── README.md
```

**Scoring logic lives in:** `uni_agent/uni_agent/reward/llm_judge.py`  
**Sandbox runners live in:** `uni_agent/examples/blackbox_recipes/`

---

## Adding a New Scoring Method

1. Write a new `compute_score` function following the verl contract:
   ```python
   def compute_score(data_source, solution_str, ground_truth, extra_info) -> dict:
       return {"score": ..., "reward_extra_info": {...}}
   ```

2. Point the config to it:
   ```yaml
   reward:
     custom_reward_function:
       path: your_module
       name: your_compute_score
   ```

No runner changes needed — the runner always posts raw data (task, agent_output, sandbox results). The `custom_reward_function` only needs to read `extra_info`.
