#!/usr/bin/env bash
# Reasoning RL (math/code/logic/STEM, rule-verifiable rewards) | Qwen3-4B | DAPO | FSDP2 | vLLM
#
#   FULLY ASYNC variant — Trainer and Rollouter run on DISAGGREGATED resources
#   and train/generate in parallel (verl.experimental.fully_async_policy, see
#   docs/advance/fully_async.md). Rollouter streams samples into a MessageQueue;
#   Trainer fetches require_batches*ppo_mini_batch_size prompts at a time and
#   NCCL-syncs weights back every trigger_parameter_sync_step fetches.
#
#   ASCEND NPU edition — single node, 16 NPUs, split 8 (Trainer) : 8 (Rollouter).
#   NPU-specific deltas vs a GPU run (from the repo's NPU fully-async recipes
#   verl/experimental/fully_async_policy/shell/*_npu.sh):
#     * trainer.device=npu (auto_set_device would also detect it; explicit here).
#     * actor_rollout_ref.actor.use_torch_compile=False — the config default is
#       True, which is CUDA-oriented; keep it off on Ascend.
#     * actor_rollout_ref.nccl_timeout raised to 7200s — HCCL init/collectives
#       are slower to time out under load; the 600s default is too tight.
#     * actor_rollout_ref.rollout.enable_sleep_mode=False — sleep/wake on
#       vllm-ascend is version-dependent; the NPU DAPO recipe disables it.
#       If partial_rollout behaves oddly, try toggling this (the geo3k NPU
#       recipe leaves it at the default True).
#     * rollout.max_num_seqs / max_num_batched_tokens set explicitly, as
#       recommended for vllm-ascend.
#   Requires: torch_npu + vllm-ascend environment (see
#   docs/ascend_tutorial/model_support/examples/ascend_vllm_best_practices.rst).
#
# Differences vs the synchronous run_qwen3_4b_reasoning_rl_dapo.sh:
#   * Entry point is verl.experimental.fully_async_policy.fully_async_main
#     (config fully_async_ppo_trainer.yaml), not verl.trainer.main_ppo.
#   * Resources are split: NNODES_TRAIN/NGPUS_TRAIN for the Trainer,
#     NNODES_ROLLOUT/NGPUS_ROLLOUT for the Rollouter (no colocate hybrid engine,
#     actor_rollout_ref.hybrid_engine=False).
#   * data.train_batch_size is INERT (must be 0); the sample budget is
#     rollout.total_rollout_steps (= train_batch_size-equivalent * TOTAL_STEPS).
#   * old_log_prob comes from the ROLLOUTER (use_rollout_log_probs=True +
#     algorithm.rollout_correction.bypass_mode=True, the yaml default). Under
#     staleness_threshold>0 those log-probs are up to ~1 param version old;
#     the kl_cov KL penalty is computed against them (approximation inherent
#     to off-policy async training).
#   * NOT SUPPORTED here (streaming pipeline has no equivalent):
#       - tiered hard-sample replay (HardReplaySampler / REASONING_RL_HARD_REPLAY)
#       - DAPO dynamic sampling (algorithm.filter_groups): uniform-reward groups
#         are NOT filtered/refilled; every generated group is trained on once.
#     If either is load-bearing for the run, use the sync script instead.
#   * test_freq / save_freq are in PARAM-VERSION units. With the defaults
#     (require_batches=4, ppo_mini_batch_size=64, trigger_parameter_sync_step=1)
#     one param version consumes 4*64*1 = 256 prompts = one sync step, so
#     frequencies and global_step_N checkpoints stay numerically comparable
#     to the sync runs, and sync <-> fully-async checkpoints are
#     interchangeable (same actor/ layout) for the response-length curriculum.
#
# Mode (docs/advance/fully_async.md "Supported Modes"): defaults give
#   "async stream pipeline with partial rollout" (mode 4, fastest):
#     staleness_threshold=0.5 + partial_rollout=True — up to 50% of a fetched
#     training batch may be stale (generated under an older param version).
#   1 is the max useful staleness (>1 buys no extra speed, per fully_async.md);
#   raise STALENESS_THRESHOLD toward 1 for more throughput, lower (e.g. 0.1)
#   if training looks unstable. Set STALENESS_THRESHOLD=0 PARTIAL_ROLLOUT=False
#   for the synchronous streaming variant (mode 2) when debugging accuracy
#   regressions.
#
# Optional env knobs (on top of the sync script's REASONING_RL_SYSTEM_PROMPT /
# SANDBOX_FUSION_URL / TRAIN_FILES / VAL_FILES / RESUME_MODE / RESUME_PATH):
#   NNODES_TRAIN=1 NNODES_ROLLOUT=1         disaggregated node split
#   NGPUS_TRAIN=8 NGPUS_ROLLOUT=8           NPUs per side (8+8 = the 16 on node)
#   MAX_NUM_SEQS=128 ENFORCE_EAGER=False    vllm-ascend concurrency / graph mode
#   NCCL_TIMEOUT=7200                       HCCL collective timeout (s)
#   TOTAL_STEPS=400                         total_rollout_steps = 256 * TOTAL_STEPS
#   STALENESS_THRESHOLD=0.5                 max fraction of stale samples (0.5 = 50%;
#                                           1 = 100% most aggressive; lower if unstable)
#   REQUIRE_BATCHES=4                       mini-batches fetched per training round
#   TRIGGER_PARAMETER_SYNC_STEP=1           rounds between weight syncs
#   PARTIAL_ROLLOUT=True                    interrupt+resume in-flight gens on sync
#   USE_TRAINER_DO_VALIDATE=0               1 = validate on hybrid replicas using
#                                           trainer GPUs (faster val, but trainer
#                                           must then share GPU memory: keep
#                                           ROLLOUT_GPU_MEM_UTIL low, e.g. 0.3-0.5)
#   VAL_N=1                                 validation samples per prompt (pass@N)
#   FSDP_STRATEGY=fsdp2                     trainer FSDP strategy

set -xeuo pipefail

# vLLM V1 engine is required for async (server-mode) rollout.
export VLLM_USE_V1=1

# Ascend/HCCL runtime knobs (single node). Overridable from the environment.
# HCCL collectives + vllm-ascend engine bring-up can exceed the tight defaults
# on long runs; align with docs/ascend_tutorial recommendations.
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
# Ray must not overwrite the NPU visibility set by the driver/launcher.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
# All 16 NPUs on the node (Trainer pool takes 8, Rollouter pool takes 8).
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
# Disaggregated resource pools: single node, 16 NPUs -> 8 Trainer : 8 Rollouter.
# Ideal split makes rollout time ~ train time; tune from the
# rollouter/idle_ratio vs trainer/idle_ratio metrics.
NNODES_TRAIN=${NNODES_TRAIN:-1}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NGPUS_TRAIN=${NGPUS_TRAIN:-8}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-8}

DATA_DIR=${DATA_DIR:-$HOME/data/reasoning_rl/final}
# Data-file hot-swap seam for mid-training dataset changes (e.g. injecting the
# instruction-following domain). Same semantics as the sync script:
#   TRAIN_FILES="['$HOME/data/reasoning_rl/final_if/train.parquet']" \
#   VAL_FILES="['$HOME/data/reasoning_rl/final_if/val.parquet']" \
#   RESUME_MODE=resume_path RESUME_PATH=<ckpt_dir> bash run_qwen3_4b_reasoning_rl_fully_async.sh
train_files=${TRAIN_FILES:-"['$DATA_DIR/train.parquet']"}
val_files=${VAL_FILES:-"['$DATA_DIR/val.parquet']"}

# Resume knobs. Fully-async checkpoints carry the actor weights plus the
# rollouter dataloader state (data.pt), so resume_mode=auto/resume_path both
# restore the data cursor. Note: in-flight (queued) samples are lost on save.
RESUME_MODE=${RESUME_MODE:-}
RESUME_PATH=${RESUME_PATH:-}

# Only used to derive the sample budget and to keep param-version == sync-step
# accounting (see header). Not passed to the config (train_batch_size stays 0).
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
# Dataset prompt caps are far below ceiling (math ~2k, code up to ~4k per DESIGN.md
# section 8); 8192 is enough headroom so nothing gets truncated.
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
# response curriculum: 8192 -> 16384 -> 24576 (raise between runs; DESIGN.md
# section 8). NOTE: Qwen3-4B's native context is 32768, so keep
# max_prompt_length + max_response_length <= 32768 or vLLM silently clamps.
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
# dynamic-bsz packing budget MUST cover the longest single (prompt+response) sequence
# or the tail gets dropped. Default = max_prompt + max_response.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}
# clip-higher: keep epsilon_low at 0.2, raise epsilon_high (DAPO). NOTE: as in
# the sync script, loss_mode=kl_cov does not consume these clip ratios.
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.4}

# kl_cov (PRIME-RL): KL-penalize the top-covariance tokens to stop entropy
# collapse. Under fully async the KL reference is the rollout log-prob, which
# may be up to ~1 param version stale (staleness_threshold).
kl_cov_ratio=${KL_COV_RATIO:-0.0005}
kl_cov_coef=${KL_COV_COEF:-0.1}

rollout_tp=${ROLLOUT_TP:-1}
# Rollout NPUs are dedicated here (no colocate), so 0.8 is safe. Lower to
# 0.3-0.5 if USE_TRAINER_DO_VALIDATE=1 (hybrid replicas share trainer NPUs).
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
rollout_n=${ROLLOUT_N:-16}
temperature=${TEMPERATURE:-1.0}
val_n=${VAL_N:-1}
# vllm-ascend tuning (see NPU recipes): cap concurrent seqs per replica and
# give the batched-token budget explicitly. enforce_eager=False keeps ACL
# graph mode on; set ENFORCE_EAGER=True to debug graph-mode issues.
max_num_seqs=${MAX_NUM_SEQS:-128}
enforce_eager=${ENFORCE_EAGER:-False}
# HCCL collective timeout (s); config default 600 is too tight for Ascend.
nccl_timeout=${NCCL_TIMEOUT:-7200}

# Sample budget: total_rollout_steps is counted in PROMPTS (one prompt ->
# rollout_n trajectories). total_epochs only sets the dataloader replay
# runway; the explicit total_rollout_steps cap stops the run first.
total_steps=${TOTAL_STEPS:-400}
total_rollout_steps=$((train_batch_size * total_steps))
total_epochs=${TOTAL_EPOCHS:-10}

# In param-version units (1 param version = require_batches * ppo_mini_batch_size
# * trigger_parameter_sync_step prompts = 256 with the defaults below).
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-10}

# Fully async knobs (docs/advance/fully_async.md "Parameter Description").
# staleness_threshold: fraction of a training batch that may be stale (samples
# generated under an older param version). 0.5 = up to 50% stale; 1 is the
# max useful value (>1 buys no extra speedup, per fully_async.md).
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
require_batches=${REQUIRE_BATCHES:-4}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}
use_trainer_do_validate=${USE_TRAINER_DO_VALIDATE:-0}

fsdp_strategy=${FSDP_STRATEGY:-fsdp2}

# Optional system prompt injected into every train/val prompt (dataset level).
# Empty (default) leaves prompts untouched.
system_prompt=${REASONING_RL_SYSTEM_PROMPT:-}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROJECT_NAME=${PROJECT_NAME:-verl_reasoning_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_reasoning_rl_fully_async_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    # In fully async, train_batch_size is inert (must be 0) and generation is
    # streamed one prompt at a time (gen_batch_size=1).
    data.train_batch_size=0
    data.gen_batch_size=1
    # AgentLoop server mode consumes raw chat and applies the template itself.
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
    data.filter_overlong_prompts=True
    # Custom dataset: REASONING_RL_SYSTEM_PROMPT injection (pass-through
    # otherwise). create_rl_dataset honors data.custom_cls in the Rollouter.
    data.custom_cls.path="$REPO_ROOT/examples/reasoning_rl/reasoning_rl_dataset.py"
    data.custom_cls.name=ReasoningRLDataset
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    # clip-higher (DAPO): epsilon_low fixed, epsilon_high raised.
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    # token-level policy gradient loss (DAPO).
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    # kl_cov policy loss: KL-penalize the top-covariance tokens to stop entropy
    # from collapsing (reinforces entropy_coeff=0 above).
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=${kl_cov_ratio}
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=${kl_cov_coef}
    # NOTE: actor.strategy is the canonical FSDP-version switch — FSDPActorConfig
    #.__post_init__ copies it onto engine.strategy (overwriting whatever sits in
    # fsdp_config.strategy). fsdp2 is validated on Ascend (geo3k NPU recipe).
    actor_rollout_ref.actor.strategy=${fsdp_strategy}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # Overlap next-layer param all-gather with current-layer compute. FSDP2 path
    # uses set_modules_to_forward_prefetch (depth=1, mirrors FSDP1); validated on
    # Ascend in the geo3k NPU recipe. No-op on torch<2.5 (guarded by hasattr).
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    # Config default is True (CUDA-oriented); torch.compile is kept off on NPU.
    actor_rollout_ref.actor.use_torch_compile=False
    # HCCL collective timeout (s); default 600 is too tight on Ascend.
    actor_rollout_ref.nccl_timeout=${nccl_timeout}
    # Only persist weights + bookkeeping (global_step/RNG); skip optimizer
    # state to save ~half the checkpoint disk.
    actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]'
    # Fully async: Trainer and Rollouter are disaggregated (no hybrid engine).
    # old_log_prob comes from the ROLLOUTER — both flags below are ON:
    #   use_rollout_log_probs=True  -> actor consumes rollout-returned log-probs
    #   bypass_mode=True            -> Trainer never recomputes old_log_prob
    # (bypass_mode is also the default in fully_async_ppo_trainer.yaml; pinned
    # explicitly here.) Under staleness_threshold=1 these log-probs can be up
    # to ~1 param version old — the intended off-policy trade-off.
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.use_rollout_log_probs=True
    algorithm.rollout_correction.bypass_mode=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    # Server-mode (AgentLoop) rollout is mandatory for fully async.
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${temperature}
    # pure temperature sampling: top_k=-1 and top_p=1.0 both disable (no topk/topp).
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # vllm-ascend knobs (NPU recipes): explicit concurrency/token budgets and
    # ACL graph mode; sleep mode off (see header note).
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager}
    +actor_rollout_ref.rollout.enable_sleep_mode=False  # key absent in trainer-side rollout schema, needs +
    # REQUIRED by fully async: Rollouter must return per-token log-probs
    # (asserted in FullyAsyncRollouter._validate_config).
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
)

REWARD=(
    reward.custom_reward_function.path="$REPO_ROOT/examples/reasoning_rl/reward/compute_score.py"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)

# sandbox-fusion URL rides through reward_kwargs (merged into every
# compute_score call by verl's _call_with_kwargs wrapper).
if [ -n "${SANDBOX_FUSION_URL:-}" ]; then
    REWARD+=(
        "+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=${SANDBOX_FUSION_URL}"
    )
fi

TRAINER=(
    # Ascend NPU platform (auto_set_device would also detect it; be explicit).
    trainer.device=npu
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    # Trainer-side resources (the Rollouter pool is configured in ASYNC below).
    trainer.n_gpus_per_node=${NGPUS_TRAIN}
    trainer.nnodes=${NNODES_TRAIN}
    trainer.val_before_train=True
    # save_freq / test_freq are in param-version units (see header). Validation
    # runs on the Rollouter unless USE_TRAINER_DO_VALIDATE=1.
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    # Only sets the dataloader replay runway; total_rollout_steps stops the run.
    trainer.total_epochs=${total_epochs}
)

ASYNC=(
    # Rollouter-side resources (top-level rollout.*; fully_async_main mirrors
    # them into actor_rollout_ref.rollout).
    rollout.nnodes=${NNODES_ROLLOUT}
    rollout.n_gpus_per_node=${NGPUS_ROLLOUT}
    # Total prompts to generate across the whole run (prompt units).
    rollout.total_rollout_steps=${total_rollout_steps}
    # Fully async scheduling knobs. Defaults: mode 4 (async stream pipeline
    # with partial rollout) at max staleness (threshold=1, see header).
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.require_batches=${require_batches}
    async_training.partial_rollout=${partial_rollout}
    async_training.use_trainer_do_validate=$([ "$use_trainer_do_validate" = "1" ] && echo True || echo False)
)

# Mid-training resume: continue from a checkpoint with the new data mix.
if [ -n "$RESUME_MODE" ]; then
    TRAINER+=(
        trainer.resume_mode=${RESUME_MODE}
    )
    if [ -n "$RESUME_PATH" ]; then
        TRAINER+=(
            trainer.resume_from_path=${RESUME_PATH}
        )
    fi
fi

# System prompt injection (ReasoningRLDataset prepends it to every prompt at
# load time). The dataset is constructed inside the remote FullyAsyncTaskRunner,
# so the value must ride the Ray runtime env to reach it.
if [ -n "$system_prompt" ]; then
    export REASONING_RL_SYSTEM_PROMPT="$system_prompt"
    DATA+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.REASONING_RL_SYSTEM_PROMPT=\"$system_prompt\""
    )
fi

if [ "${REASONING_RL_HARD_REPLAY:-0}" = "1" ]; then
    echo "Fully async does not support hard replay / filter_groups (streaming pipeline)." >&2
    echo "Unset REASONING_RL_HARD_REPLAY, or use run_qwen3_4b_reasoning_rl_dapo.sh instead." >&2
    exit 1
fi

########################### launch ###########################
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC[@]}" \
    "$@"
