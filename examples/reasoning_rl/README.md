# Reasoning RL 数据处理执行手册

> 面向执行者（人或 agent）的 step-by-step 手册。设计原理见 [DESIGN.md](DESIGN.md)，
> 本手册只讲"怎么跑、产出什么、失败怎么办"。严格按顺序执行，每步有验证标准。

---

## 0. 前置条件

| 项 | 要求 | 验证命令 |
|---|---|---|
| Python | 3.10+，`datasets` / `numpy` / `requests` 可用 | `python3 -c "import datasets, numpy, requests"` |
| HF 网络 | 能访问 huggingface.co（GPQA 为 gated，需 `HF_TOKEN`） | `python3 -c "from huggingface_hub import whoami; print(whoami())"` |
| embedding 服务（仅 Step 4 需要） | 本机部署 Qwen3-Embedding-8B | 见 Step 4 |
| 磁盘 | ~50GB（原始集 + parquet + embedding 流式处理） | `df -h ~` |

工作目录：仓库根目录 `/home/charles/workspace/verl`。所有脚本用 `python3` 运行。

---

## 1. 全流程一览

```
Step 1  数学域转换   to_parquet_math.py   → train_math.parquet
Step 2  代码域转换   to_parquet_code.py   → train_code.parquet
Step 3  逻辑域转换   to_parquet_logic.py  → train_logic.parquet (+ eval_seeds.json)
Step 4  STEM 域转换  to_parquet_stem.py   → train_stem.parquet
Step 4b IF 域转换   to_parquet_if.py     → train_if.parquet  (可选，训练中程注入)
Step 5  验证集准备   (手动下载/整理)       → val/*.parquet|jsonl
Step 6  去污染       decontaminate.py     → *.decontaminated.parquet + 审计 CSV + report.json
Step 7  难度预筛     difficulty_tag.py    → *_tagged.parquet (extra_info.pass_rate 回写)
Step 8  混合配比     mix.py               → final/train.parquet + final/val.parquet
Step 9  训练         run_qwen3_4b_reasoning_rl_dapo.sh
```

数据落盘统一在 `~/data/reasoning_rl/` 下（可用 `--local_save_dir` / `--out_dir` 改）。

---

## 2. Step 1：数学域转换

```bash
python3 examples/reasoning_rl/scripts/to_parquet_math.py \
    --local_save_dir ~/data/reasoning_rl/math
```

- **自动下载**：`SynthLabsAI/Big-Math-RL-Verified` + `BytedTsinghua-SIA/DAPO-Math-17k`。
  已有本地副本时用 `--big_math_path` / `--dapo_path` 指向本地路径。
- **处理内容**：统一 prompt 指令（`\boxed{}` 收尾）→ DAPO 优先的 exact 去重 →
  MinHash 近重去重（词级 5-gram，Jaccard≥0.6）。
- **产出**：`~/data/reasoning_rl/math/train_math.parquet`
- **验证标准**：日志末尾打印行数与来源构成（`N dapo + M bigmath`）；
  N 应接近 1.7 万，M 应在 20 万量级。若 M 远小于此，检查去重日志是否异常。
- **调试**：`--max_samples 200` 小样本试跑；`--no_minhash` 跳过近重去重加速。

## 3. Step 2：代码域转换

```bash
python3 examples/reasoning_rl/scripts/to_parquet_code.py \
    --local_save_dir ~/data/reasoning_rl/code
```

- **自动下载**：DeepCoder 训练集（骨架）+ `deepmind/code_contests` train + `codeparrot/apps` train。
  本地副本用 `--deepcoder_path` / `--codecontests_path` / `--apps_path`。
- **处理内容**：测试用例统一为 `{"inputs","outputs"(,"fn_name")}`（每题 ≤20 条，
  0 条弃题）→ 测试哈希去重 → 题面 exact + MinHash 去重。
- **产出**：`~/data/reasoning_rl/code/train_code.parquet`
- **验证标准**：日志逐源打印 `loaded N rows`，末尾打印来源计数
  （deepcoder 约 2 万量级、codecontests 约 1 万、apps 数千）。`skipped` 数很大是正常的
  （APPS/TACO 系相当部分题无测试用例），但任一源 loaded=0 必须排查。

## 4. Step 3：验证集准备（手动）

去污染的比对对象。放入 `~/data/reasoning_rl/val/`，每个集合一个 parquet 或 jsonl，
**至少包含 `problem` / `question` / `content` 之一列**（否则用 `--val_text_field` 指定）。

| 验证集 | 推荐来源 | 用途 |
|---|---|---|
| AIME 2024/2025(/2026) | HF 搜 `aime_2024` / `aime_2025`（如 HuggingFaceH4/aime_2024） | 数学去污染比对 + 最终 eval |
| MATH 官方 test split | `DigitalLearningGmbH/MATH-lighteval` 的 test split | 同上 |
| GPQA / GPQA-Diamond | `Idavidrein/gpqa`（gated，需 token） | STEM 去污染比对 + eval |
| SuperGPQA | `m-a-p/SuperGPQA` | STEM 比对 |
| SciBench | `xw27/scibench` | STEM 比对 |
| HLE（仅 STEM 题） | `cais/hle`（gated）按学科过滤 | STEM 比对 |
| LiveCodeBench 2024.8+ 段 | `livecodebench/code_generation_lite`，按 `contest_date` 过滤 | 代码 eval（不进去污染比对也行，因训练池已按时间切） |

**红线**：验证集只用官方 test/eval split；AIME 全部年份留作验证，不进训练。

## 5. Step 4：去污染（全自动，无需人工抽查）

先起 embedding 服务（单卡即可）：

```bash
vllm serve Qwen/Qwen3-Embedding-8B --task embed --port 8001
# 验证：curl http://127.0.0.1:8001/v1/models
```

数学池（对 AIME + MATH test）：

```bash
python3 examples/reasoning_rl/scripts/decontaminate.py \
    --train_parquet ~/data/reasoning_rl/math/train_math.parquet \
    --val_file ~/data/reasoning_rl/val/aime.parquet \
    --val_file ~/data/reasoning_rl/val/math_test.parquet \
    --out_dir ~/data/reasoning_rl/math/decontaminated
```

STEM 池就绪后同理（比对 GPQA + SuperGPQA + SciBench + HLE），`--val_file` 可重复传。

- **机制**：Stage A 词级 8-gram 重叠即删；Stage B embedding 余弦，阈值由
  正/负对照组自动标定（Youden's J，带 0.80–0.90 安全带）。固定阈值用 `--threshold 0.85`。
- **产出**（在 `--out_dir`）：
  - `*.decontaminated.parquet` —— 训练可用的干净池；
  - `*.dropped.csv` —— 每条删除的审计记录（train 题目前 200 字、命中 val 索引、cosine、阶段）；
  - `*.report.json` —— 删除数、标定阈值、保留数。
- **验证标准**：report 中 `threshold` 落在 0.80–0.90；数学池 embedding 阶段删除率
  通常在 0.1%–2%，若 >5% 需抽查 dropped.csv 确认阈值是否过激。

## 6. Step 3/4：逻辑与 STEM 域转换

逻辑域（DESIGN.md §2.3）：

```bash
python3 examples/reasoning_rl/scripts/to_parquet_logic.py \
    --local_save_dir ~/data/reasoning_rl/logic
```

- **自动下载**：`MiniMaxAI/SynLogic`（easy+hard 两个 config 的官方 train split）+
  `BytedTsinghua-SIA/Enigmata-Data`（按 36 个 task 的 `<task>/en/train.jsonl` 逐文件加载，
  整库 schema cast 会失败，勿改为一次性 `load_dataset`）。
- **Reasoning Gym**：默认每 task 生成 1000 条（`--rgym_per_task 0` 关闭）；seed 硬切分
  train ∈ [0, 10⁶)，eval 三元组登记到 `<local_save_dir>/eval_seeds.json`，每条样本
  `extra_info.seed` 必填。
- **产出**：`train_logic.parquet`（ground_truth 统一为 `{"answer","task"}` JSON）。
- **调试**：`--no_synlogic` / `--no_enigmata` 单源试跑；`--enigmata_max_per_task 30` 限流。

STEM 域（DESIGN.md §2.4）：

```bash
python3 examples/reasoning_rl/scripts/to_parquet_stem.py \
    --local_save_dir ~/data/reasoning_rl/stem
```

- **自动下载**：`MiniByte-666/Dr.SCI` 的 `Dr_SCI_verifiable.parquet`（约 41.5 万条，
  全部 `match_rule=True`；同仓库的 `Dr_SCI_open-ended.parquet` 是 rubric/LLM-judge 题，
  不可规则验证，**不入库**）。
- **大池子**：461K 全量混合时占比会稀释其他域，可用 `--max_rows 150000` 截流，
  或在 Step 8 由 mix.py 按比例采样（推荐后者，池子大不影响）。
- **产出**：`train_stem.parquet`；prompt 自带 `\boxed{}` 指令，原样保留。

## 6b. 指令遵循域（训练中程注入，可选）

`nvidia/Nemotron-RL-instruction_following`（约 4.6 万条，WildChat-1M 提示 +
Open-Instruct / IFBench 可验证约束，最多 5 条/样本）。该域设计为**训练中程混入**，
不参与首发 v1 配比，跑一段时间 reasoning 后按比例注入提升精确指令遵循。

```bash
python3 examples/reasoning_rl/scripts/to_parquet_if.py \
    --local_save_dir ~/data/reasoning_rl/if
```

- **自动下载**：`nvidia/Nemotron-RL-instruction_following`（`--if_path` 可指本地
  jsonl/parquet 镜像）。
- **schema 兼容**：loader 同时容忍三种约束写法——平行列表
  （`instruction_id_list=[...]` + `kwargs=[...]`）、约束字典列表、以及 kwargs 为
  JSON 字符串；`prompt` 可为 chat 消息列表或纯字符串。空 prompt / 无约束样本自动跳过。
- **ground_truth**：非单一答案，而是 `{"constraints": [{"id","kwargs"}, ...]}` JSON；
  reward 侧逐条校验，全部满足才给 1.0（NeMo-Gym 约定）。
- **产出**：`train_if.parquet`（`data_source=if_nemotron`、`ability=if`、`domain=if`）。

混合时按比例注入（`--if_ratio` 默认 0 = 关闭；其余域按比例缩放到 1.0）：

```bash
python3 examples/reasoning_rl/scripts/mix.py \
    --input_dir ~/data/reasoning_rl \
    --output_dir ~/data/reasoning_rl/final_if \
    --if_ratio 0.10   # math 0.405 / code 0.225 / logic 0.135 / stem 0.135 / if 0.10
```

**中程接续训练**（不动 DATA_DIR，直接指向新 mix）：

```bash
TRAIN_FILES="['$HOME/data/reasoning_rl/final_if/train.parquet']" \
VAL_FILES="['$HOME/data/reasoning_rl/final_if/val.parquet']" \
RESUME_MODE=resume_path RESUME_PATH=<checkpoint_dir> \
bash examples/reasoning_rl/run_qwen3_4b_reasoning_rl_dapo.sh
```

## 7. Step 7/8：难度预筛与混合配比

pass@k 预筛（DESIGN.md §4，可选但推荐；需要 GPU + vLLM）：

```bash
python3 examples/reasoning_rl/scripts/difficulty_tag.py \
    --input ~/data/reasoning_rl/logic/train_logic.parquet \
    --output ~/data/reasoning_rl/logic/train_logic_tagged.parquet \
    --model Qwen/Qwen3-4B --k 8
```

- 数学/逻辑 `--k 8`，代码 `--k 4` 且必须传 `--sandbox_fusion_url`（本地执行太贵）。
- 行为：`pass_rate` 回写 `extra_info`；全对题默认丢弃（零梯度）；全错题保留 7.5%
  （`--keep_zero_frac`）供 hard replay 探索；预算不足先 `--subsample_frac 0.33`。

混合（DESIGN.md §5）：

```bash
python3 examples/reasoning_rl/scripts/mix.py \
    --input_dir ~/data/reasoning_rl \
    --output_dir ~/data/reasoning_rl/final
```

- 按 math 45% / code 25% / logic 15% / STEM 15% 分层采样，先切 256 条 val（分层、
  每域 ≤10%），再从余量按比例取 train；某域池子不足时带告警有放回采样。
- **产出**：`final/train.parquet` + `final/val.parquet` + `mix_stats.json`。
- 该 val 只是 in-distribution 冒烟验证；正式评估用 DESIGN.md §7 的固定协议。

## 8. Step 9：训练

```bash
# 先决条件：sandbox-fusion 服务已部署（代码域正式训练必需，DESIGN.md §6）
export SANDBOX_FUSION_URL=http://<host>/run_code
bash examples/reasoning_rl/run_qwen3_4b_reasoning_rl_dapo.sh
```

- DAPO 开关：clip-higher 0.28、filter_groups（hard-replay sampler 自持）、token-mean
  loss、无长度惩罚；rollout n=16、temp=1.0；response 长度课程用
  `MAX_RESPONSE_LENGTH=8192/16384/24576` 逐段提升。
- hard replay 默认开（`REASONING_RL_HARD_REPLAY=0` 关闭并退回框架 filter_groups）。
- **冒烟 run**：不配 `SANDBOX_FUSION_URL` 时代码 reward 落到 prime_code 本地执行，
  只建议小规模验证 reward 分布与 filter 比例（DESIGN.md §9 Checklist 第 8 条）。

## 9. 数据契约（下游全部依赖，改动必须同步 DESIGN.md）

**parquet schema**（每行）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `data_source` | str | `{domain}_{dataset}`：`math_bigmath` / `math_dapo` / `code_deepcoder` / `code_codecontests` / `code_apps` / `logic_synlogic` / `logic_enigmata` / `logic_reasoning_gym` / `stem_drsci` / `if_nemotron` |
| `prompt` | list[dict] | `[{"role": "user", "content": ...}]`，单轮 |
| `ability` | str | `math` / `code` / `logic` / `stem` / `if` |
| `reward_model` | dict | `{"style": "rule", "ground_truth": ...}`；数学/STEM 为答案字符串，代码为 `{"inputs","outputs"(,"fn_name")}` JSON，逻辑为 `{"answer","task"}` JSON，IF 为 `{"constraints":[{"id","kwargs"}]}` JSON |
| `extra_info` | dict | 键固定：`split(str)` `index(int)` `task_id(str)` `domain(str)` `source(str)` `difficulty(str)` `prior_solve_rate(float)` `seed(int,-1=无)`；`difficulty_tag.py` 追加 `pass_rate(float)`（mix.py 对未打标池自动补 -1.0） |

**关键约束**：

- `extra_info` 的键与类型在所有域必须完全一致，否则混合时 parquet schema 冲突；
- 代码 prompt 必须要求 ` ```python ` 代码块（verl prime_code 按此提取答案）；
- `task_id` 全局唯一，是 hard replay 池的 dedup key；
- 去污染只删训练题，验证集本身永不修改、永不进训练。

## 10. 常见问题排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Big-Math schema changed: missing columns ...` | HF 数据集字段改名 | 按报错中的 available columns 更新 `to_parquet_math.py` 的 `load_bigmath()` 字段映射 |
| 代码域某源 `loaded 0 rows` | repo id 失效或字段改名 | 先 `datasets.load_dataset` 打印 `column_names`，再修 `to_parquet_code.py` 对应 loader；DeepCoder 失效则在 HF 搜索最新镜像并用 `--deepcoder_path` 指本地 |
| `codeparrot/apps` 加载报错 | 旧版脚本式数据集 | 加 `--trust_remote_code`，或换用 parquet 镜像（如 `RoganInglis/apps`）后用 `--apps_path` |
| Enigmata 整库加载 CastError | 36 个 task jsonl 字段异构 | 已按 task 逐文件加载（`load_enigmata`），勿改回一次性 `load_dataset(ENIGMATA_REPO)` |
| SynLogic `Config name is missing` | 需要 config 名 | 用 `easy` / `hard` 两个 config（loader 已内置） |
| Dr.SCI 全是证明题、无短答案 | 误载 `Dr_SCI_open-ended.parquet` | 必须加载 `Dr_SCI_verifiable.parquet`（loader 已内置 data_files 指定） |
| decontaminate 报 embedding 连接失败 | vLLM 服务未起或端口不对 | 确认 `curl http://127.0.0.1:8001/v1/models`，或 `--embedding_base_url` 指向实际端点 |
| decontaminate 极慢 | embedding 批量太小 | 调大 `--batch_size`（默认 64，A100 可到 256+） |
| MinHash 阶段内存高 | 签名矩阵 128×N×8B（N=27 万约 276MB） | 属正常量级；更大池可分批跑 |
| `datasets` 下载中断 | 网络抖动 | 重跑即可（HF 缓存断点续传），或先 `hf download` 再用本地路径参数 |
| HF 直连超时 | 网络受限 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 训练 step 收尾（导出回放池）报 `KeyError: key "tools" not found in TensorDict with keys ['data_source', 'extra_info', 'raw_prompt', 'reward_model']` | 导出回放样本用的是 tool_rl 的 `ROW_FIELDS`（含 `tools`）；reasoning_rl parquet 没有 `tools` 列，TransferQueue 的 `select_fields` 会**静默丢弃**轨迹里不存在的字段，旧代码却按请求列表逐个索引 | 已修（`examples/tool_rl/hard_replay.py`）：`_tq_fetch_rows` 只回读轨迹实际存在的字段（缺的字段打一条 info 日志），并跳过没有 `raw_prompt` 的导出 |
