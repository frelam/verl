# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Preprocess QA datasets (NQ, TriviaQA, HotpotQA) for agentic search RL.

This produces a parquet dataset where the agent must:

1. Decide whether it already knows the answer (epistemic awareness).
2. If unsure, call the ``search`` tool to retrieve evidence.
3. Verify the retrieved evidence against the question.
4. Submit a final answer with a confidence score.

This covers capability 2 (tool routing), 7 (factuality / calibration), and
the epistemic reward module from the reward design doc.

The script downloads from the PeterJinGo/nq_hotpotqa_train HuggingFace repo
(the same one used by Search-R1) so no separate retrieval server is needed
for the data itself. For the search *tool* during rollout, point the tool
config at your retrieval service (see tool_config_search.yaml).

Run::

    python -m Rl_Specilist.agent.RL.data_preprocess.prepare_qa_search \
        --local_save_dir ~/data/agentic_qa
"""

import argparse
import logging
import os

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a factual question-answering agent. Follow this workflow:\n"
    "1. <think> Assess whether you know the answer with high confidence.\n"
    "2. If you are not confident, call the `search` tool with a good query.\n"
    "3. Read the search results carefully and verify the evidence.\n"
    "4. Call `submit_answer` with your answer and an honest confidence score "
    "(0.0-1.0). Lower confidence if the evidence is weak or conflicting.\n"
    "5. If the tool says your answer is wrong, reformulate your search query "
    "and try again.\n"
    "Do not fabricate citations. Do not claim you searched if you did not. "
    "Always output reasoning inside <think>...</think> before any tool call."
)

USER_CONTENT_PREFIX = (
    "Answer the following question. You must conduct reasoning inside "
    "<think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the "
    "search tool. You can search as many times as you want. If you find no "
    "further external knowledge needed, you can directly provide the answer. "
    "Question: "
)


def process_single_row(row, split_name, row_index, data_source_tag):
    question = row.get("question", "")

    user_content = USER_CONTENT_PREFIX + question
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source_tag,
            }
        },
        "submit_answer": {
            "create_kwargs": {
                "ground_truth": ground_truth if isinstance(ground_truth, str) else str(ground_truth[0]),
                "task_type": "qa",
            }
        },
    }

    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split_name,
        "tools_kwargs": tools_kwargs,
        "task_type": "qa",
    }

    return pd.Series(
        {
            "data_source": data_source_tag,
            "prompt": prompt,
            "ability": "qa",
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main(hf_repo_id: str, local_save_dir: str):
    os.makedirs(local_save_dir, exist_ok=True)
    data_source_tag = "searchR1_" + hf_repo_id.split("/")[-1]

    for split in ["train", "test"]:
        parquet_filename = f"{split}.parquet"
        logger.info(f"Processing {split} split from {hf_repo_id} ...")

        try:
            local_parquet_filepath = hf_hub_download(
                repo_id=hf_repo_id,
                filename=parquet_filename,
                repo_type="dataset",
                local_dir=local_save_dir + "/_raw",
            )
            df_raw = pd.read_parquet(local_parquet_filepath)
            logger.info(f"  Loaded {len(df_raw)} rows")

            def apply_process_row(row, _split=split):
                return process_single_row(row, _split, row.name, data_source_tag)

            df_processed = df_raw.apply(apply_process_row, axis=1)
            output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
            df_processed.to_parquet(output_file_path, index=False)
            logger.info(f"  Saved {len(df_processed)} processed rows to {output_file_path}")

        except EntryNotFoundError:
            logger.warning(f"  {parquet_filename} not found in {hf_repo_id}, skipping")
        except Exception as e:
            logger.error(f"  Error processing {split}: {e}")

    print(f"\nDone! Output: {local_save_dir}")
    print(f"data_source tag: {data_source_tag}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Prepare QA datasets for agentic search RL.")
    parser.add_argument(
        "--hf_repo_id",
        default="PeterJinGo/nq_hotpotqa_train",
        help="HuggingFace dataset repo (Search-R1 format).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/agentic_qa",
        help="Output directory for processed parquet files.",
    )
    args = parser.parse_args()
    main(args.hf_repo_id, os.path.expanduser(args.local_save_dir))
