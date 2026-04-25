"""
Custom SFT dataset that only computes loss on format tokens in assistant messages.

Format tokens include XML-like tags, JSON structural characters, JSON keys, and
special tokens. Content tokens (reasoning text, tool argument values, response
text) are masked out so their loss does not participate in backpropagation.

This is useful for learning multi-turn agent formats (tool call structure,
reasoning tags, etc.) without overfitting to specific content.

Usage in training script:
    data.custom_cls.path=/path/to/format_only_sft_dataset.py
    data.custom_cls.name=FormatOnlySFTDataset
    data.format_only=true  # optional, defaults to True
"""

import logging
import os
import re
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Default regex patterns identifying format tokens.
# A token whose character span is entirely within a match of any of these
# patterns is treated as a format token (loss_mask=1).
DEFAULT_FORMAT_PATTERNS = [
    # XML-like tags, e.g. thinking/answer/tool_call open and close tags
    r"</?[a-zA-Z_][\w-]*(\s[^>]*)?/?>",
    # JSON structural characters
    r"[{}\[\]:,]",
    # JSON string keys: a quoted string immediately followed by a colon
    # (the colon itself is matched by the structural pattern above)
    r'"[^"]*"\s*(?=:(?!=))',
    # HuggingFace special tokens like <|im_start|>, <|im_end|>
    r"<\|[^|]+\|>",
]


class FormatOnlySFTDataset(MultiTurnSFTDataset):
    """Multi-turn SFT dataset that masks content tokens, keeping only format tokens in the loss.

    Extends ``MultiTurnSFTDataset`` by overriding ``_process_single_message`` so that
    assistant messages only contribute loss on format tokens. Non-assistant messages
    remain fully masked (loss_mask=0), identical to the parent class.

    Config options (read from the ``config`` DictConfig):
        format_only (bool): If True, apply format-only masking to assistant messages.
            Defaults to True. Set to False to fall back to the parent behavior.
        format_patterns (list[str], optional): Custom regex patterns for format tokens.
            If not provided, ``DEFAULT_FORMAT_PATTERNS`` is used.
    """

    def __init__(
        self,
        parquet_files,
        tokenizer: PreTrainedTokenizer,
        config,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(parquet_files, tokenizer, config, processor, max_samples)

        self.format_only = config.get("format_only", True)
        config_patterns = config.get("format_patterns", None)
        self.format_patterns = list(config_patterns) if config_patterns else list(DEFAULT_FORMAT_PATTERNS)
        self._compiled_patterns = [re.compile(p) for p in self.format_patterns]

        if self.format_only:
            logger.warning(
                "FormatOnlySFTDataset is enabled: only format tokens in assistant messages "
                "will contribute to the loss. Content tokens are masked."
            )

    def _find_format_spans(self, text: str) -> list[tuple[int, int]]:
        """Find all character spans in ``text`` that match any format pattern."""
        spans: list[tuple[int, int]] = []
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end()))
        if not spans:
            return []
        # Merge overlapping/adjacent spans
        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _span_within(token_start: int, token_end: int, format_spans: list[tuple[int, int]]) -> bool:
        """Return True if the token span [token_start, token_end) is entirely within a format span."""
        for fmt_start, fmt_end in format_spans:
            if token_start >= fmt_start and token_end <= fmt_end:
                return True
        return False

    def _get_token_offsets(self, content_ids: torch.Tensor) -> tuple[str, list[tuple[int, int]]]:
        """Return (decoded_text, per-token character offsets) for the given token ids.

        Tries the fast path of re-tokenizing the decoded text with ``return_offsets_mapping``
        and verifies that the re-tokenized ids match the originals. If they do not match
        (which can happen at tokenization boundaries), falls back to incremental decoding
        which is always correct but slower.
        """
        content_ids_list = content_ids.tolist()
        content_text = self.tokenizer.decode(content_ids, skip_special_tokens=False)

        # Fast path: re-tokenize and compare
        try:
            encoded = self.tokenizer(
                content_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            if list(encoded.input_ids) == content_ids_list:
                offsets = [(int(s), int(e)) for s, e in encoded.offset_mapping]
                return content_text, offsets
            else:
                logger.debug(
                    "Re-tokenization mismatch for content; falling back to incremental decoding."
                )
        except Exception:
            logger.debug("Re-tokenization failed; falling back to incremental decoding.")

        # Fallback: decode token by token and accumulate character positions
        text = ""
        offsets = []
        for tid in content_ids_list:
            token_text = self.tokenizer.decode([tid], skip_special_tokens=False)
            offsets.append((len(text), len(text) + len(token_text)))
            text += token_text
        return text, offsets

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        # Delegate base tokenization to the parent class
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )

        # Only recompute loss_mask for assistant messages when format_only is enabled
        if not self.format_only or message["role"] != "assistant":
            return input_ids, loss_mask, attention_mask, inputs

        # The generation prompt prefix is already masked by the parent (loss_mask=0).
        # Recompute loss_mask for the remaining content tokens.
        gen_prompt_len = len(self.generation_prompt)
        content_ids = input_ids[gen_prompt_len:]

        new_loss_mask = torch.zeros_like(loss_mask)
        if len(content_ids) == 0:
            return input_ids, new_loss_mask, attention_mask, inputs

        # Map tokens to character spans and find format regions
        content_text, token_offsets = self._get_token_offsets(content_ids)
        format_spans = self._find_format_spans(content_text)

        if not format_spans:
            # No format tokens detected; nothing to train on for this message
            return input_ids, new_loss_mask, attention_mask, inputs

        for i, (token_start, token_end) in enumerate(token_offsets):
            if self._span_within(token_start, token_end, format_spans):
                new_loss_mask[gen_prompt_len + i] = 1

        return input_ids, new_loss_mask, attention_mask, inputs
