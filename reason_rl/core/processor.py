import torch
from transformers import PreTrainedTokenizer

from .types import (
    AdvantageSample,
    PromptBatch,
    Sample,
    TrainingBatch,
)


class DataProcessor:
    """
    Data processor implementing DataProcessorProtocol.

    Handles tokenization and padding for training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        truncate_max_length: int,
    ):
        """
        Initialize the data processor.

        Args:
            tokenizer: HuggingFace tokenizer for encoding/decoding.
            truncate_max_length: Maximum sequence length for training. Sequences longer
                       than this will be truncated from the end.
        """
        self.tokenizer = tokenizer
        self.truncate_max_length = truncate_max_length

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def to_prompt_batch(self, samples: list[Sample]) -> PromptBatch:
        """
        Convert samples to PromptBatch for vLLM inference (no padding).

        Prompts are assumed to be already formatted (chat template applied externally).

        Args:
            samples: List of Sample objects containing prompts.

        Returns:
            PromptBatch with tokenized prompt IDs.
        """
        prompts_token_ids = [self.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"] for sample in samples]
        return PromptBatch(samples=samples, prompts_token_ids=prompts_token_ids)

    def to_training_batch(
        self,
        advantage_batch: list[AdvantageSample],
        truncate_max_length: int | None = None,
    ) -> TrainingBatch:
        """
        Convert list[AdvantageSample] to TrainingBatch with padding.

        - Uses right-padding (pad_sequence default)
        - Truncates from the END if sequence exceeds truncate_max_length

        Args:
            advantage_batch: List of samples with computed advantages.
            truncate_max_length: Optional override for maximum sequence length.
                       If None, uses self.truncate_max_length.

        Returns:
            TrainingBatch with padded tensors ready for training.
        """
        max_len = truncate_max_length or self.truncate_max_length

        batch_result = {
            "input_ids": [],
            "response_mask": [],
            "attention_mask": [],
            "old_log_probs": [],
            "advantages": [],
        }
        for sample in advantage_batch:
            input_ids = torch.tensor(sample.prompt_ids + sample.response_ids, dtype=torch.long)
            response_mask = torch.zeros_like(input_ids)
            response_mask[len(sample.prompt_ids) :] = 1
            attention_mask = torch.ones_like(input_ids)
            old_log_probs = torch.tensor([0.0] * len(sample.prompt_ids) + sample.old_log_probs, dtype=torch.float)

            # Truncate from the end if exceeds max_len
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                response_mask = response_mask[:max_len]
                attention_mask = attention_mask[:max_len]
                old_log_probs = old_log_probs[:max_len]

            batch_result["input_ids"].append(input_ids)
            batch_result["response_mask"].append(response_mask)
            batch_result["attention_mask"].append(attention_mask)
            batch_result["old_log_probs"].append(old_log_probs)
            batch_result["advantages"].append(sample.advantage)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_result["input_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_mask = torch.nn.utils.rnn.pad_sequence(
            batch_result["response_mask"], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_result["attention_mask"], batch_first=True, padding_value=0
        )
        old_log_probs = torch.nn.utils.rnn.pad_sequence(
            batch_result["old_log_probs"], batch_first=True, padding_value=0.0
        )
        advantages = torch.tensor(batch_result["advantages"], dtype=torch.float)

        return TrainingBatch(
            input_ids=input_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            old_log_probs=old_log_probs,
            advantages=advantages,
        )
