
import torch
from transformers import PreTrainedTokenizer

from .types import (
    AdvantageBatch,
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
        max_seq_length: int,
    ):
        """
        Initialize the data processor.

        Args:
            tokenizer: HuggingFace tokenizer for encoding/decoding.
            truncate_max_length: Maximum sequence length for training. Sequences longer
                       than this will be truncated from the end.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

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
        prompts_token_ids = [
            self.tokenizer.encode(sample.prompt, add_special_tokens=False)
            for sample in samples
        ]
        return PromptBatch(samples=samples, prompts_token_ids=prompts_token_ids)

    def to_training_batch(
        self,
        advantage_batch: AdvantageBatch,
        truncate_max_length: int | None = None,
    ) -> TrainingBatch:
        """
        Convert AdvantageBatch to TrainingBatch with padding.

        - Uses left-padding for causal LM training
        - Truncates from the END if sequence exceeds truncate_max_length

        Args:
            advantage_batch: Batch of samples with computed advantages.
            truncate_max_length: Optional override for maximum sequence length.
                       If None, uses self.truncate_max_length.

        Returns:
            TrainingBatch with padded tensors ready for training.
        """
        max_len = (
            truncate_max_length
            if truncate_max_length is not None
            else self.max_seq_length
        )
        samples = advantage_batch.samples
        batch_size = len(samples)
        pad_token_id = self.tokenizer.pad_token_id

        # Initialize tensors
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        response_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        old_log_probs = torch.zeros(batch_size, max_len, dtype=torch.float)
        advantages = torch.zeros(batch_size, dtype=torch.float)

        for i, sample in enumerate(samples):
            # Concatenate prompt + response
            full_seq = sample.prompt_ids + sample.response_ids
            prompt_len = len(sample.prompt_ids)
            resp_len = len(sample.response_ids)
            log_probs = sample.old_log_probs

            # Truncate from the END if too long
            if len(full_seq) > max_len:
                full_seq = full_seq[:max_len]
                # Adjust response length and log_probs if response got truncated
                new_resp_len = max(0, max_len - prompt_len)
                if new_resp_len < resp_len:
                    log_probs = log_probs[:new_resp_len]
                    resp_len = new_resp_len

            seq_len = len(full_seq)
            pad_len = max_len - seq_len

            # Left-pad input_ids
            input_ids[i, pad_len:] = torch.tensor(full_seq, dtype=torch.long)

            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask[i, pad_len:] = 1

            # Response mask: 1 for response tokens only
            response_start = pad_len + prompt_len
            response_mask[i, response_start : response_start + resp_len] = 1

            # Old log probs: fill only response positions
            if len(log_probs) > 0:
                old_log_probs[i, response_start : response_start + len(log_probs)] = (
                    torch.tensor(log_probs, dtype=torch.float)
                )

            # Per-sample advantage
            advantages[i] = sample.advantage

        return TrainingBatch(
            input_ids=input_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            old_log_probs=old_log_probs,
            advantages=advantages,
        )
