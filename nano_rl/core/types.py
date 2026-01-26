from dataclasses import dataclass, field

import torch


@dataclass
class Sample:
    """
    Raw data sample format returned by dataset and dataloader.

    Attributes:
        prompt: The input prompt text.
        answer: Optional for sft answer.
        ground_truth: Optional ground truth answer for evaluation.
    """

    prompt: str
    answer: str | None = None
    ground_truth: str | None = None


@dataclass
class PromptBatch:
    """
    Input batch for inference workers.

    Attributes:
        samples: List of Sample objects containing prompts.
        prompts_token_ids: Optional pre-tokenized prompt token IDs.
    """

    samples: list[Sample]
    prompts_token_ids: list[list[int]] | None = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int | slice):
        """Support indexing and slicing operations."""
        if isinstance(idx, slice):
            return PromptBatch(
                samples=self.samples[idx],
                prompts_token_ids=self.prompts_token_ids[idx]
                if self.prompts_token_ids
                else None,
            )

        return PromptBatch(
            samples=[self.samples[idx]],
            prompts_token_ids=[self.prompts_token_ids[idx]]
            if self.prompts_token_ids
            else None,
        )

    def __iter__(self):
        """Enable iteration over individual samples."""
        for i in range(len(self)):
            yield self[i]


@dataclass
class SingleRollout:
    """
    Single rollout result from generation.

    Attributes:
        response_ids: Token IDs of the generated response.
        old_log_probs: Log probabilities of each token under the generation policy.
        response_text: Decoded text of the generated response.
    """

    response_ids: list[int]
    old_log_probs: list[float]
    response_text: str


@dataclass
class GroupRollout:
    """
    Group of rollouts for a single prompt (for GRPO with group_size > 1).

    Attributes:
        sample: The original sample containing the prompt.
        prompt_ids: Token IDs of the prompt.
        items: List of SingleRollout results for this prompt.
    """

    sample: Sample
    prompt_ids: list[int]
    items: list[SingleRollout]


@dataclass
class RolloutBatch:
    """
    Final output from inference workers containing all rollout groups.

    Attributes:
        groups: List of GroupRollout, one per input prompt.
    """

    groups: list[GroupRollout]


@dataclass
class RewardBatch:
    """
    Intermediate result after reward computation, before advantage calculation.

    This type bridges RolloutBatch and TrainingBatch, enabling:
    - Decoupled reward functions (rule-based, model-based, etc.)
    - Group-level advantage normalization for GRPO

    Attributes:
        groups: Original rollout groups (preserved for reference).
        rewards: Nested list of rewards, rewards[prompt_idx][rollout_idx].
    """

    groups: list[GroupRollout]
    rewards: list[list[float]]

    def __post_init__(self):
        """Validate that rewards match the rollout structure."""
        assert len(self.rewards) == len(self.groups), (
            f"Rewards length {len(self.rewards)} != groups length {len(self.groups)}"
        )
        for i, (group, group_rewards) in enumerate(zip(self.groups, self.rewards)):
            assert len(group_rewards) == len(group.items), (
                f"Group {i}: rewards length {len(group_rewards)} != items length {len(group.items)}"
            )


@dataclass
class AdvantageSample:
    """
    Single training sample with computed advantage (before padding).

    Attributes:
        prompt_ids: Token IDs of the prompt.
        response_ids: Token IDs of the generated response.
        old_log_probs: Log probabilities from the rollout policy.
        advantage: Computed advantage value (normalized within group).
    """

    prompt_ids: list[int]
    response_ids: list[int]
    old_log_probs: list[float]
    advantage: float


@dataclass
class TrainingBatch:
    """
    Batch of training data for policy optimization.

    Attributes:
        input_ids (batch_size, seq_len): Token IDs of the input sequence.
        response_mask (batch_size, seq_len): Binary mask indicating response tokens (1) vs prompt tokens (0).
        attention_mask (batch_size, seq_len): Optional attention mask for transformer models.
        old_log_probs (batch_size, seq_len): Log probabilities from the rollout policy (for off-policy training).
        advantages (batch_size,): Computed advantage values for each response.
    """

    input_ids: torch.Tensor
    response_mask: torch.Tensor
    attention_mask: torch.Tensor | None = None
    old_log_probs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        """Support indexing for mini-batch creation."""
        return TrainingBatch(
            input_ids=self.input_ids[idx],
            response_mask=self.response_mask[idx],
            attention_mask=self.attention_mask[idx]
            if self.attention_mask is not None
            else None,
            old_log_probs=self.old_log_probs[idx]
            if self.old_log_probs is not None
            else None,
            advantages=self.advantages[idx] if self.advantages is not None else None,
        )

    def to(self, device):
        """Move all tensors to the specified device."""
        return TrainingBatch(
            input_ids=self.input_ids.to(device),
            response_mask=self.response_mask.to(device),
            attention_mask=self.attention_mask.to(device)
            if self.attention_mask is not None
            else None,
            old_log_probs=self.old_log_probs.to(device)
            if self.old_log_probs is not None
            else None,
            advantages=self.advantages.to(device)
            if self.advantages is not None
            else None,
        )

    def minibatches(self, batch_size: int, shuffle: bool = True):
        """
        Yield minibatches from the full training batch.

        Args:
            batch_size: Size of each minibatch.
            shuffle: Whether to shuffle the data before batching.
        """
        num_samples = len(self)
        indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self[batch_indices]


@dataclass
class StepMetrics:
    """
    Training metrics returned after each optimization step (for logging to WandB).

    Attributes:
        loss: The training loss value.
        learning_rate: Current learning rate (filled by Trainer).
        gradient_norm: Optional gradient norm after clipping.
        entropy: Optional policy entropy for monitoring exploration.
        stats: Additional custom statistics as key-value pairs.
    """

    loss: float
    learning_rate: float | None = None
    gradient_norm: float | None = None
    entropy: float | None = None
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelCheckpoint:
    """
    Checkpoint for synchronizing weights between ModelWorker and RolloutWorker.

    Attributes:
        global_step: Current training step number.
        state_dict: Model state dictionary containing all parameters.
    """

    global_step: int
    state_dict: dict
