from typing import Protocol, runtime_checkable

from .config import GenerationConfig
from .types import (
    AdvantageBatch,
    ModelCheckpoint,
    PromptBatch,
    RewardBatch,
    RolloutBatch,
    Sample,
    StepMetrics,
    TrainingBatch,
)


@runtime_checkable
class RolloutWorkerProtocol(Protocol):
    """
    Protocol for rollout generation workers (e.g., vLLM inference workers).

    RolloutWorkers are responsible for:
    - Generating responses from prompts using the current policy
    - Updating weights when the policy is updated
    """

    def generate(
        self,
        requests: PromptBatch,
        generation_config: GenerationConfig | None = None,
    ) -> RolloutBatch:
        """
        Generate responses for a batch of prompts.

        Args:
            requests: Batch of prompts to generate responses for.
            generation_config: Optional generation parameters (temperature, top_p, etc.).
                              If None, uses the worker's default configuration.

        Returns:
            RolloutBatch containing generated responses and associated metadata.
        """
        ...

    def update_weights(self, checkpoint: ModelCheckpoint) -> None:
        """
        Update the worker's model weights from a checkpoint.

        This method is called after training steps to synchronize the
        inference model with the updated policy.

        Args:
            checkpoint: Model checkpoint containing the new weights.
        """
        ...


@runtime_checkable
class ModelWorkerProtocol(Protocol):
    """
    Protocol for model training workers (e.g., PyTorch training workers).

    ModelWorkers are responsible for:
    - Performing gradient updates on the policy model
    - Providing current weights for synchronization with rollout workers
    - Saving model checkpoints to disk
    """

    def train_step(self, batch: TrainingBatch) -> StepMetrics:
        """
        Perform a single training step on the given batch.

        Args:
            batch: Training batch containing rollouts and computed advantages.

        Returns:
            StepMetrics containing training statistics (loss, gradients, etc.).
        """
        ...

    def get_weights(self) -> ModelCheckpoint:
        """
        Get the current model weights for synchronization.

        Returns:
            ModelCheckpoint containing the current model state dict.
            If LoRA is enabled, returns merged full model weights.
        """
        ...

    def save_checkpoint(self, path: str) -> None:
        """
        Save the model checkpoint to disk.

        Args:
            path: Directory path to save the checkpoint.
                  If LoRA is enabled, saves the merged full model.
        """
        ...


@runtime_checkable
class RewardComputerProtocol(Protocol):
    """
    Compute rewards for rollout responses.

    Provides two interfaces:
    - __call__: Batch processing, transforms RolloutBatch → RewardBatch
    - compute: Single response reward calculation
    """

    def __call__(self, rollout_batch: RolloutBatch) -> RewardBatch: ...

    def compute(self, response: str, ground_truth: str) -> float: ...


@runtime_checkable
class AdvantageComputerProtocol(Protocol):
    """
    Compute advantages from rewards.

    Transforms RewardBatch → AdvantageBatch by normalizing rewards within groups (for GRPO).
    Does NOT handle padding/tensorization - that's DataProcessor's job.
    """

    def compute(self, reward_batch: RewardBatch) -> AdvantageBatch: ...


@runtime_checkable
class DataProcessorProtocol(Protocol):
    """
    Handle tokenization and data format conversion.

    Responsible for:
    - Converting samples to vLLM-compatible format (tokenize, no padding)
    - Converting advantage batch to training tensors (padding + tensorization)
    """

    def to_prompt_batch(self, samples: list[Sample]) -> PromptBatch:
        """Tokenize samples for vLLM inference (no padding)."""
        ...

    def to_training_batch(
        self,
        advantage_batch: AdvantageBatch,
        max_length: int | None = None,
    ) -> TrainingBatch:
        """Pad and tensorize samples for training."""
        ...
