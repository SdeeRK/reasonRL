import logging
from collections.abc import Iterator

import ray
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..core.config import RLConfig
from ..core.interfaces import AdvantageComputerProtocol, RewardComputerProtocol
from ..core.processor import DataProcessor
from ..core.types import Sample, StepMetrics
from ..data.dataset import PromptDataset
from ..workers.model_worker import ModelWorker
from ..workers.rollout_worker import RolloutWorker

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO training orchestrator.

    Initializes and coordinates all components:
    - RolloutWorker: Generate responses via vLLM
    - ModelWorker: Train policy via PyTorch
    - RewardComputer / AdvantageComputer: Compute rewards and advantages
    - DataProcessor: Handle tokenization and padding
    """

    def __init__(
        self,
        config: RLConfig,
        train_dataset: PromptDataset,
        reward_computer: RewardComputerProtocol,
        advantage_computer: AdvantageComputerProtocol,
        eval_dataset: PromptDataset | None = None,
    ):
        self.config = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_config.model_name_or_path
        )

        # Initialize DataProcessor
        self.data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            max_seq_length=config.train_config.max_seq_length,
        )

        # Initialize workers (Ray actors) with configurable GPU allocation
        self.model_worker = ModelWorker.options(
            num_gpus=config.model_config.model_num_gpus
        ).remote(
            config.model_config,
            config.train_config,
        )
        self.rollout_worker = RolloutWorker.options(
            num_gpus=config.model_config.rollout_num_gpus
        ).remote(
            config.model_config,
            config.generation_config,
        )

        # Store computers
        self.reward_computer = reward_computer
        self.advantage_computer = advantage_computer

        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_config.train_batch_size,
            shuffle=True,
            collate_fn=list,  # Return List[Sample] directly
        )

        self.eval_dataloader = None
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.train_config.train_batch_size,
                shuffle=False,
                collate_fn=list,
            )

        self.global_step = 0
        self._train_iter: Iterator | None = None

    def _get_next_batch(self) -> list[Sample]:
        """Get next batch from dataloader, cycling if exhausted."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_dataloader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        return batch

    def train(self) -> None:
        """
        Main training loop.

        Runs for n_grpo_steps iterations, each consisting of:
        1. Sample batch from dataloader
        2. Generate rollouts via RolloutWorker
        3. Compute rewards and advantages
        4. Train policy for epochs_per_rollout_batch epochs
        5. Sync weights to RolloutWorker
        """
        train_config = self.config.train_config

        for step in range(train_config.n_grpo_steps):
            self.global_step = step

            # 1. Get batch of samples
            samples = self._get_next_batch()

            # 2. Tokenize for vLLM (no padding)
            prompt_batch = self.data_processor.to_prompt_batch(samples)

            # 3. Generate rollouts (Ray remote call)
            rollout_batch = ray.get(
                self.rollout_worker.generate.remote(
                    prompt_batch, self.config.generation_config
                )
            )

            # 4. Compute rewards
            reward_batch = self.reward_computer(rollout_batch)

            # 5. Compute advantages (group-level normalization)
            advantage_batch = self.advantage_computer.compute(reward_batch)

            # 6. Convert to training tensors (padding)
            training_batch = self.data_processor.to_training_batch(advantage_batch)

            # 7. Inner training loop (epochs per rollout batch)
            for epoch in range(train_config.epochs_per_rollout_batch):
                metrics = ray.get(self.model_worker.train_step.remote(training_batch))
                self._log_metrics(metrics, step, epoch)

            # 8. Sync weights to rollout worker
            checkpoint = ray.get(self.model_worker.get_weights.remote(self.global_step))
            ray.get(self.rollout_worker.update_weights.remote(checkpoint))

            # 9. Periodic evaluation
            if self.eval_dataloader and (step + 1) % 100 == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Step {step + 1} eval: {eval_metrics}")

        logger.info("Training completed!")

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on eval_dataloader."""
        if self.eval_dataloader is None:
            return {}

        total_reward = 0.0
        total_samples = 0

        for samples in self.eval_dataloader:
            prompt_batch = self.data_processor.to_prompt_batch(samples)
            rollout_batch = ray.get(self.rollout_worker.generate.remote(prompt_batch))
            reward_batch = self.reward_computer(rollout_batch)

            for group_rewards in reward_batch.rewards:
                total_reward += sum(group_rewards)
                total_samples += len(group_rewards)

        return {
            "eval/mean_reward": total_reward / max(1, total_samples),
            "eval/total_samples": total_samples,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        ray.get(self.model_worker.save_checkpoint.remote(path))
        logger.info(f"Checkpoint saved to {path}")

    def _log_metrics(self, metrics: StepMetrics, step: int, epoch: int) -> None:
        """Log training metrics."""
        logger.info(
            f"Step {step + 1}/{self.config.train_config.n_grpo_steps} "
            f"Epoch {epoch + 1}/{self.config.train_config.epochs_per_rollout_batch} | "
            f"Loss: {metrics.loss:.4f} | LR: {metrics.learning_rate:.2e}"
        )
